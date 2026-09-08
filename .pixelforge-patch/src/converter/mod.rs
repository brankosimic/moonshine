//! GPU-accelerated color format conversion using Vulkan compute shaders.
//!
//! This module provides efficient color space conversion on the GPU,
//! converting from RGB/BGR formats to YUV for video encoding.
//!
//! The output remains on the GPU as a `vk::Image` to avoid unnecessary
//! CPU round-trips when used with a GPU-based video encoder.

mod pipeline;

use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use tracing::debug;

/// Color space for RGB→YUV conversion matrix selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// BT.709 (standard SDR). Used for all HD/UHD SDR content.
    #[default]
    Bt709,
    /// BT.2020 (HDR / wide color gamut).
    Bt2020,
    /// sRGB input converted to BT.2020 + PQ output.
    /// Used for SDR content during HDR streaming sessions.
    /// Applies: inverse sRGB EOTF → BT.709→BT.2020 gamut mapping → PQ OETF.
    SrgbToBt2020Pq,
    /// scRGB input (linear BT.709, FP16) converted to BT.2020 + PQ output.
    /// Used for HDR games submitting a `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT` swapchain.
    /// Applies: BT.709→BT.2020 gamut mapping → PQ OETF. No EOTF decode — input is already linear.
    Bt709LinearToBt2020Pq,
}

/// Supported input pixel formats for color conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum InputFormat {
    /// BGRx (32-bit, blue first, x = unused).
    BGRx,
    /// RGBx (32-bit, red first, x = unused).
    RGBx,
    /// BGRA (32-bit, blue first, alpha last).
    BGRA,
    /// RGBA (32-bit, red first, alpha last).
    RGBA,
    /// ABGR2101010 (packed 10-bit per channel, 2-bit alpha).
    /// Maps to DRM_FORMAT_ABGR2101010 / VK_FORMAT_A2B10G10R10_UNORM_PACK32.
    ABGR2101010,
    /// RGBA16F (64-bit, 16-bit float per channel).
    /// Maps to DRM_FORMAT_ABGR16161616F / VK_FORMAT_R16G16B16A16_SFLOAT.
    ///
    /// The converter treats FP16 data the same as other formats: passthrough
    /// for `ColorSpace::Bt709` / `Bt2020`, sRGB→BT.2020+PQ conversion for
    /// `ColorSpace::SrgbToBt2020Pq`, and scRGB-linear→BT.2020+PQ for
    /// `ColorSpace::Bt709LinearToBt2020Pq` (the natural pairing for FP16 input
    /// from an `EXTENDED_SRGB_LINEAR_EXT` swapchain). No linear→PQ transfer
    /// function is applied automatically; if the source is PQ-encoded
    /// (e.g. via the gamescope WSI layer), use `ColorSpace::Bt2020` for
    /// passthrough.
    RGBA16F,
}

impl InputFormat {
    /// Bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            InputFormat::BGRx
            | InputFormat::RGBx
            | InputFormat::BGRA
            | InputFormat::RGBA
            | InputFormat::ABGR2101010 => 4,
            InputFormat::RGBA16F => 8,
        }
    }

    /// Vulkan format for creating image views of this input format.
    pub fn vk_format(&self) -> vk::Format {
        match self {
            InputFormat::BGRx | InputFormat::BGRA => vk::Format::B8G8R8A8_UNORM,
            InputFormat::RGBx | InputFormat::RGBA => vk::Format::R8G8B8A8_UNORM,
            InputFormat::ABGR2101010 => vk::Format::A2B10G10R10_UNORM_PACK32,
            InputFormat::RGBA16F => vk::Format::R16G16B16A16_SFLOAT,
        }
    }
}

/// Supported output YUV formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// NV12: Y plane followed by interleaved UV (4:2:0), 8-bit.
    NV12,
    /// I420: Y plane, U plane, V plane (4:2:0), 8-bit.
    I420,
    /// YUV444 8-bit: 2-plane semi-planar (Y plane + interleaved UV) at full resolution.
    YUV444,
    /// P010: Y plane followed by interleaved UV (4:2:0), 10-bit in 16-bit words.
    P010,
    /// YUV444 10-bit: 2-plane semi-planar (Y plane + interleaved UV) in 16-bit words.
    YUV444P10,
}

impl OutputFormat {
    /// Calculate output size in bytes for given dimensions.
    ///
    /// The returned size is always a multiple of 4, since the compute shader writes
    /// to a `uint[]` buffer and `vkCmdFillBuffer` requires 4-byte aligned sizes.
    pub fn output_size(&self, width: u32, height: u32) -> usize {
        let pixel_count = (width * height) as usize;
        let raw = match self {
            OutputFormat::NV12 | OutputFormat::I420 => pixel_count * 3 / 2,
            OutputFormat::YUV444 => {
                // Y plane (aligned to 4 bytes) + UV interleaved plane.
                crate::align4(pixel_count) + pixel_count * 2
            }
            // 10-bit formats use 2 bytes per sample.
            OutputFormat::P010 => pixel_count * 3, // Y (2 bytes) + UV (1 byte each, half res)
            OutputFormat::YUV444P10 => {
                // Y plane (2 bytes/sample, aligned to 4 bytes) + UV interleaved (4 bytes/pixel).
                crate::align4(pixel_count * 2) + pixel_count * 4
            }
        };
        crate::align4(raw)
    }

    /// Get the Vulkan format for this output format.
    pub fn vulkan_format(&self) -> vk::Format {
        match self {
            OutputFormat::NV12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
            OutputFormat::I420 => vk::Format::G8_B8_R8_3PLANE_420_UNORM,
            OutputFormat::YUV444 => vk::Format::G8_B8R8_2PLANE_444_UNORM,
            OutputFormat::P010 => vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
            OutputFormat::YUV444P10 => vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16,
        }
    }

    /// Returns true if this is a 10-bit format.
    pub fn is_10bit(&self) -> bool {
        matches!(self, OutputFormat::P010 | OutputFormat::YUV444P10)
    }

    /// Bytes per sample for this format.
    pub fn bytes_per_sample(&self) -> usize {
        if self.is_10bit() { 2 } else { 1 }
    }
}

/// Configuration for the color converter.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ColorConverterConfig {
    /// Input frame width.
    pub width: u32,
    /// Input frame height.
    pub height: u32,
    /// Input pixel format.
    pub input_format: InputFormat,
    /// Output YUV format.
    pub output_format: OutputFormat,
    /// Color space for the RGB→YUV matrix.
    /// Use Bt709 for SDR, Bt2020 for HDR.
    pub color_space: ColorSpace,
    /// Full range (0-255 luma) or limited/studio range (16-235 luma).
    /// Must match the `full_range` flag in `ColorDescription` for correct playback.
    pub full_range: bool,
    /// SDR reference white level in nits for the sRGB→BT.2020+PQ conversion.
    ///
    /// Per ITU-R BT.2408 the standard value is 203 nits.
    /// Only used when `color_space` is `SrgbToBt2020Pq` or `Bt709LinearToBt2020Pq`
    /// (for the latter, the scRGB spec defines 1.0 == 80 cd/m²).
    pub sdr_reference_white_nits: f32,
}

impl ColorConverterConfig {
    /// Create a new configuration with BT.709 color space and limited range.
    pub fn new(
        width: u32,
        height: u32,
        input_format: InputFormat,
        output_format: OutputFormat,
    ) -> Self {
        Self {
            width,
            height,
            input_format,
            output_format,
            color_space: ColorSpace::Bt709,
            full_range: false,
            sdr_reference_white_nits: 203.0,
        }
    }
}

/// GPU-based color format converter.
///
/// Uses Vulkan compute shaders to convert RGB/BGR formats to YUV.
/// with minimal latency and high throughput.
pub struct ColorConverter {
    context: VideoContext,
    config: ColorConverterConfig,

    // Compute pipeline resources.
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    // Sampler for texelFetch on the source image.
    sampler: vk::Sampler,

    // Cached ImageView for the source image (avoids per-frame recreation).
    cached_src_view: Option<(vk::Image, vk::ImageView)>,

    // Output buffer (compute shader writes here).
    output_buffer: vk::Buffer,
    output_memory: vk::DeviceMemory,
    output_buffer_size: usize,

    // Plain descriptor pool + set (one per-frame update). We deliberately avoid
    // the VK_EXT_descriptor_buffer path: on this driver (RADV/AMD) populating a
    // COMBINED_IMAGE_SAMPLER into a descriptor buffer via vkGetDescriptorEXT made
    // texelFetch return only the R channel (G/B/A read as zero), producing a
    // green-screen stream. A plain descriptor set works correctly (the GPU scaler
    // uses one and produces valid full-color output).
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,

    // Command resources.
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl ColorConverter {
    /// Create a new color converter.
    pub fn new(context: VideoContext, config: ColorConverterConfig) -> Result<Self> {
        pipeline::create_converter(context, config)
    }

    /// Get the video context.
    ///
    /// Returns a reference to the VideoContext used by this converter.
    pub fn context(&self) -> &VideoContext {
        &self.context
    }

    /// Get the converter configuration.
    pub fn config(&self) -> &ColorConverterConfig {
        &self.config
    }

    /// Get the output buffer.
    ///
    /// Returns the Vulkan buffer containing raw YUV data.
    /// This is useful for direct GPU-to-GPU transfers.
    pub fn output_buffer(&self) -> vk::Buffer {
        self.output_buffer
    }

    /// Set the color space for subsequent conversions.
    ///
    /// This takes effect on the next `convert()` call without recreating the pipeline,
    /// since the color space is passed via push constants.
    pub fn set_color_space(&mut self, color_space: ColorSpace) {
        self.config.color_space = color_space;
    }

    /// Set the full-range flag for subsequent conversions.
    ///
    /// This takes effect on the next `convert()` call without recreating the pipeline,
    /// since the full-range flag is passed via push constants.
    pub fn set_full_range(&mut self, full_range: bool) {
        self.config.full_range = full_range;
    }

    /// Set the SDR reference white level for subsequent conversions.
    ///
    /// This takes effect on the next `convert()` call without recreating the pipeline,
    /// since the value is passed via push constants.  Only consumed by the
    /// `SrgbToBt2020Pq` and `Bt709LinearToBt2020Pq` color spaces.
    pub fn set_sdr_reference_white_nits(&mut self, nits: f32) {
        self.config.sdr_reference_white_nits = nits;
    }

    /// Build buffer-to-image copy regions for multi-planar YUV formats.
    ///
    /// For multi-planar formats like NV12, I420, and YUV444, we need separate.
    /// copy regions for each plane with the appropriate aspect mask.
    fn build_buffer_to_image_copy_regions(&self) -> Vec<vk::BufferImageCopy> {
        match self.config.output_format {
            OutputFormat::NV12 => {
                // NV12: Y plane (PLANE_0) followed by interleaved UV plane (PLANE_1)
                let y_size = (self.config.width * self.config.height) as u64;
                vec![
                    // Y plane.
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // UV plane (interleaved, half resolution)
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                ]
            }
            OutputFormat::I420 => {
                // I420: Y plane, U plane, V plane (all separate)
                let y_size = (self.config.width * self.config.height) as u64;
                let uv_size = y_size / 4;
                vec![
                    // Y plane.
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // U plane.
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                    // V plane.
                    vk::BufferImageCopy {
                        buffer_offset: y_size + uv_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_2,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                ]
            }
            OutputFormat::YUV444 => {
                // YUV444 8-bit 2-plane: Y plane at full resolution, UV interleaved at full resolution.
                // Align Y plane size to 4 bytes for VkBufferImageCopy::bufferOffset compliance.
                let y_size =
                    crate::align4((self.config.width * self.config.height) as usize) as u64;
                vec![
                    // Y plane.
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // UV plane (interleaved, full resolution).
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                ]
            }
            OutputFormat::P010 => {
                // P010: 10-bit NV12 with 16-bit samples.
                // Y plane: full resolution, 2 bytes per sample.
                // UV plane: half resolution, interleaved, 2 bytes per component.
                let y_size = (self.config.width * self.config.height * 2) as u64;
                vec![
                    // Y plane (16-bit samples).
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // UV plane (interleaved, half resolution, 16-bit per component).
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                ]
            }
            OutputFormat::YUV444P10 => {
                // YUV444 10-bit: 2-plane format (Y plane, UV interleaved).
                // Align Y plane size to 4 bytes for VkBufferImageCopy::bufferOffset compliance.
                let y_size =
                    crate::align4((self.config.width * self.config.height * 2) as usize) as u64;
                vec![
                    // Y plane (16-bit samples).
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // UV plane (interleaved, full resolution, 16-bit per component).
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                ]
            }
        }
    }

    /// Convert an input image directly to a target image (zero-copy encoder path).
    ///
    /// This is the most efficient path for encoding: it converts the source image.
    /// directly into the encoder's input image, eliminating an intermediate copy.
    ///
    /// The target image must:
    /// - Have the same dimensions as the converter's configuration
    /// - Be in a format compatible with NV12/YUV (G8_B8R8_2PLANE_420_UNORM)
    /// - Have TRANSFER_DST usage flag
    ///
    /// After this call, the target image will be in VIDEO_ENCODE_SRC_KHR layout,
    /// ready for encoding.
    ///
    /// # Arguments
    /// * `src_image` - Source RGB/BGR image (e.g., from DMA-BUF import)
    /// * `src_layout` - Current layout of the source image (e.g., `GENERAL` for cached
    ///   imports, `UNDEFINED` for first-time imports that haven't been transitioned yet)
    /// * `target_image` - Target image to write YUV data to (e.g., encoder's input_image)
    ///
    /// # Returns
    /// Returns `Ok(())` on success. The target_image is transitioned to VIDEO_ENCODE_SRC_KHR.
    pub fn convert(
        &mut self,
        src_image: vk::Image,
        src_layout: vk::ImageLayout,
        target_image: vk::Image,
    ) -> Result<()> {
        let start = std::time::Instant::now();

        // One-shot diagnostic: read the source image back via a plain Vulkan
        // buffer copy (no sampler/descriptor involved) to settle whether the
        // image *actually* holds R-only data or whether the converter's
        // descriptor-buffer sampling is what drops G/B/A. Gated on env var.
        if std::env::var("MOONSHINE_SCALE_DUMP").is_ok() {
            static SRC_READ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SRC_READ.swap(true, std::sync::atomic::Ordering::SeqCst) {
                dump_src_image_rgba(
                    self.context.device(),
                    self.context.compute_queue(),
                    self.context.memory_properties(),
                    src_image,
                    self.config.width,
                    self.config.height,
                );
            }
        }

        // Get or create ImageView for the source image (must happen before
        // borrowing device immutably, since this takes &mut self).
        let src_view = self.get_or_create_src_view(src_image)?;

        let device = self.context.device();

        // Reset and record command buffer.
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

            // --- Phase 1: Transition source image for shader read ---

            // For external memory (DMA-BUF) imports with EXCLUSIVE sharing,
            // the first use must include a queue family acquire operation.
            // Set srcQueueFamilyIndex to EXTERNAL (ownership is foreign)
            // and dstQueueFamilyIndex to the encoder's compute queue family.
            let needs_acquire = src_layout == vk::ImageLayout::UNDEFINED;
            let src_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(src_layout)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(if needs_acquire {
                    vk::QUEUE_FAMILY_EXTERNAL
                } else {
                    vk::QUEUE_FAMILY_IGNORED
                })
                .dst_queue_family_index(if needs_acquire {
                    self.context.compute_queue_family()
                } else {
                    vk::QUEUE_FAMILY_IGNORED
                })
                .image(src_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(if needs_acquire {
                    vk::AccessFlags::empty()
                } else {
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
                })
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[src_barrier],
            );

            // --- Phase 2: Run compute shader (reads source image directly) ---
            let effective_output_format = self.config.output_format as u32;
            let output_size: usize = self
                .config
                .output_format
                .output_size(self.config.width, self.config.height);
            device.cmd_fill_buffer(
                self.command_buffer,
                self.output_buffer,
                0,
                output_size as vk::DeviceSize,
                0,
            );

            // Barrier: fill buffer write -> shader read/write
            let fill_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .buffer(self.output_buffer)
                .size(vk::WHOLE_SIZE);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[fill_barrier],
                &[],
            );

            // Bind pipeline.
            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            // --- Populate the plain descriptor set (recreated each frame) ---
            //
            // We use a regular descriptor set rather than VK_EXT_descriptor_buffer:
            // on this driver populating a COMBINED_IMAGE_SAMPLER into a descriptor
            // buffer via vkGetDescriptorEXT made texelFetch return only the R
            // channel (G/B/A read as zero), producing a green-screen stream. A
            // plain descriptor set samples correctly (the GPU scaler relies on one).

            let image_info = [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(src_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(self.output_buffer)
                .offset(0)
                .range(self.output_buffer_size as u64)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&buffer_info),
            ];
            device.update_descriptor_sets(&writes, &[]);

            // Bind the descriptor set.
            let sets = [self.descriptor_set];
            device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &sets,
                &[],
            );

            // Push constants: width, height, input_format, output_format, color_space, full_range, sdr_white_nits.
            let push_constants: [u32; 7] = [
                self.config.width,
                self.config.height,
                self.config.input_format as u32,
                effective_output_format,
                self.config.color_space as u32,
                self.config.full_range as u32,
                self.config.sdr_reference_white_nits.to_bits(),
            ];
            let push_constants_bytes: &[u8] = std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                std::mem::size_of_val(&push_constants),
            );
            device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants_bytes,
            );

            // Dispatch workgroups (8x8 workgroup size)
            let workgroup_x = self.config.width.div_ceil(8);
            let workgroup_y = self.config.height.div_ceil(8);
            device.cmd_dispatch(self.command_buffer, workgroup_x, workgroup_y, 1);

            // --- Phase 3: Copy output buffer to target image (encoder's input) ---

            // Memory barrier: buffer write -> buffer read
            let output_buffer_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(self.output_buffer)
                .size(vk::WHOLE_SIZE);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[output_buffer_barrier],
                &[],
            );

            // Transition target image (encoder's input) to TRANSFER_DST layout.
            // The encoder's init clears the input image and leaves it in
            // VIDEO_ENCODE_SRC_KHR, so we must use that (not UNDEFINED) as the
            // old layout on every frame, including the first.
            let target_barrier_to_transfer = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(target_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[target_barrier_to_transfer],
            );

            // Copy buffer to target image - use per-plane copies for multi-planar formats.
            let copy_regions = self.build_buffer_to_image_copy_regions();

            device.cmd_copy_buffer_to_image(
                self.command_buffer,
                self.output_buffer,
                target_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_regions,
            );

            // Transition target image to VIDEO_ENCODE_SRC_KHR layout for encoding.
            let target_barrier_to_encode = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
                .image(target_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            // Transition source image back to GENERAL for reuse.
            let src_barrier_back = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(src_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[target_barrier_to_encode, src_barrier_back],
            );

            device
                .end_command_buffer(self.command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        // Submit and wait.
        unsafe {
            device
                .reset_fences(&[self.fence])
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

            let command_buffers = [self.command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            device
                .queue_submit(self.context.compute_queue(), &[submit_info], self.fence)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        // One-shot diagnostic: read back the converter's YUV output buffer and dump
        // it so we can tell whether the *conversion* produced black or real content.
        // Gated on MOONSHINE_SCALE_DUMP. The scaler dst was already proven valid, so
        // this isolates the RGBA->YUV compute step (descriptor-buffer path).
        if std::env::var("MOONSHINE_SCALE_DUMP").is_ok() {
            static DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !DUMPED.swap(true, std::sync::atomic::Ordering::SeqCst) {
                dump_nv12_output(
                    self.context.device(),
                    self.context.compute_queue(),
                    self.context.memory_properties(),
                    self.output_buffer,
                    self.config.width,
                    self.config.height,
                );
            }
        }

        let elapsed = start.elapsed();
        debug!("ColorConverter::convert() took {:?}", elapsed);

        Ok(())
    }

    /// Get or create an ImageView for the source image.
    fn get_or_create_src_view(&mut self, src_image: vk::Image) -> Result<vk::ImageView> {
        // Return cached view if it matches the current source image.
        if let Some((cached_image, cached_view)) = self.cached_src_view {
            if cached_image == src_image {
                return Ok(cached_view);
            }
            // Different image — destroy the old view.
            unsafe {
                self.context.device().destroy_image_view(cached_view, None);
            }
        }

        // Create a new ImageView for the source image.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(src_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(self.config.input_format.vk_format())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { self.context.device().create_image_view(&view_info, None) }
            .map_err(|e| PixelForgeError::ResourceCreation(format!("source image view: {}", e)))?;

        self.cached_src_view = Some((src_image, view));
        Ok(view)
    }
}

impl Drop for ColorConverter {
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();

            // Destroy cached source image view.
            if let Some((_, view)) = self.cached_src_view.take() {
                device.destroy_image_view(view, None);
            }

            // Destroy sampler.
            device.destroy_sampler(self.sampler, None);

            // Destroy output buffer and its memory.
            device.destroy_buffer(self.output_buffer, None);
            device.free_memory(self.output_memory, None);

            // Free the descriptor set, then destroy the pool.
            let _ = device.free_descriptor_sets(self.descriptor_pool, std::slice::from_ref(&self.descriptor_set));
            device.destroy_descriptor_pool(self.descriptor_pool, None);

            // Destroy pipeline resources.
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            // Destroy command resources.
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

/// One-shot diagnostic: copy a DEVICE_LOCAL NV12 buffer back to CPU and write a PNG.
/// Used to tell whether the *converter output* (RGBA->YUV) is black or correct.
fn dump_nv12_output(
	device: &ash::Device,
	queue: vk::Queue,
	mem_props: &vk::PhysicalDeviceMemoryProperties,
	src_buffer: vk::Buffer,
	width: u32,
	height: u32,
) {
	let path = "/tmp/moonshine_yuv_debug.nv12";
	let y_size = (width as u64) * (height as u64);
	let total = y_size * 3 / 2; // NV12: Y plane + interleaved UV at half res

	// Staging buffer in host-visible memory.
	let staging = match unsafe {
		device.create_buffer(
			&vk::BufferCreateInfo::default()
				.size(total)
				.usage(vk::BufferUsageFlags::TRANSFER_DST)
				.sharing_mode(vk::SharingMode::EXCLUSIVE),
			None,
		)
	} {
		Ok(b) => b,
		Err(e) => {
			tracing::warn!("yuv dump: create staging failed: {e}");
			return;
		},
	};
	let reqs = unsafe { device.get_buffer_memory_requirements(staging) };
	// Pick a HOST_VISIBLE + HOST_COHERENT type so we can map it back.
	let mtype = crate::encoder::resources::find_memory_type(
		mem_props,
		reqs.memory_type_bits,
		vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
	);
	let Some(mtype) = mtype else {
		unsafe { device.destroy_buffer(staging, None) };
		tracing::warn!("yuv dump: no host-visible memory type");
		return;
	};
	let mem = match unsafe {
		device.allocate_memory(
			&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(mtype),
			None,
		)
	} {
		Ok(m) => m,
		Err(e) => {
			unsafe { device.destroy_buffer(staging, None) };
			tracing::warn!("yuv dump: alloc failed: {e}");
			return;
		},
	};
	if let Err(e) = unsafe { device.bind_buffer_memory(staging, mem, 0) } {
		unsafe {
			device.free_memory(mem, None);
			device.destroy_buffer(staging, None);
		};
		tracing::warn!("yuv dump: bind failed: {e}");
		return;
	}

	let copy = vk::BufferCopy::default().size(total);
	let pool = match unsafe {
		device.create_command_pool(
			&vk::CommandPoolCreateInfo::default()
				.queue_family_index(0)
				.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
			None,
		)
	} {
		Ok(p) => p,
		Err(e) => {
			unsafe {
				device.free_memory(mem, None);
				device.destroy_buffer(staging, None);
			};
			tracing::warn!("yuv dump: pool failed: {e}");
			return;
		},
	};
	let cb = match unsafe {
		device.allocate_command_buffers(
			&vk::CommandBufferAllocateInfo::default()
				.command_pool(pool)
				.level(vk::CommandBufferLevel::PRIMARY)
				.command_buffer_count(1),
		)
	} {
		Ok(c) => c[0],
		Err(e) => {
			unsafe {
				device.destroy_command_pool(pool, None);
				device.free_memory(mem, None);
				device.destroy_buffer(staging, None);
			};
			tracing::warn!("yuv dump: cb failed: {e}");
			return;
		},
	};
	let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
		Ok(f) => f,
		Err(e) => {
			unsafe {
				device.free_command_buffers(pool, &[cb]);
				device.destroy_command_pool(pool, None);
				device.free_memory(mem, None);
				device.destroy_buffer(staging, None);
			};
			tracing::warn!("yuv dump: fence failed: {e}");
			return;
		},
	};

	let ok = (|| -> bool {
		if let Err(_) = unsafe { device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) } {
			return false;
		}
		if let Err(_) = unsafe {
			device.begin_command_buffer(
				cb,
				&vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
			)
		} {
			return false;
		}
		unsafe { device.cmd_copy_buffer(cb, src_buffer, staging, &[copy]) };
		if let Err(_) = unsafe { device.end_command_buffer(cb) } {
			return false;
		}
		if let Err(_) = unsafe {
			device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cb])], fence)
		} {
			return false;
		}
		if let Err(_) = unsafe { device.wait_for_fences(&[fence], true, 2_000_000_000) } {
			return false;
		}
		true
	})();

	if !ok {
		unsafe {
			device.destroy_fence(fence, None);
			device.free_command_buffers(pool, &[cb]);
			device.destroy_command_pool(pool, None);
			device.free_memory(mem, None);
			device.destroy_buffer(staging, None);
		};
		tracing::warn!("yuv dump: submit failed");
		return;
	}

	let mapped = match unsafe { device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
		Ok(p) => p as *const u8,
		Err(e) => {
			unsafe {
				device.destroy_fence(fence, None);
				device.free_command_buffers(pool, &[cb]);
				device.destroy_command_pool(pool, None);
				device.free_memory(mem, None);
				device.destroy_buffer(staging, None);
			};
			tracing::warn!("yuv dump: map failed: {e:?}");
			return;
		},
	};

	let mut bytes = vec![0u8; total as usize];
	unsafe { std::ptr::copy_nonoverlapping(mapped, bytes.as_mut_ptr(), total as usize) };
	unsafe { device.unmap_memory(mem) };

	unsafe {
		device.destroy_fence(fence, None);
		device.free_command_buffers(pool, &[cb]);
		device.destroy_command_pool(pool, None);
		device.free_memory(mem, None);
		device.destroy_buffer(staging, None);
	};

	match std::fs::write(path, &bytes) {
		Ok(_) => tracing::info!("yuv dump: wrote {path} ({}x{} NV12, {} bytes)", width, height, bytes.len()),
		Err(e) => tracing::warn!("yuv dump: write failed: {e}"),
	}
}

/// One-shot diagnostic: copy a device-local R8G8B8A8_UNORM image straight to a
/// host-visible buffer via vkCmdCopyImageToBuffer (no sampler, no descriptor) and
/// write the raw RGBA. This tells us whether the source image itself holds only
/// the R channel (an upstream upload/scale bug) or whether the converter's
/// descriptor-buffer sampling is what drops G/B/A.
fn dump_src_image_rgba(
	device: &ash::Device,
	queue: vk::Queue,
	mem_props: &vk::PhysicalDeviceMemoryProperties,
	image: vk::Image,
	width: u32,
	height: u32,
) {
	let path = "/tmp/moonshine_srcimg_debug.rgba";
	let total = (width as u64) * (height as u64) * 4;

	let buf = match unsafe {
		device.create_buffer(
			&vk::BufferCreateInfo::default()
				.size(total)
				.usage(vk::BufferUsageFlags::TRANSFER_DST)
				.sharing_mode(vk::SharingMode::EXCLUSIVE),
			None,
		)
	} {
		Ok(b) => b,
		Err(e) => {
			tracing::warn!("srcimg dump: create buffer failed: {e}");
			return;
		},
	};
	let reqs = unsafe { device.get_buffer_memory_requirements(buf) };
	let Some(mtype) = crate::encoder::resources::find_memory_type(
		mem_props,
		reqs.memory_type_bits,
		vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
	) else {
		unsafe { device.destroy_buffer(buf, None) };
		tracing::warn!("srcimg dump: no host-visible memory type");
		return;
	};
	let mem = match unsafe {
		device.allocate_memory(
			&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(mtype),
			None,
		)
	} {
		Ok(m) => m,
		Err(e) => {
			unsafe { device.destroy_buffer(buf, None) };
			tracing::warn!("srcimg dump: alloc failed: {e}");
			return;
		},
	};
	if let Err(e) = unsafe { device.bind_buffer_memory(buf, mem, 0) } {
		unsafe {
			device.free_memory(mem, None);
			device.destroy_buffer(buf, None);
		};
		tracing::warn!("srcimg dump: bind failed: {e}");
		return;
	}

	let pool = match unsafe {
		device.create_command_pool(
			&vk::CommandPoolCreateInfo::default()
				.queue_family_index(0)
				.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
			None,
		)
	} {
		Ok(p) => p,
		Err(e) => {
			unsafe {
				device.free_memory(mem, None);
				device.destroy_buffer(buf, None);
			};
			tracing::warn!("srcimg dump: pool failed: {e}");
			return;
		},
	};
	let cb = match unsafe {
		device.allocate_command_buffers(
			&vk::CommandBufferAllocateInfo::default()
				.command_pool(pool)
				.level(vk::CommandBufferLevel::PRIMARY)
				.command_buffer_count(1),
		)
	} {
		Ok(c) => c[0],
		Err(e) => {
			unsafe {
				device.destroy_command_pool(pool, None);
				device.free_memory(mem, None);
				device.destroy_buffer(buf, None);
			};
			tracing::warn!("srcimg dump: cb failed: {e}");
			return;
		},
	};
	let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
		Ok(f) => f,
		Err(e) => {
			unsafe {
				device.free_command_buffers(pool, &[cb]);
				device.destroy_command_pool(pool, None);
				device.free_memory(mem, None);
				device.destroy_buffer(buf, None);
			};
			tracing::warn!("srcimg dump: fence failed: {e}");
			return;
		},
	};

	let copy = vk::BufferImageCopy::default()
		.buffer_offset(0)
		.buffer_image_height(height)
		.image_subresource(vk::ImageSubresourceLayers {
			aspect_mask: vk::ImageAspectFlags::COLOR,
			mip_level: 0,
			base_array_layer: 0,
			layer_count: 1,
		})
		.image_offset(vk::Offset3D::default())
		.image_extent(vk::Extent3D {
			width,
			height,
			depth: 1,
		});

	let ok = (|| -> bool {
		if let Err(_) = unsafe { device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) } {
			return false;
		}
		if let Err(_) = unsafe {
			device.begin_command_buffer(
				cb,
				&vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
			)
		} {
			return false;
		}
		unsafe { device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::GENERAL, buf, &[copy]) };
		if let Err(_) = unsafe { device.end_command_buffer(cb) } {
			return false;
		}
		if let Err(_) = unsafe {
			device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cb])], fence)
		} {
			return false;
		}
		if let Err(_) = unsafe { device.wait_for_fences(&[fence], true, 2_000_000_000) } {
			return false;
		}
		true
	})();

	if !ok {
		unsafe {
			device.destroy_fence(fence, None);
			device.free_command_buffers(pool, &[cb]);
			device.destroy_command_pool(pool, None);
			device.free_memory(mem, None);
			device.destroy_buffer(buf, None);
		};
		tracing::warn!("srcimg dump: submit failed");
		return;
	}

	let mapped = match unsafe { device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
		Ok(p) => p as *const u8,
		Err(e) => {
			unsafe {
				device.destroy_fence(fence, None);
				device.free_command_buffers(pool, &[cb]);
				device.destroy_command_pool(pool, None);
				device.free_memory(mem, None);
				device.destroy_buffer(buf, None);
			};
			tracing::warn!("srcimg dump: map failed: {e:?}");
			return;
		},
	};

	let mut bytes = vec![0u8; total as usize];
	unsafe { std::ptr::copy_nonoverlapping(mapped, bytes.as_mut_ptr(), total as usize) };
	unsafe { device.unmap_memory(mem) };

	unsafe {
		device.destroy_fence(fence, None);
		device.free_command_buffers(pool, &[cb]);
		device.destroy_command_pool(pool, None);
		device.free_memory(mem, None);
		device.destroy_buffer(buf, None);
	};

	match std::fs::write(path, &bytes) {
		Ok(_) => tracing::info!("srcimg dump: wrote {path} ({}x{} RGBA, {} bytes)", width, height, bytes.len()),
		Err(e) => tracing::warn!("srcimg dump: write failed: {e}"),
	}
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================
    // InputFormat tests.
    // ========================

    #[test]
    fn test_input_format_bytes_per_pixel() {
        assert_eq!(InputFormat::BGRx.bytes_per_pixel(), 4);
        assert_eq!(InputFormat::RGBx.bytes_per_pixel(), 4);
        assert_eq!(InputFormat::BGRA.bytes_per_pixel(), 4);
        assert_eq!(InputFormat::RGBA.bytes_per_pixel(), 4);
        assert_eq!(InputFormat::ABGR2101010.bytes_per_pixel(), 4);
        assert_eq!(InputFormat::RGBA16F.bytes_per_pixel(), 8);
    }

    #[test]
    fn test_input_format_enum_values() {
        // Verify enum values match shader expectations.
        assert_eq!(InputFormat::BGRx as u32, 0);
        assert_eq!(InputFormat::RGBx as u32, 1);
        assert_eq!(InputFormat::BGRA as u32, 2);
        assert_eq!(InputFormat::RGBA as u32, 3);
        assert_eq!(InputFormat::ABGR2101010 as u32, 4);
        assert_eq!(InputFormat::RGBA16F as u32, 5);
    }

    #[test]
    fn test_input_format_vk_format() {
        assert_eq!(InputFormat::BGRx.vk_format(), vk::Format::B8G8R8A8_UNORM);
        assert_eq!(InputFormat::BGRA.vk_format(), vk::Format::B8G8R8A8_UNORM);
        assert_eq!(InputFormat::RGBx.vk_format(), vk::Format::R8G8B8A8_UNORM);
        assert_eq!(InputFormat::RGBA.vk_format(), vk::Format::R8G8B8A8_UNORM);
        assert_eq!(
            InputFormat::ABGR2101010.vk_format(),
            vk::Format::A2B10G10R10_UNORM_PACK32
        );
        assert_eq!(
            InputFormat::RGBA16F.vk_format(),
            vk::Format::R16G16B16A16_SFLOAT
        );
    }

    // ========================
    // ColorSpace tests.
    // ========================

    #[test]
    fn test_color_space_enum_values() {
        // Verify enum values match shader push constant expectations:
        // 0=BT.709, 1=BT.2020, 2=sRGB→BT.2020+PQ, 3=scRGB (BT.709 linear)→BT.2020+PQ.
        assert_eq!(ColorSpace::Bt709 as u32, 0);
        assert_eq!(ColorSpace::Bt2020 as u32, 1);
        assert_eq!(ColorSpace::SrgbToBt2020Pq as u32, 2);
        assert_eq!(ColorSpace::Bt709LinearToBt2020Pq as u32, 3);
    }

    // ========================
    // OutputFormat tests.
    // ========================

    #[test]
    fn test_output_format_size_nv12() {
        // NV12: Y plane + interleaved UV at half resolution = 1.5 * pixel_count.
        assert_eq!(OutputFormat::NV12.output_size(8, 8), 96); // 64 * 1.5 = 96
        assert_eq!(OutputFormat::NV12.output_size(16, 16), 384); // 256 * 1.5 = 384
        assert_eq!(
            OutputFormat::NV12.output_size(1920, 1080),
            1920 * 1080 * 3 / 2
        );
    }

    #[test]
    fn test_output_format_size_i420() {
        // I420: Y plane + U plane (quarter) + V plane (quarter) = 1.5 * pixel_count.
        assert_eq!(OutputFormat::I420.output_size(8, 8), 96);
        assert_eq!(OutputFormat::I420.output_size(16, 16), 384);
        assert_eq!(
            OutputFormat::I420.output_size(1920, 1080),
            1920 * 1080 * 3 / 2
        );
    }

    #[test]
    fn test_output_format_size_yuv444() {
        // YUV444: Full resolution Y + U + V = 3 * pixel_count.
        assert_eq!(OutputFormat::YUV444.output_size(8, 8), 192); // 64 * 3 = 192
        assert_eq!(OutputFormat::YUV444.output_size(16, 16), 768); // 256 * 3 = 768
        assert_eq!(
            OutputFormat::YUV444.output_size(1920, 1080),
            1920 * 1080 * 3
        );
    }

    #[test]
    fn test_output_format_size_standard_resolutions() {
        // Common video resolutions.
        let resolutions = [
            (320, 240),   // QVGA
            (640, 480),   // VGA
            (1280, 720),  // HD
            (1920, 1080), // Full HD
            (3840, 2160), // 4K
        ];

        for (width, height) in resolutions {
            let pixels = (width * height) as usize;

            // NV12 and I420 should be 1.5x pixels.
            assert_eq!(
                OutputFormat::NV12.output_size(width, height),
                pixels * 3 / 2
            );
            assert_eq!(
                OutputFormat::I420.output_size(width, height),
                pixels * 3 / 2
            );

            // YUV444 should be 3x pixels.
            assert_eq!(OutputFormat::YUV444.output_size(width, height), pixels * 3);
        }
    }

    #[test]
    fn test_output_format_enum_values() {
        // Verify enum values match shader expectations.
        assert_eq!(OutputFormat::NV12 as u32, 0);
        assert_eq!(OutputFormat::I420 as u32, 1);
        assert_eq!(OutputFormat::YUV444 as u32, 2);
    }

    // ========================
    // ColorConverterConfig tests.
    // ========================

    #[test]
    fn test_config_clone() {
        let config = ColorConverterConfig {
            width: 1920,
            height: 1080,
            input_format: InputFormat::BGRx,
            output_format: OutputFormat::NV12,
            color_space: ColorSpace::Bt709,
            full_range: true,
            sdr_reference_white_nits: 203.0,
        };

        let cloned = config.clone();
        assert_eq!(cloned.width, 1920);
        assert_eq!(cloned.height, 1080);
        assert_eq!(cloned.input_format, InputFormat::BGRx);
        assert_eq!(cloned.output_format, OutputFormat::NV12);
        assert_eq!(cloned.color_space, ColorSpace::Bt709);
        assert!(cloned.full_range);
    }

    #[test]
    fn test_config_debug() {
        let config = ColorConverterConfig {
            width: 640,
            height: 480,
            input_format: InputFormat::RGBA,
            output_format: OutputFormat::I420,
            color_space: ColorSpace::Bt709,
            full_range: true,
            sdr_reference_white_nits: 203.0,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("640"));
        assert!(debug_str.contains("480"));
        assert!(debug_str.contains("RGBA"));
        assert!(debug_str.contains("I420"));
    }

    // ========================
    // ColorConverter Vulkan tests (require hardware)
    // ========================
    // These tests require Vulkan support and are gated behind a feature.
    // or will be skipped if Vulkan initialization fails.

    /// Helper to create a Vulkan context for testing.
    /// Returns None if Vulkan is not available.
    fn create_test_context() -> Option<VideoContext> {
        use crate::VideoContextBuilder;

        VideoContextBuilder::new()
            .app_name("ColorConverter Test")
            .enable_validation(false)
            .build()
            .ok()
    }

    #[test]
    fn test_converter_creation() {
        let Some(context) = create_test_context() else {
            eprintln!("Skipping test_converter_creation: Vulkan not available");
            return;
        };

        let config = ColorConverterConfig {
            width: 64,
            height: 64,
            input_format: InputFormat::BGRx,
            output_format: OutputFormat::NV12,
            color_space: ColorSpace::Bt709,
            full_range: true,
            sdr_reference_white_nits: 203.0,
        };

        let result = ColorConverter::new(context, config);
        assert!(
            result.is_ok(),
            "Failed to create ColorConverter: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_converter_creation_various_formats() {
        let Some(context) = create_test_context() else {
            eprintln!("Skipping test: Vulkan not available");
            return;
        };

        let input_formats = [
            InputFormat::BGRx,
            InputFormat::RGBx,
            InputFormat::BGRA,
            InputFormat::RGBA,
        ];
        let output_formats = [OutputFormat::NV12, OutputFormat::I420, OutputFormat::YUV444];

        for input_format in &input_formats {
            for output_format in &output_formats {
                let config = ColorConverterConfig {
                    width: 32,
                    height: 32,
                    input_format: *input_format,
                    output_format: *output_format,
                    color_space: ColorSpace::Bt709,
                    full_range: true,
                    sdr_reference_white_nits: 203.0,
                };

                let result = ColorConverter::new(context.clone(), config);
                assert!(
                    result.is_ok(),
                    "Failed to create ColorConverter with {:?} -> {:?}: {:?}",
                    input_format,
                    output_format,
                    result.err()
                );
            }
        }
    }
}
