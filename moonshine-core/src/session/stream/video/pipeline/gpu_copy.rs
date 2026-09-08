//! GPU-side import for DMA-BUF frames the driver cannot import as external memory.
//!
//! Some drivers (e.g. RADV at certain large resolutions) report no usable memory
//! type for a portal buffer's DMA-BUF fd, so zero-copy `vkImportMemoryFd` fails.
//! Rather than copying pixels on the CPU, this uploads the mapped plane into a
//! host-visible staging buffer and runs `vkCmdCopyBufferToImage` into a
//! device-local image — the pixel move happens on the GPU (DMA), keeping the
//! whole pipeline off the CPU.

use ash::vk;
use pixelforge::VideoContext;
use std::os::fd::RawFd;

struct CachedImage {
	image: vk::Image,
	memory: vk::DeviceMemory,
	width: u32,
	height: u32,
	format: vk::Format,
	row_pitch: u64,
	first_use: bool,
}

struct StagingBuffer {
	buffer: vk::Buffer,
	memory: vk::DeviceMemory,
	size: vk::DeviceSize,
}

pub(crate) struct GpuCopyImporter {
	context: VideoContext,
	image: Option<CachedImage>,
	staging: Option<StagingBuffer>,
	command_pool: vk::CommandPool,
	command_buffer: vk::CommandBuffer,
	fence: vk::Fence,
}

impl GpuCopyImporter {
	pub fn new(context: VideoContext) -> Result<Self, String> {
		let device = context.device();

		let command_pool = unsafe {
			device.create_command_pool(
				&vk::CommandPoolCreateInfo::default()
					.queue_family_index(context.transfer_queue_family())
					.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
				None,
			)
		}
		.map_err(|e| format!("gpu copy command pool: {e}"))?;

		let command_buffer = unsafe {
			device.allocate_command_buffers(
				&vk::CommandBufferAllocateInfo::default()
					.command_pool(command_pool)
					.level(vk::CommandBufferLevel::PRIMARY)
					.command_buffer_count(1),
			)
		}
		.map_err(|e| format!("gpu copy command buffer alloc: {e}"))?[0];

		let fence = unsafe {
			device.create_fence(
				&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
				None,
			)
		}
		.map_err(|e| format!("gpu copy fence: {e}"))?;

		Ok(Self {
			context,
			image: None,
			staging: None,
			command_pool,
			command_buffer,
			fence,
		})
	}

	fn ensure_image(&mut self, width: u32, height: u32, format: vk::Format) -> Result<(vk::Image, u64, bool), String> {
		let device = self.context.device();
		let needs_create = self
			.image
			.as_ref()
			.is_none_or(|c| c.width != width || c.height != height || c.format != format);
		if needs_create {
			if let Some(old) = self.image.take() {
				unsafe {
					device.destroy_image(old.image, None);
					device.free_memory(old.memory, None);
				}
			}

			let create_info = vk::ImageCreateInfo::default()
				.image_type(vk::ImageType::TYPE_2D)
				.format(format)
				.extent(vk::Extent3D {
					width,
					height,
					depth: 1,
				})
				.mip_levels(1)
				.array_layers(1)
				.samples(vk::SampleCountFlags::TYPE_1)
				.tiling(vk::ImageTiling::LINEAR)
				.usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
				.sharing_mode(vk::SharingMode::EXCLUSIVE)
				.initial_layout(vk::ImageLayout::UNDEFINED);
			let image =
				unsafe { device.create_image(&create_info, None) }.map_err(|e| format!("gpu copy image: {e}"))?;

			let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
			let mem_type = self
				.context
				.find_memory_type(mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
				.ok_or_else(|| "gpu copy: no DEVICE_LOCAL memory type".to_string())?;
			let memory = unsafe {
				let alloc_info = vk::MemoryAllocateInfo::default()
					.allocation_size(mem_reqs.size)
					.memory_type_index(mem_type);
				device.allocate_memory(&alloc_info, None)
			}
			.map_err(|e| {
				unsafe { device.destroy_image(image, None) };
				format!("gpu copy image memory: {e}")
			})?;
			if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
				unsafe {
					device.free_memory(memory, None);
					device.destroy_image(image, None);
				};
				return Err(format!("gpu copy image bind: {e}"));
			}

			let subresource = vk::ImageSubresource::default()
				.aspect_mask(vk::ImageAspectFlags::COLOR)
				.mip_level(0)
				.array_layer(0);
			let layout = unsafe { device.get_image_subresource_layout(image, subresource) };

			self.image = Some(CachedImage {
				image,
				memory,
				width,
				height,
				format,
				row_pitch: layout.row_pitch,
				first_use: true,
			});
		}
		let cached = self.image.as_ref().expect("gpu copy image just created");
		Ok((cached.image, cached.row_pitch, cached.first_use))
	}

	fn ensure_staging(&mut self, size: vk::DeviceSize) -> Result<vk::Buffer, String> {
		let device = self.context.device();
		let needs = self.staging.as_ref().is_none_or(|s| s.size < size);
		if needs {
			if let Some(old) = self.staging.take() {
				unsafe {
					device.free_memory(old.memory, None);
					device.destroy_buffer(old.buffer, None);
				}
			}
			let buffer = unsafe {
				device.create_buffer(
					&vk::BufferCreateInfo::default()
						.size(size)
						.usage(vk::BufferUsageFlags::TRANSFER_SRC)
						.sharing_mode(vk::SharingMode::EXCLUSIVE),
					None,
				)
			}
			.map_err(|e| format!("gpu copy staging buffer: {e}"))?;
			let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
			let mem_type = self
				.context
				.find_memory_type(
					reqs.memory_type_bits,
					vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
				)
				.ok_or_else(|| "gpu copy: no HOST_VISIBLE memory type".to_string())?;
			let memory = unsafe {
				device.allocate_memory(
					&vk::MemoryAllocateInfo::default()
						.allocation_size(reqs.size)
						.memory_type_index(mem_type),
					None,
				)
			}
			.map_err(|e| {
				unsafe { device.destroy_buffer(buffer, None) };
				format!("gpu copy staging memory: {e}")
			})?;
			if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
				unsafe {
					device.free_memory(memory, None);
					device.destroy_buffer(buffer, None);
				};
				return Err(format!("gpu copy staging bind: {e}"));
			}
			self.staging = Some(StagingBuffer {
				buffer,
				memory,
				size: reqs.size,
			});
		}
		let s = self.staging.as_ref().expect("gpu copy staging just created");
		Ok(s.buffer)
	}

	/// Copy a CPU-visible plane into a device-local image on the GPU.
	///
	/// Returns `(image, needs_transition)` — `needs_transition` is `true` for a
	/// freshly (re)created image still in `UNDEFINED` layout.
	pub fn upload_or_reuse(
		&mut self,
		src: *const u8,
		width: u32,
		height: u32,
		stride: u32,
		format: vk::Format,
	) -> Result<(vk::Image, bool), String> {
		// Wait for the previous submit before reusing the command buffer/staging;
		// resetting under in-flight work would be use-after-free.
		{
			let device = self.context.device();
			let status = unsafe { device.get_fence_status(self.fence) };
			match status {
				Ok(true) => {},
				Ok(false) => {
					if let Err(e) = unsafe { device.wait_for_fences(&[self.fence], true, 500_000_000) } {
						return Err(format!("gpu copy: timed out waiting for previous frame: {e:?}"));
					}
				},
				Err(e) => return Err(format!("gpu copy: get_fence_status failed: {e:?}")),
			}
			if let Err(e) = unsafe { device.reset_fences(&[self.fence]) } {
				tracing::warn!("gpu copy: failed to reset fence: {e:?}");
			}
		}

		let (image, row_pitch, first_use) = self.ensure_image(width, height, format)?;

		let staging_size = (stride as vk::DeviceSize).saturating_mul(height as vk::DeviceSize);
		let staging_buffer = self.ensure_staging(staging_size)?;
		let staging_mem = self.staging.as_ref().expect("staging just created").memory;

		let device = self.context.device();
		let mapped = unsafe { device.map_memory(staging_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
			.map_err(|e| format!("gpu copy map staging: {e}"))?;
		let dst = mapped as *mut u8;
		if stride as u64 == row_pitch {
			let total = (stride as u64)
				.saturating_mul(height as u64)
				.min(row_pitch.saturating_mul(height as u64)) as usize;
			unsafe { std::ptr::copy_nonoverlapping(src, dst, total) };
		} else {
			let row_len = (stride as u64).min(row_pitch) as usize;
			let mut src_off = 0usize;
			let mut dst_off = 0u64;
			for _ in 0..height {
				let len = row_len.min(stride as usize);
				unsafe { std::ptr::copy_nonoverlapping(src.add(src_off), dst.add(dst_off as usize), len) };
				src_off += stride as usize;
				dst_off += row_pitch;
			}
		}
		unsafe { device.unmap_memory(staging_mem) };

		let barrier_in = vk::ImageMemoryBarrier::default()
			.old_layout(if first_use {
				vk::ImageLayout::UNDEFINED
			} else {
				vk::ImageLayout::GENERAL
			})
			.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(image)
			.subresource_range(full_range())
			.src_access_mask(if first_use {
				vk::AccessFlags::empty()
			} else {
				vk::AccessFlags::SHADER_READ
			})
			.dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
		let barrier_out = vk::ImageMemoryBarrier::default()
			.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
			.new_layout(vk::ImageLayout::GENERAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(image)
			.subresource_range(full_range())
			.src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
			.dst_access_mask(vk::AccessFlags::SHADER_READ);

		let copy = vk::BufferImageCopy::default()
			.buffer_offset(0)
			.buffer_row_length(stride)
			.buffer_image_height(height)
			.image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR))
			.image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
			.image_extent(vk::Extent3D {
				width,
				height,
				depth: 1,
			});

		if let Err(e) =
			unsafe { device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES) }
		{
			return Err(format!("gpu copy: reset command buffer: {e:?}"));
		}
		let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
		if let Err(e) = unsafe { device.begin_command_buffer(self.command_buffer, &begin_info) } {
			return Err(format!("gpu copy: begin command buffer: {e:?}"));
		}
		unsafe {
			device.cmd_pipeline_barrier(
				self.command_buffer,
				vk::PipelineStageFlags::TOP_OF_PIPE,
				vk::PipelineStageFlags::TRANSFER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[barrier_in],
			);
			device.cmd_copy_buffer_to_image(
				self.command_buffer,
				staging_buffer,
				image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&[copy],
			);
			device.cmd_pipeline_barrier(
				self.command_buffer,
				vk::PipelineStageFlags::TRANSFER,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[barrier_out],
			);
		}
		if let Err(e) = unsafe { device.end_command_buffer(self.command_buffer) } {
			return Err(format!("gpu copy: end command buffer: {e}"));
		}

		let wait_stages = [vk::PipelineStageFlags::TRANSFER];
		let submit_info = vk::SubmitInfo::default()
			.command_buffers(std::slice::from_ref(&self.command_buffer))
			.wait_dst_stage_mask(&wait_stages);
		if let Err(e) = unsafe {
			device.queue_submit(
				self.context.transfer_queue(),
				std::slice::from_ref(&submit_info),
				self.fence,
			)
		} {
			return Err(format!("gpu copy: submit: {e:?}"));
		}
		if let Err(e) = unsafe { device.wait_for_fences(&[self.fence], true, 2_000_000_000) } {
			return Err(format!("gpu copy: fence wait failed: {e:?}"));
		}

		self.image.as_mut().expect("image just used").first_use = false;
		Ok((image, first_use))
	}
}

fn full_range() -> vk::ImageSubresourceRange {
	vk::ImageSubresourceRange {
		aspect_mask: vk::ImageAspectFlags::COLOR,
		base_mip_level: 0,
		level_count: 1,
		base_array_layer: 0,
		layer_count: 1,
	}
}

impl Drop for GpuCopyImporter {
	fn drop(&mut self) {
		let device = self.context.device();
		if let Err(e) = unsafe { device.wait_for_fences(&[self.fence], true, 2_000_000_000) } {
			tracing::warn!("gpu copy drop: timed out waiting for fence: {e:?}");
		}
		unsafe {
			if let Some(img) = self.image.take() {
				device.destroy_image(img.image, None);
				device.free_memory(img.memory, None);
			}
			if let Some(s) = self.staging.take() {
				device.free_memory(s.memory, None);
				device.destroy_buffer(s.buffer, None);
			}
			device.destroy_fence(self.fence, None);
			device.free_command_buffers(self.command_pool, std::slice::from_ref(&self.command_buffer));
			device.destroy_command_pool(self.command_pool, None);
		}
	}
}

/// Map a DMA-BUF plane fd read-only for the CPU fallback. Returns `(ptr, size)`
/// on success so the caller can feed it through the GPU-copy path.
pub(crate) fn map_dmabuf_plane(fd: RawFd, offset: u32, stride: u32, height: u32) -> Option<(*const u8, usize)> {
	if fd < 0 {
		return None;
	}
	let size = (stride as usize) * (height as usize);
	if size == 0 {
		return None;
	}
	let ptr = unsafe {
		libc::mmap(
			std::ptr::null_mut(),
			size,
			libc::PROT_READ,
			libc::MAP_PRIVATE,
			fd,
			offset as libc::off_t,
		)
	};
	if ptr == libc::MAP_FAILED {
		tracing::warn!("DMA-BUF GPU-copy fallback: mmap failed fd={fd} size={size}");
		return None;
	}
	Some((ptr as *const u8, size))
}
