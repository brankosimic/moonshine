use ash::vk;
use pixelforge::VideoContext;

fn spirv_words(bytes: &[u8]) -> Vec<u32> {
	bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

const SCALER_SPIRV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scaler.spv"));

/// A Vulkan image plus its bound memory.
struct GpuImage {
	image: vk::Image,
	view: vk::ImageView,
	memory: vk::DeviceMemory,
	width: u32,
	height: u32,
}

/// Reusable host-visible staging buffer, grown on demand and reused across
/// frames to avoid per-frame `vkCreateBuffer` +
/// `vkAllocateMemory` + `vkFreeMemory` churn in the encode hot path.
/// Grown (never shrunk) when a larger frame arrives.
struct StagingBuffer {
	buffer: vk::Buffer,
	memory: vk::DeviceMemory,
	size: vk::DeviceSize,
}

fn create_image(
	context: &VideoContext,
	width: u32,
	height: u32,
	usage: vk::ImageUsageFlags,
) -> Result<GpuImage, String> {
	let device = context.device();

	let image_info = vk::ImageCreateInfo::default()
		.image_type(vk::ImageType::TYPE_2D)
		.format(vk::Format::R8G8B8A8_UNORM)
		.extent(vk::Extent3D {
			width,
			height,
			depth: 1,
		})
		.mip_levels(1)
		.array_layers(1)
		.samples(vk::SampleCountFlags::TYPE_1)
		.tiling(vk::ImageTiling::OPTIMAL)
		.usage(usage)
		.sharing_mode(vk::SharingMode::EXCLUSIVE)
		.initial_layout(vk::ImageLayout::UNDEFINED);
	let image = unsafe { device.create_image(&image_info, None) }.map_err(|e| format!("scaler image: {e}"))?;

	let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
	let mem_type = context
		.find_memory_type(mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
		.or_else(|| context.find_memory_type(mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::empty()))
		.ok_or_else(|| "scaler image: no suitable memory type".to_string())?;
	let memory = unsafe {
		let alloc_info = vk::MemoryAllocateInfo::default()
			.allocation_size(mem_reqs.size)
			.memory_type_index(mem_type);
		device.allocate_memory(&alloc_info, None)
	}
	.map_err(|e| {
		unsafe { device.destroy_image(image, None) };
		format!("scaler image memory: {e}")
	})?;
	if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
		unsafe {
			device.free_memory(memory, None);
			device.destroy_image(image, None);
		};
		return Err(format!("scaler image bind: {e}"));
	}

	let view_info = vk::ImageViewCreateInfo::default()
		.image(image)
		.view_type(vk::ImageViewType::TYPE_2D)
		.format(vk::Format::R8G8B8A8_UNORM)
		.subresource_range(vk::ImageSubresourceRange {
			aspect_mask: vk::ImageAspectFlags::COLOR,
			base_mip_level: 0,
			level_count: 1,
			base_array_layer: 0,
			layer_count: 1,
		});
	let view = unsafe { device.create_image_view(&view_info, None) }.map_err(|e| {
		unsafe {
			device.free_memory(memory, None);
			device.destroy_image(image, None);
		}
		format!("scaler image view: {e}")
	})?;

	Ok(GpuImage {
		image,
		view,
		memory,
		width,
		height,
	})
}

fn destroy_image(context: &VideoContext, img: GpuImage) {
	let device = context.device();
	unsafe {
		device.destroy_image_view(img.view, None);
		device.destroy_image(img.image, None);
		device.free_memory(img.memory, None);
	}
}

/// GPU compute-shader scaler. Uploads a packed RGBA source to a sampled image and
/// runs a bilinear-filtered scale into a device-local target image in one submit.
pub(crate) struct GpuScaler {
	context: VideoContext,
	src_image: Option<GpuImage>,
	dst_image: Option<GpuImage>,
	/// Whether `src_image`/`dst_image` are freshly (re)created and still in
	/// `UNDEFINED` layout. Cleared after the first scale so reused frames take
	/// a `GENERAL -> GENERAL` barrier instead of an invalid `UNDEFINED` one.
	src_fresh: bool,
	dst_fresh: bool,
	staging: Option<StagingBuffer>,
	sampler: vk::Sampler,
	descriptor_set_layout: vk::DescriptorSetLayout,
	pipeline_layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
	descriptor_pool: vk::DescriptorPool,
	descriptor_set: Option<vk::DescriptorSet>,
	command_pool: vk::CommandPool,
	command_buffer: vk::CommandBuffer,
	fence: vk::Fence,
	dumped: std::sync::atomic::AtomicBool,
	dumped_dst: std::sync::atomic::AtomicBool,
}

// Push-constant layout must match the shader's `Params` block (10 u32 words).
const PUSH_SIZE: u32 = 40;

/// Build push constants for an aspect-preserving fit of `src` into `dst`.
///
/// Returns `[src_w, src_h, dst_w, dst_h, inv_out_w, inv_out_h, off_x, off_y,
/// out_w, out_h]`. The fitted rect is centered; the shader paints the area
/// outside it black (letterbox) instead of stretching the image.
fn letterbox_push(src_w: u32, src_h: u32, target_w: u32, target_h: u32) -> [u32; 10] {
	let scale = ((target_w as f32) / (src_w.max(1) as f32)).min((target_h as f32) / (src_h.max(1) as f32));
	let out_w = (((src_w as f32) * scale).round() as u32).clamp(1, target_w.max(1));
	let out_h = (((src_h as f32) * scale).round() as u32).clamp(1, target_h.max(1));
	let off_x = (target_w.saturating_sub(out_w)) / 2;
	let off_y = (target_h.saturating_sub(out_h)) / 2;
	[
		src_w,
		src_h,
		target_w,
		target_h,
		(1.0f32 / out_w.max(1) as f32).to_bits(),
		(1.0f32 / out_h.max(1) as f32).to_bits(),
		off_x,
		off_y,
		out_w,
		out_h,
	]
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

impl GpuScaler {
	pub fn new(context: VideoContext) -> Result<Self, String> {
		let device = context.device();

		let command_pool = unsafe {
			device.create_command_pool(
				&vk::CommandPoolCreateInfo::default()
					.queue_family_index(context.compute_queue_family())
					.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
				None,
			)
		}
		.map_err(|e| format!("scaler command pool: {e}"))?;

		let command_buffer = unsafe {
			device.allocate_command_buffers(
				&vk::CommandBufferAllocateInfo::default()
					.command_pool(command_pool)
					.level(vk::CommandBufferLevel::PRIMARY)
					.command_buffer_count(1),
			)
		}
		.map_err(|e| format!("scaler command buffer alloc: {e}"))?[0];

		let fence = unsafe {
			device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
		}
		.map_err(|e| format!("scaler fence: {e}"))?;

		let sampler_info = vk::SamplerCreateInfo::default()
			.mag_filter(vk::Filter::LINEAR)
			.min_filter(vk::Filter::LINEAR)
			.address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.mip_lod_bias(0.0)
			.anisotropy_enable(false)
			.compare_enable(false)
			.max_anisotropy(1.0);
		let sampler = unsafe { device.create_sampler(&sampler_info, None) }.map_err(|e| format!("scaler sampler: {e}"))?;

		let bindings = [
			vk::DescriptorSetLayoutBinding::default()
				.binding(0)
				.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
			vk::DescriptorSetLayoutBinding::default()
				.binding(1)
				.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
		];
		let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }
			.map_err(|e| format!("scaler dsl: {e}"))?;

		let push_range = vk::PushConstantRange::default()
			.stage_flags(vk::ShaderStageFlags::COMPUTE)
			.offset(0)
			.size(PUSH_SIZE);
		let pipeline_layout = unsafe {
			device.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(std::slice::from_ref(&descriptor_set_layout))
					.push_constant_ranges(std::slice::from_ref(&push_range)),
				None,
			)
		}
		.map_err(|e| format!("scaler pipeline layout: {e}"))?;

		let spirv_words = spirv_words(SCALER_SPIRV_BYTES);
		let shader_info = vk::ShaderModuleCreateInfo::default().code(&spirv_words);
		let shader_module = unsafe { device.create_shader_module(&shader_info, None) }.map_err(|e| format!("scaler shader module: {e}"))?;
		let entry_point = std::ffi::CString::new("main").expect("valid entry point");
		let stage_info = vk::PipelineShaderStageCreateInfo::default()
			.stage(vk::ShaderStageFlags::COMPUTE)
			.module(shader_module)
			.name(&entry_point);
		let pipeline_info = vk::ComputePipelineCreateInfo::default().stage(stage_info).layout(pipeline_layout);
		let pipeline = unsafe {
			device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
		}
		.map_err(|(_, e)| format!("scaler compute pipeline: {e:?}"))?[0];
		unsafe { device.destroy_shader_module(shader_module, None) };

		let descriptor_pool = unsafe {
			device.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo::default()
					.max_sets(1)
					.					pool_sizes(&[
						vk::DescriptorPoolSize::default()
							.ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
							.descriptor_count(1),
						vk::DescriptorPoolSize::default()
							.ty(vk::DescriptorType::STORAGE_IMAGE)
							.descriptor_count(1),
					]),
				None,
			)
		}
		.map_err(|e| format!("scaler descriptor pool: {e}"))?;

		Ok(Self {
			context,
			src_image: None,
			dst_image: None,
			src_fresh: false,
			dst_fresh: false,
			staging: None,
			sampler,
			descriptor_set_layout,
			pipeline_layout,
			pipeline,
			descriptor_pool,
			descriptor_set: None,
			command_pool,
			command_buffer,
			fence,
			dumped: std::sync::atomic::AtomicBool::new(false),
			dumped_dst: std::sync::atomic::AtomicBool::new(false),
		})
	}

	fn ensure_src(&mut self, width: u32, height: u32) -> Result<&GpuImage, String> {
		let needs = self
			.src_image
			.as_ref()
			.is_none_or(|i| i.width != width || i.height != height);
		if needs {
			if let Some(old) = self.src_image.take() {
				destroy_image(&self.context, old);
			}
			self.src_image = Some(create_image(
				&self.context,
				width,
				height,
				vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
			)?);
			self.src_fresh = true;
		}
		Ok(self.src_image.as_ref().expect("src just created"))
	}

	fn ensure_dst(&mut self, width: u32, height: u32) -> Result<&GpuImage, String> {
		let needs = self
			.dst_image
			.as_ref()
			.is_none_or(|i| i.width != width || i.height != height);
		if needs {
			if let Some(old) = self.dst_image.take() {
				destroy_image(&self.context, old);
			}
			self.dst_image = Some(create_image(
				&self.context,
				width,
				height,
				vk::ImageUsageFlags::STORAGE
					| vk::ImageUsageFlags::TRANSFER_SRC
					| vk::ImageUsageFlags::SAMPLED,
			)?);
			self.dst_fresh = true;
		}
		Ok(self.dst_image.as_ref().expect("dst just created"))
	}

	/// Return a host-visible staging buffer of at least `size` bytes, reusing
	/// (or growing) the cached one instead of allocating per frame.
	fn ensure_staging(&mut self, size: vk::DeviceSize) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
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
			.map_err(|e| format!("scaler upload buffer: {e}"))?;
			let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
			let mem_type = context_find_host_visible(self, reqs.memory_type_bits)
				.ok_or_else(|| "scaler upload: no HOST_VISIBLE memory".to_string())?;
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
				format!("scaler upload memory: {e}")
			})?;
			if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
				unsafe {
					device.free_memory(memory, None);
					device.destroy_buffer(buffer, None);
				};
				return Err(format!("scaler upload bind: {e}"));
			}
			self.staging = Some(StagingBuffer { buffer, memory, size: reqs.size });
		}
		let s = self.staging.as_ref().expect("staging just created");
		Ok((s.buffer, s.memory))
	}

	pub fn scale(
		&mut self,
		src: *const u8,
		src_w: u32,
		src_h: u32,
		stride: u32,
		target_w: u32,
		target_h: u32,
		swap_rb: bool,
	) -> Result<vk::Image, String> {
		let queue = self.context.compute_queue();
		let fence = self.fence;
		let command_buffer = self.command_buffer;

		// Wait for the previous scaler submit only if one is outstanding (the fence is
		// signaled when idle). A timeout here means the device is wedged: resetting
		// the command buffer underneath in-flight work would be use-after-free, so
		// fail instead of proceeding.
		{
			let device = self.context.device();
			let status = unsafe { device.get_fence_status(fence) };
			match status {
				Ok(true) => {
					// Signaled: nothing pending.
				},
				Ok(false) => {
					if let Err(e) = unsafe { device.wait_for_fences(&[fence], true, 500_000_000) } {
						return Err(format!("scaler: timed out waiting for previous frame: {e:?}"));
					}
				},
				Err(e) => return Err(format!("scaler: get_fence_status failed: {e:?}")),
			}
			if let Err(e) = unsafe { device.reset_fences(&[fence]) } {
				tracing::warn!("scaler: failed to reset fence: {e:?}");
			}
		}

		let src_image = self.ensure_src(src_w, src_h)?.image;
		let dst_image = self.ensure_dst(target_w, target_h)?.image;
		let src_fresh = self.src_fresh;
		let dst_fresh = self.dst_fresh;
		let src_view = self.src_image.as_ref().expect("src image").view;
		let dst_view = self.dst_image.as_ref().expect("dst image").view;

		// Upload the packed source rows into the sampled image via the reused
		// staging buffer (grown on demand, never per-frame allocated).
		let upload_size = src_w as vk::DeviceSize * src_h as vk::DeviceSize * 4;
		let (upload_staging, upload_mem) = self.ensure_staging(upload_size)?;
		let device = self.context.device();
		let mapped = unsafe {
			device.map_memory(upload_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
		}
		.map_err(|e| format!("scaler map upload: {e}"))? as *mut u8;

		let row_pitch = src_w as usize * 4;
		let stride_usize = stride as usize;
		if swap_rb {
			// BGRx -> RGBA swizzle while copying.
			for y in 0..src_h as usize {
				let srow = unsafe { src.add(y * stride_usize) };
				let drow = unsafe { mapped.add(y * row_pitch) };
				for x in 0..src_w as usize {
					let s = unsafe { srow.add(x * 4) };
					let d = unsafe { drow.add(x * 4) };
					unsafe {
						*d.add(0) = *s.add(2); // R <- B
						*d.add(1) = *s.add(1); // G <- G
						*d.add(2) = *s.add(0); // B <- R
						*d.add(3) = *s.add(3); // A <- X
					}
				}
			}
		} else if stride_usize == row_pitch {
			unsafe { std::ptr::copy_nonoverlapping(src, mapped, src_w as usize * src_h as usize * 4) };
		} else {
			for y in 0..src_h as usize {
				let srow = unsafe { src.add(y * stride_usize) };
				let drow = unsafe { mapped.add(y * row_pitch) };
				unsafe { std::ptr::copy_nonoverlapping(srow, drow, row_pitch) };
			}
		}
		// One-shot diagnostic: dump the CPU source pixels (already swizzled into the
		// upload staging buffer while mapped) so we can tell if the *source* capture
		// is black or valid. Gated on MOONSHINE_SCALE_DUMP.
		if std::env::var("MOONSHINE_SCALE_DUMP").is_ok()
			&& !self.dumped.swap(true, std::sync::atomic::Ordering::SeqCst)
		{
			dump_rgba_to_png(mapped, src_w, src_h, row_pitch as u32, "/tmp/moonshine_src_debug.png");
		}
		unsafe { device.unmap_memory(upload_mem) };

		// Reuse a single descriptor set, resetting its pool each frame (the views are
		// recreated when the resolution changes, so the set must be re-populated).
		if self.descriptor_set.is_none() {
			let set_layouts = [self.descriptor_set_layout];
			let alloc_info = vk::DescriptorSetAllocateInfo::default()
				.descriptor_pool(self.descriptor_pool)
				.set_layouts(&set_layouts);
			match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
				Ok(sets) => self.descriptor_set = Some(sets[0]),
				Err(e) => {
					return Err(format!("scaler allocate descriptors: {e:?}"));
				},
			}
		} else if let Err(e) =
			unsafe { device.reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty()) }
		{
			tracing::warn!("scaler: failed to reset descriptor pool: {e:?}");
		}
		let descriptor_set = self.descriptor_set.expect("descriptor set allocated above");
		let src_info = vk::DescriptorImageInfo::default()
			.sampler(self.sampler)
			.image_view(src_view)
			.image_layout(vk::ImageLayout::GENERAL);
		let dst_info = vk::DescriptorImageInfo::default()
			.image_view(dst_view)
			.image_layout(vk::ImageLayout::GENERAL);
		let src_infos = [src_info];
		let dst_infos = [dst_info];
		let bindings = [
			vk::WriteDescriptorSet::default()
				.dst_set(descriptor_set)
				.dst_binding(0)
				.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.descriptor_count(1)
				.image_info(&src_infos),
			vk::WriteDescriptorSet::default()
				.dst_set(descriptor_set)
				.dst_binding(1)
				.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
				.descriptor_count(1)
				.image_info(&dst_infos),
		];
		unsafe { device.update_descriptor_sets(&bindings, &[]) };

		// Freshly (re)created images start in UNDEFINED; reused images rest in
		// GENERAL from the previous frame's release barrier.
		let src_barrier_in = vk::ImageMemoryBarrier::default()
			.old_layout(if src_fresh {
				vk::ImageLayout::UNDEFINED
			} else {
				vk::ImageLayout::GENERAL
			})
			.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(src_image)
			.subresource_range(full_range())
			.src_access_mask(if src_fresh {
				vk::AccessFlags::empty()
			} else {
				vk::AccessFlags::SHADER_READ
			})
			.dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
		// The dst is written by the compute shader as a storage image, so it
		// must be in GENERAL (not TRANSFER_DST_OPTIMAL) at dispatch time, and
		// it stays GENERAL afterwards for the converter to sample.
		let dst_barrier_in = vk::ImageMemoryBarrier::default()
			.old_layout(if dst_fresh {
				vk::ImageLayout::UNDEFINED
			} else {
				vk::ImageLayout::GENERAL
			})
			.new_layout(vk::ImageLayout::GENERAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(dst_image)
			.subresource_range(full_range())
			.src_access_mask(if dst_fresh {
				vk::AccessFlags::empty()
			} else {
				vk::AccessFlags::SHADER_READ
			})
			.dst_access_mask(vk::AccessFlags::SHADER_WRITE);

		let copy = vk::BufferImageCopy::default()
			.buffer_offset(0)
			.buffer_row_length(src_w)
			.buffer_image_height(src_h)
			.image_subresource(vk::ImageSubresourceLayers {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				mip_level: 0,
				base_array_layer: 0,
				layer_count: 1,
			})
			.image_offset(vk::Offset3D::default())
			.image_extent(vk::Extent3D {
				width: src_w,
				height: src_h,
				depth: 1,
			});

		let push = letterbox_push(src_w, src_h, target_w, target_h);

		if let Err(e) = unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES) } {
			return Err(format!("scaler: reset command buffer: {e:?}"));
		}
		let begin_info = vk::CommandBufferBeginInfo::default()
			.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
		if let Err(e) = unsafe { device.begin_command_buffer(command_buffer, &begin_info) } {
			return Err(format!("scaler: begin command buffer: {e:?}"));
		}
		unsafe {
			device.cmd_pipeline_barrier(
				command_buffer,
				if src_fresh {
					vk::PipelineStageFlags::TOP_OF_PIPE
				} else {
					vk::PipelineStageFlags::COMPUTE_SHADER
				},
				vk::PipelineStageFlags::TRANSFER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[src_barrier_in],
			);
			device.cmd_copy_buffer_to_image(command_buffer, upload_staging, src_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);
			device.cmd_pipeline_barrier(
				command_buffer,
				vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[vk::ImageMemoryBarrier::default()
					.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.new_layout(vk::ImageLayout::GENERAL)
					.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
					.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
					.image(src_image)
					.subresource_range(full_range())
					.src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
					.dst_access_mask(vk::AccessFlags::SHADER_READ),
					dst_barrier_in],
			);
			device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
			device.cmd_bind_descriptor_sets(
				command_buffer,
				vk::PipelineBindPoint::COMPUTE,
				self.pipeline_layout,
				0,
				&[descriptor_set],
				&[],
			);
			let mut push_bytes = [0u8; PUSH_SIZE as usize];
			for (i, w) in push.iter().enumerate() {
				push_bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
			}
			device.cmd_push_constants(command_buffer, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, &push_bytes);
			device.cmd_dispatch(command_buffer, target_w.div_ceil(32), target_h.div_ceil(8), 1);
			// Release barrier leaving dst in GENERAL: makes the shader write
			// visible to the converter's acquire, and keeps the layout stable
			// so the next frame's barrier can assume GENERAL.
			device.cmd_pipeline_barrier(
				command_buffer,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::PipelineStageFlags::BOTTOM_OF_PIPE,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[vk::ImageMemoryBarrier::default()
					.old_layout(vk::ImageLayout::GENERAL)
					.new_layout(vk::ImageLayout::GENERAL)
					.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
					.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
					.image(dst_image)
					.subresource_range(full_range())
					.src_access_mask(vk::AccessFlags::SHADER_WRITE)
					.dst_access_mask(vk::AccessFlags::SHADER_READ)],
			);
			let end_res = device.end_command_buffer(command_buffer);
			if let Err(e) = end_res {
				return Err(format!("scaler end command buffer: {e}"));
			}
			device
				.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[command_buffer])], fence)
				.map_err(|e| format!("scaler submit: {e}"))?;
			let wait_res = device.wait_for_fences(&[fence], true, 2_000_000_000);
			match wait_res {
				Ok(_) => tracing::debug!("scaler: fence signaled OK"),
				Err(e) => {
					return Err(format!("scaler: fence wait failed: {e:?}"));
				},
			}
		}

		// One-shot diagnostic: read back the scaled dst image and dump a PNG so we
		// can tell whether the *scaled output* is black or correct. Gated on
		// MOONSHINE_SCALE_DUMP. The dst rests in GENERAL (see the release barrier
		// above); the dump helper transitions it back to GENERAL when done.
		if std::env::var("MOONSHINE_SCALE_DUMP").is_ok() && !self.dumped_dst.swap(true, std::sync::atomic::Ordering::SeqCst) {
			dump_gpu_image_to_png(&self.context, dst_image, target_w, target_h);
		}

		self.src_fresh = false;
		self.dst_fresh = false;

		Ok(dst_image)
	}
}

fn context_find_host_visible(scaler: &GpuScaler, bits: u32) -> Option<u32> {
	scaler.context.find_memory_type(
		bits,
		vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
	)
}

impl Drop for GpuScaler {
	fn drop(&mut self) {
		let device = self.context.device();
		if let Some(t) = self.src_image.take() {
			destroy_image(&self.context, t);
		}
		if let Some(t) = self.dst_image.take() {
			destroy_image(&self.context, t);
		}
		if let Some(s) = self.staging.take() {
			unsafe {
				device.free_memory(s.memory, None);
				device.destroy_buffer(s.buffer, None);
			}
		}
		let fence = self.fence;
		let command_buffer = self.command_buffer;
		let command_pool = self.command_pool;
		if let Err(e) = unsafe { device.wait_for_fences(&[fence], true, 5_000_000_000) } {
			tracing::warn!("scaler drop: timed out waiting for fence: {e:?}");
		}
		unsafe {
			if let Some(set) = self.descriptor_set.take() {
				let _ = device.free_descriptor_sets(self.descriptor_pool, &[set]);
			}
			device.destroy_fence(fence, None);
			device.free_command_buffers(command_pool, &[command_buffer]);
			device.destroy_command_pool(command_pool, None);
			device.destroy_descriptor_pool(self.descriptor_pool, None);
			device.destroy_pipeline(self.pipeline, None);
			device.destroy_pipeline_layout(self.pipeline_layout, None);
			device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
			device.destroy_sampler(self.sampler, None);
		};
	}
}

/// GPU image-to-image bilinear scaler. Samples an imported (DMA-BUF or CPU-uploaded)
/// source image and scales it into a device-local RGBA target using the same compute
/// shader as [`GpuScaler`]. Letterboxes to preserve aspect ratio.
pub(crate) struct GpuImageScaler {
	context: VideoContext,
	target: Option<GpuImage>,
	target_fresh: bool,
	sampler: vk::Sampler,
	descriptor_set_layout: vk::DescriptorSetLayout,
	pipeline_layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
	descriptor_pool: vk::DescriptorPool,
	descriptor_set: Option<vk::DescriptorSet>,
	/// Source view from the previous submit, kept alive until that submit's
	/// fence signals (the descriptor references it during GPU execution).
	pending_view: Option<vk::ImageView>,
	command_pool: vk::CommandPool,
	command_buffer: vk::CommandBuffer,
	fence: vk::Fence,
}

impl GpuImageScaler {
	pub fn new(context: VideoContext) -> Result<Self, String> {
		let device = context.device();

		let command_pool = unsafe {
			device.create_command_pool(
				&vk::CommandPoolCreateInfo::default()
					.queue_family_index(context.compute_queue_family())
					.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
				None,
			)
		}
		.map_err(|e| format!("image scaler command pool: {e}"))?;

		let command_buffer = unsafe {
			device.allocate_command_buffers(
				&vk::CommandBufferAllocateInfo::default()
					.command_pool(command_pool)
					.level(vk::CommandBufferLevel::PRIMARY)
					.command_buffer_count(1),
			)
		}
		.map_err(|e| format!("image scaler command buffer alloc: {e}"))?[0];

		let fence = unsafe {
			device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
		}
		.map_err(|e| format!("image scaler fence: {e}"))?;

		let sampler_info = vk::SamplerCreateInfo::default()
			.mag_filter(vk::Filter::LINEAR)
			.min_filter(vk::Filter::LINEAR)
			.address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.mip_lod_bias(0.0)
			.anisotropy_enable(false)
			.compare_enable(false)
			.max_anisotropy(1.0);
		let sampler = unsafe { device.create_sampler(&sampler_info, None) }.map_err(|e| format!("image scaler sampler: {e}"))?;

		let bindings = [
			vk::DescriptorSetLayoutBinding::default()
				.binding(0)
				.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
			vk::DescriptorSetLayoutBinding::default()
				.binding(1)
				.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::COMPUTE),
		];
		let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }
			.map_err(|e| format!("image scaler dsl: {e}"))?;

		let push_range = vk::PushConstantRange::default()
			.stage_flags(vk::ShaderStageFlags::COMPUTE)
			.offset(0)
			.size(PUSH_SIZE);
		let pipeline_layout = unsafe {
			device.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::default()
					.set_layouts(std::slice::from_ref(&descriptor_set_layout))
					.push_constant_ranges(std::slice::from_ref(&push_range)),
				None,
			)
		}
		.map_err(|e| format!("image scaler pipeline layout: {e}"))?;

		let spirv_words = spirv_words(SCALER_SPIRV_BYTES);
		let shader_info = vk::ShaderModuleCreateInfo::default().code(&spirv_words);
		let shader_module = unsafe { device.create_shader_module(&shader_info, None) }.map_err(|e| format!("image scaler shader module: {e}"))?;
		let entry_point = std::ffi::CString::new("main").expect("valid entry point");
		let stage_info = vk::PipelineShaderStageCreateInfo::default()
			.stage(vk::ShaderStageFlags::COMPUTE)
			.module(shader_module)
			.name(&entry_point);
		let pipeline_info = vk::ComputePipelineCreateInfo::default().stage(stage_info).layout(pipeline_layout);
		let pipeline = unsafe {
			device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
		}
		.map_err(|(_, e)| format!("image scaler compute pipeline: {e:?}"))?[0];
		unsafe { device.destroy_shader_module(shader_module, None) };

		let descriptor_pool = unsafe {
			device.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo::default()
					.max_sets(1)
					.pool_sizes(&[
						vk::DescriptorPoolSize::default()
							.ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
							.descriptor_count(1),
						vk::DescriptorPoolSize::default()
							.ty(vk::DescriptorType::STORAGE_IMAGE)
							.descriptor_count(1),
					]),
				None,
			)
		}
		.map_err(|e| format!("image scaler descriptor pool: {e}"))?;

		Ok(Self {
			context,
			target: None,
			target_fresh: false,
			sampler,
			descriptor_set_layout,
			pipeline_layout,
			pipeline,
			descriptor_pool,
			descriptor_set: None,
			pending_view: None,
			command_pool,
			command_buffer,
			fence,
		})
	}

	fn ensure_target(&mut self, width: u32, height: u32) -> Result<&GpuImage, String> {
		let needs = self
			.target
			.as_ref()
			.is_none_or(|t| t.width != width || t.height != height);
		if needs {
			if let Some(old) = self.target.take() {
				destroy_image(&self.context, old);
			}
			self.target = Some(create_image(
				&self.context,
				width,
				height,
				vk::ImageUsageFlags::STORAGE
					| vk::ImageUsageFlags::TRANSFER_SRC
					| vk::ImageUsageFlags::TRANSFER_DST
					| vk::ImageUsageFlags::SAMPLED,
			)?);
			self.target_fresh = true;
		}
		Ok(self.target.as_ref().expect("target just created"))
	}

	pub fn scale(
		&mut self,
		src: vk::Image,
		src_format: vk::Format,
		src_layout: vk::ImageLayout,
		src_w: u32,
		src_h: u32,
		target_w: u32,
		target_h: u32,
	) -> Result<vk::Image, String> {
		let queue = self.context.compute_queue();
		let fence = self.fence;
		let command_buffer = self.command_buffer;

		{
			let device = self.context.device();
			if let Err(e) = unsafe { device.wait_for_fences(&[fence], true, 5_000_000_000) } {
				return Err(format!("image scaler: timed out waiting for previous frame: {e:?}"));
			}
			if let Err(e) = unsafe { device.reset_fences(&[fence]) } {
				tracing::warn!("image scaler: failed to reset fence: {e:?}");
			}
			// The previous submit has completed: its source view is no longer
			// referenced by the GPU and can be destroyed now.
			if let Some(old_view) = self.pending_view.take() {
				unsafe { device.destroy_image_view(old_view, None) };
			}
		}

		let target = self.ensure_target(target_w, target_h)?;
		let target_image = target.image;
		let dst_view = target.view;
		let target_fresh = self.target_fresh;
		let device = self.context.device();

		// Reuse a single descriptor set, resetting its pool each frame.
		if self.descriptor_set.is_none() {
			let set_layouts = [self.descriptor_set_layout];
			let alloc_info = vk::DescriptorSetAllocateInfo::default()
				.descriptor_pool(self.descriptor_pool)
				.set_layouts(&set_layouts);
			match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
				Ok(sets) => self.descriptor_set = Some(sets[0]),
				Err(e) => return Err(format!("image scaler allocate descriptors: {e:?}")),
			}
		} else if let Err(e) =
			unsafe { device.reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty()) }
		{
			tracing::warn!("image scaler: failed to reset descriptor pool: {e:?}");
		}
		let descriptor_set = self.descriptor_set.expect("descriptor set allocated above");

		// Create a sampler view over the imported source image. The view format must
		// match the actual source image format (the imported DMA-BUF / uploaded image).
		let src_view_info = vk::ImageViewCreateInfo::default()
			.image(src)
			.view_type(vk::ImageViewType::TYPE_2D)
			.format(src_format)
			.subresource_range(full_range());
		let src_view = unsafe { device.create_image_view(&src_view_info, None) }.map_err(|e| format!("image scaler src view: {e}"))?;

		let src_info = vk::DescriptorImageInfo::default()
			.sampler(self.sampler)
			.image_view(src_view)
			.image_layout(vk::ImageLayout::GENERAL);
		let dst_info = vk::DescriptorImageInfo::default()
			.image_view(dst_view)
			.image_layout(vk::ImageLayout::GENERAL);
		let src_infos = [src_info];
		let dst_infos = [dst_info];
		let bindings = [
			vk::WriteDescriptorSet::default()
				.dst_set(descriptor_set)
				.dst_binding(0)
				.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.descriptor_count(1)
				.image_info(&src_infos),
			vk::WriteDescriptorSet::default()
				.dst_set(descriptor_set)
				.dst_binding(1)
				.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
				.descriptor_count(1)
				.image_info(&dst_infos),
		];
		unsafe { device.update_descriptor_sets(&bindings, &[]) };

		// The imported source image may be in UNDEFINED (first import) or GENERAL
		// (cached) layout; transition it to GENERAL for sampling.
		//
		// DMA-BUF imports are created with EXCLUSIVE sharing and foreign ownership
		// (see `dmabuf.rs`), so the *first* use must include a queue-family acquire:
		// srcQueueFamilyIndex = QUEUE_FAMILY_EXTERNAL, dstQueueFamilyIndex = our
		// compute family. Without this the device never gains ownership of the
		// external memory and sampling reads uninitialized data (black frames).
		// Cached images already belong to us, so plain IGNORED/IGNORED is correct.
		let needs_acquire = src_layout == vk::ImageLayout::UNDEFINED;
		let src_barrier = vk::ImageMemoryBarrier::default()
			.old_layout(src_layout)
			.new_layout(vk::ImageLayout::GENERAL)
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
			.image(src)
			.subresource_range(full_range())
			.src_access_mask(vk::AccessFlags::empty())
			.dst_access_mask(vk::AccessFlags::SHADER_READ);
		// Freshly (re)created targets start in UNDEFINED; reused targets rest in
		// GENERAL from the previous frame's release barrier. The target is
		// written as a storage image, so it must be GENERAL at dispatch time.
		let dst_barrier_in = vk::ImageMemoryBarrier::default()
			.old_layout(if target_fresh {
				vk::ImageLayout::UNDEFINED
			} else {
				vk::ImageLayout::GENERAL
			})
			.new_layout(vk::ImageLayout::GENERAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(target_image)
			.subresource_range(full_range())
			.src_access_mask(if target_fresh {
				vk::AccessFlags::empty()
			} else {
				vk::AccessFlags::SHADER_READ
			})
			.dst_access_mask(vk::AccessFlags::SHADER_WRITE);
		let dst_barrier_out = vk::ImageMemoryBarrier::default()
			.old_layout(vk::ImageLayout::GENERAL)
			.new_layout(vk::ImageLayout::GENERAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(target_image)
			.subresource_range(full_range())
			.src_access_mask(vk::AccessFlags::SHADER_WRITE)
			.dst_access_mask(vk::AccessFlags::SHADER_READ);

		let push = letterbox_push(src_w, src_h, target_w, target_h);

		if let Err(e) = unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES) } {
			unsafe { device.destroy_image_view(src_view, None) };
			return Err(format!("image scaler: reset command buffer: {e:?}"));
		}
		let begin_info = vk::CommandBufferBeginInfo::default()
			.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
		if let Err(e) = unsafe { device.begin_command_buffer(command_buffer, &begin_info) } {
			unsafe { device.destroy_image_view(src_view, None) };
			return Err(format!("image scaler: begin command buffer: {e:?}"));
		}
		unsafe {
			device.cmd_pipeline_barrier(
				command_buffer,
				vk::PipelineStageFlags::TOP_OF_PIPE,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[src_barrier, dst_barrier_in],
			);
			device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
			device.cmd_bind_descriptor_sets(
				command_buffer,
				vk::PipelineBindPoint::COMPUTE,
				self.pipeline_layout,
				0,
				&[descriptor_set],
				&[],
			);
			let mut push_bytes = [0u8; PUSH_SIZE as usize];
			for (i, w) in push.iter().enumerate() {
				push_bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
			}
			device.cmd_push_constants(command_buffer, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, &push_bytes);
			device.cmd_dispatch(command_buffer, target_w.div_ceil(32), target_h.div_ceil(8), 1);
			device.cmd_pipeline_barrier(
				command_buffer,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::PipelineStageFlags::BOTTOM_OF_PIPE,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[dst_barrier_out],
			);
			let end_res = device.end_command_buffer(command_buffer);
			if let Err(e) = end_res {
				device.destroy_image_view(src_view, None);
				return Err(format!("image scaler end command buffer: {e}"));
			}
			device
				.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[command_buffer])], fence)
				.map_err(|e| format!("image scaler submit: {e}"))?;
		}

		// The source view is referenced by the submitted descriptor set for as
		// long as the GPU executes this submit. Retire it on the next frame
		// (after the fence signals) instead of destroying it while in flight.
		self.pending_view = Some(src_view);
		self.target_fresh = false;

		Ok(target_image)
	}
}

impl Drop for GpuImageScaler {
	fn drop(&mut self) {
		if let Some(t) = self.target.take() {
			destroy_image(&self.context, t);
		}
		let device = self.context.device();
		let fence = self.fence;
		let command_buffer = self.command_buffer;
		let command_pool = self.command_pool;
		if let Err(e) = unsafe { device.wait_for_fences(&[fence], true, 5_000_000_000) } {
			tracing::warn!("image scaler drop: timed out waiting for fence: {e:?}");
		}
		unsafe {
			if let Some(view) = self.pending_view.take() {
				device.destroy_image_view(view, None);
			}
			if let Some(set) = self.descriptor_set.take() {
				let _ = device.free_descriptor_sets(self.descriptor_pool, &[set]);
			}
			device.destroy_descriptor_pool(self.descriptor_pool, None);
			device.destroy_pipeline_layout(self.pipeline_layout, None);
			device.destroy_pipeline(self.pipeline, None);
			device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
			device.destroy_sampler(self.sampler, None);
			device.destroy_fence(fence, None);
			device.free_command_buffers(command_pool, &[command_buffer]);
			device.destroy_command_pool(command_pool, None);
		};
	}
}

/// One-shot diagnostic: write RGBA rows from a CPU pointer to a PNG. Used to tell
/// whether the *source* capture is black or valid. `row_pitch` is bytes per row.
fn dump_rgba_to_png(src: *const u8, width: u32, height: u32, row_pitch: u32, path: &str) {
	let rp = row_pitch as usize;
	let mut rgba = vec![0u8; (width as usize * 4) * height as usize];
	for y in 0..height as usize {
		let srow = unsafe { src.add(y * rp) };
		let drow = &mut rgba[y * width as usize * 4..(y + 1) * width as usize * 4];
		unsafe { std::ptr::copy_nonoverlapping(srow, drow.as_mut_ptr(), width as usize * 4) };
	}
	match image::RgbaImage::from_raw(width, height, rgba) {
		Some(img) => match img.save(path) {
			Ok(_) => tracing::info!("src dump: wrote {path} ({}x{})", width, height),
			Err(e) => tracing::warn!("src dump: PNG save failed: {e}"),
		},
		None => tracing::warn!("src dump: from_raw returned None"),
	}
}

/// One-shot diagnostic: copy a device-local R8G8B8A8_UNORM image back to CPU and
/// write a PNG. Used to tell whether the *scaled GPU output* is black or correct.
fn dump_gpu_image_to_png(context: &VideoContext, image: vk::Image, width: u32, height: u32) {
	let path = "/tmp/moonshine_dst_debug.png";
	let device = context.device();
	let size = (width as u64 * height as u64 * 4).max(1);

	let buf = match unsafe {
		device.create_buffer(
			&vk::BufferCreateInfo::default()
				.size(size)
				.usage(vk::BufferUsageFlags::TRANSFER_DST)
				.sharing_mode(vk::SharingMode::EXCLUSIVE),
			None,
		)
	} {
		Ok(b) => b,
		Err(e) => {
			tracing::warn!("dst dump: create buffer failed: {e}");
			return;
		},
	};
	let reqs = unsafe { device.get_buffer_memory_requirements(buf) };
	let mtype = match context.find_memory_type(
		reqs.memory_type_bits,
		vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
	) {
		Some(t) => t,
		None => {
			unsafe { device.destroy_buffer(buf, None) };
			tracing::warn!("dst dump: no host-visible memory");
			return;
		},
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
			tracing::warn!("dst dump: alloc failed: {e}");
			return;
		},
	};
	if let Err(e) = unsafe { device.bind_buffer_memory(buf, mem, 0) } {
		unsafe {
			device.free_memory(mem, None);
			device.destroy_buffer(buf, None);
		};
		tracing::warn!("dst dump: bind failed: {e}");
		return;
	}

	let pool = match unsafe {
		device.create_command_pool(
			&vk::CommandPoolCreateInfo::default()
				.queue_family_index(context.compute_queue_family())
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
			tracing::warn!("dst dump: command pool failed: {e}");
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
			tracing::warn!("dst dump: command buffer failed: {e}");
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
			tracing::warn!("dst dump: fence failed: {e}");
			return;
		},
	};

	// Transition dst (resting in TRANSFER_DST_OPTIMAL after the shader) to GENERAL,
	// then to TRANSFER_SRC_OPTIMAL for the readback copy.
	let to_general = vk::ImageMemoryBarrier::default()
		.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
		.new_layout(vk::ImageLayout::GENERAL)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.image(image)
		.subresource_range(full_range())
		.src_access_mask(vk::AccessFlags::SHADER_WRITE)
		.dst_access_mask(vk::AccessFlags::SHADER_READ);
	let to_src = vk::ImageMemoryBarrier::default()
		.old_layout(vk::ImageLayout::GENERAL)
		.new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.image(image)
		.subresource_range(full_range())
		.src_access_mask(vk::AccessFlags::SHADER_READ)
		.dst_access_mask(vk::AccessFlags::TRANSFER_READ);
	// Back to GENERAL afterwards: the scaler's barriers assume GENERAL as the
	// resting layout on every subsequent frame.
	let back_to_general = vk::ImageMemoryBarrier::default()
		.old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
		.new_layout(vk::ImageLayout::GENERAL)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.image(image)
		.subresource_range(full_range())
		.src_access_mask(vk::AccessFlags::TRANSFER_READ)
		.dst_access_mask(vk::AccessFlags::SHADER_READ);

	let copy = vk::BufferImageCopy::default()
		.buffer_offset(0)
		.buffer_row_length(width)
		.buffer_image_height(height)
		.image_subresource(vk::ImageSubresourceLayers {
			aspect_mask: vk::ImageAspectFlags::COLOR,
			mip_level: 0,
			base_array_layer: 0,
			layer_count: 1,
		})
		.image_offset(vk::Offset3D::default())
		.image_extent(vk::Extent3D { width, height, depth: 1 });

	let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
	let ok = (|| -> bool {
		if let Err(_) = unsafe { device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) } {
			return false;
		}
		if let Err(_) = unsafe { device.begin_command_buffer(cb, &begin_info) } {
			return false;
		}
		unsafe {
			device.cmd_pipeline_barrier(
				cb,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::PipelineStageFlags::BOTTOM_OF_PIPE,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[to_general],
			);
			device.cmd_pipeline_barrier(
				cb,
				vk::PipelineStageFlags::BOTTOM_OF_PIPE,
				vk::PipelineStageFlags::TRANSFER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[to_src],
			);
			device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf, &[copy]);
			device.cmd_pipeline_barrier(
				cb,
				vk::PipelineStageFlags::TRANSFER,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[back_to_general],
			);
		}
		if let Err(_) = unsafe { device.end_command_buffer(cb) } {
			return false;
		}
		let queue = context.compute_queue();
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
		tracing::warn!("dst dump: command submit failed");
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
			tracing::warn!("dst dump: map failed: {e:?}");
			return;
		},
	};
	let row_pitch = width as usize * 4;
	let mut rgba = vec![0u8; row_pitch * height as usize];
	for y in 0..height as usize {
		let srow = unsafe { mapped.add(y * row_pitch) };
		let drow = &mut rgba[y * row_pitch..(y + 1) * row_pitch];
		unsafe { std::ptr::copy_nonoverlapping(srow, drow.as_mut_ptr(), row_pitch) };
	}
	unsafe { device.unmap_memory(mem) };

	unsafe {
		device.destroy_fence(fence, None);
		device.free_command_buffers(pool, &[cb]);
		device.destroy_command_pool(pool, None);
		device.free_memory(mem, None);
		device.destroy_buffer(buf, None);
	};

	match image::RgbaImage::from_raw(width, height, rgba) {
		Some(img) => match img.save(path) {
			Ok(_) => tracing::info!("dst dump: wrote {path} ({}x{})", width, height),
			Err(e) => tracing::warn!("dst dump: PNG save failed: {e}"),
		},
		None => tracing::warn!("dst dump: from_raw returned None"),
	}
}

#[cfg(test)]
mod tests {
	use super::letterbox_push;

	fn unpack(push: [u32; 10]) -> (u32, u32, u32, u32, f32, f32, u32, u32, u32, u32) {
		(
			push[0],
			push[1],
			push[2],
			push[3],
			f32::from_bits(push[4]),
			f32::from_bits(push[5]),
			push[6],
			push[7],
			push[8],
			push[9],
		)
	}

	#[test]
	fn letterbox_same_aspect_is_full_bleed() {
		let (_, _, _, _, inv_w, inv_h, off_x, off_y, out_w, out_h) =
			unpack(letterbox_push(1920, 1080, 1920, 1080));
		assert_eq!((off_x, off_y, out_w, out_h), (0, 0, 1920, 1080));
		assert!((inv_w - 1.0 / 1920.0).abs() < f32::EPSILON);
		assert!((inv_h - 1.0 / 1080.0).abs() < f32::EPSILON);
	}

	#[test]
	fn letterbox_wide_src_in_narrow_dst_bars_top_bottom() {
		// 16:9 source into a 4:3-ish target: fit width, center vertically.
		let (_, _, _, _, inv_w, inv_h, off_x, off_y, out_w, out_h) =
			unpack(letterbox_push(1920, 1080, 1280, 1080));
		assert_eq!(out_w, 1280);
		assert_eq!(out_h, 720);
		assert_eq!(off_x, 0);
		assert_eq!(off_y, (1080 - 720) / 2);
		assert!((inv_w - 1.0 / 1280.0).abs() < f32::EPSILON);
		assert!((inv_h - 1.0 / 720.0).abs() < f32::EPSILON);
	}

	#[test]
	fn letterbox_narrow_src_in_wide_dst_bars_left_right() {
		// 4:3 source into 16:9 target: fit height, center horizontally.
		let (_, _, _, _, _, _, off_x, off_y, out_w, out_h) =
			unpack(letterbox_push(1280, 960, 1920, 1080));
		assert_eq!(out_h, 1080);
		assert_eq!(out_w, 1440);
		assert_eq!(off_x, (1920 - 1440) / 2);
		assert_eq!(off_y, 0);
	}

	#[test]
	fn letterbox_uv_covers_fitted_rect_exactly() {
		// The inverse scale must map the fitted rect (not the whole dst) onto
		// the full source, otherwise the image stretches or crops.
		let (_, _, dst_w, dst_h, inv_w, inv_h, off_x, off_y, out_w, out_h) =
			unpack(letterbox_push(2560, 1440, 1920, 1080));
		assert_eq!((dst_w, dst_h, off_x, off_y), (1920, 1080, 0, 0));
		assert_eq!((out_w, out_h), (1920, 1080));
		let _ = (inv_w, inv_h);
	}
}
