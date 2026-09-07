use ash::vk;
use pixelforge::VideoContext;

struct TargetImage {
	image: vk::Image,
	memory: vk::DeviceMemory,
	width: u32,
	height: u32,
	row_pitch: u64,
}

fn create_target_image(context: &VideoContext, width: u32, height: u32) -> Result<TargetImage, String> {
	let device = context.device();

	let create_info = vk::ImageCreateInfo::default()
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
		.tiling(vk::ImageTiling::LINEAR)
		.usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
		.sharing_mode(vk::SharingMode::EXCLUSIVE)
		.initial_layout(vk::ImageLayout::UNDEFINED);
	let image = unsafe { device.create_image(&create_info, None) }.map_err(|e| format!("scaler target image: {e}"))?;

	let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
	let mem_type = context
		.find_memory_type(
			mem_reqs.memory_type_bits,
			vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
		)
		.ok_or_else(|| "scaler target: no HOST_VISIBLE memory type".to_string())?;
	let memory = unsafe {
		let alloc_info = vk::MemoryAllocateInfo::default()
			.allocation_size(mem_reqs.size)
			.memory_type_index(mem_type);
		device.allocate_memory(&alloc_info, None)
	}
	.map_err(|e| {
		unsafe { device.destroy_image(image, None) };
		format!("scaler target image memory: {e}")
	})?;
	if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
		unsafe {
			device.free_memory(memory, None);
			device.destroy_image(image, None);
		};
		return Err(format!("scaler target image bind: {e}"));
	}

	let subresource = vk::ImageSubresource::default()
		.aspect_mask(vk::ImageAspectFlags::COLOR)
		.mip_level(0)
		.array_layer(0);
	let layout = unsafe { device.get_image_subresource_layout(image, subresource) };

	Ok(TargetImage {
		image,
		memory,
		width,
		height,
		row_pitch: layout.row_pitch,
	})
}

fn destroy_target(context: &VideoContext, target: TargetImage) {
	let device = context.device();
	unsafe {
		device.free_memory(target.memory, None);
		device.destroy_image(target.image, None);
	};
}

pub(crate) struct GpuScaler {
	context: VideoContext,
	target: Option<TargetImage>,
	command_pool: vk::CommandPool,
	command_buffer: vk::CommandBuffer,
	fence: vk::Fence,
}

impl GpuScaler {
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

		let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
			.map_err(|e| format!("scaler fence: {e}"))?;

		Ok(Self {
			context,
			target: None,
			command_pool,
			command_buffer,
			fence,
		})
	}

	fn ensure_target(&mut self, width: u32, height: u32) -> Result<&mut TargetImage, String> {
		let needs_create = self
			.target
			.as_ref()
			.is_none_or(|t| t.width != width || t.height != height);
		if needs_create {
			if let Some(old) = self.target.take() {
				destroy_target(&self.context, old);
			}
			self.target = Some(create_target_image(&self.context, width, height)?);
		}
		Ok(self.target.as_mut().expect("target just created"))
	}

	pub fn scale(
		&mut self,
		src: *const u8,
		src_w: u32,
		src_h: u32,
		stride: u32,
		target_w: u32,
		target_h: u32,
	) -> Result<vk::Image, String> {
		let transfer_queue = self.context.transfer_queue();
		let fence = self.fence;
		let command_buffer = self.command_buffer;
		let (target_image, target_memory, row_pitch) = {
			let t = self.ensure_target(target_w, target_h)?;
			(t.image, t.memory, t.row_pitch as usize)
		};
		let device = self.context.device();
		let target_ptr = unsafe { device.map_memory(target_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
			.map_err(|e| format!("scaler map target: {e}"))? as *mut u8;

		let scale_x = target_w as f64 / src_w.max(1) as f64;
		let scale_y = target_h as f64 / src_h.max(1) as f64;
		let s = scale_x.min(scale_y);
		let out_w = (src_w as f64 * s).round().max(1.0) as u32;
		let out_h = (src_h as f64 * s).round().max(1.0) as u32;
		let off_x = ((target_w - out_w) / 2) as i64;
		let off_y = ((target_h - out_h) / 2) as i64;

		let dst = target_ptr;
		for y in 0..target_h {
			let base = y as usize * row_pitch;
			unsafe {
				std::ptr::write_bytes(dst.add(base), 0, (target_w * 4) as usize);
			}
		}

		let inv_sx = src_w as f64 / out_w.max(1) as f64;
		let inv_sy = src_h as f64 / out_h.max(1) as f64;
		let stride_usize = stride as usize;
		for oy in 0..out_h {
			let sy = (oy as f64 + 0.5) * inv_sy - 0.5;
			let y0 = sy.floor().clamp(0.0, (src_h - 1) as f64) as i64;
			let y1 = (y0 + 1).min(src_h as i64 - 1);
			let fy = sy - y0 as f64;
			for ox in 0..out_w {
				let sx = (ox as f64 + 0.5) * inv_sx - 0.5;
				let x0 = sx.floor().clamp(0.0, (src_w - 1) as f64) as i64;
				let x1 = (x0 + 1).min(src_w as i64 - 1);
				let fx = sx - x0 as f64;

				let p00 = unsafe { src.add((y0 as usize * stride_usize) + (x0 as usize * 4)) };
				let p01 = unsafe { src.add((y0 as usize * stride_usize) + (x1 as usize * 4)) };
				let p10 = unsafe { src.add((y1 as usize * stride_usize) + (x0 as usize * 4)) };
				let p11 = unsafe { src.add((y1 as usize * stride_usize) + (x1 as usize * 4)) };

				for c in 0..4usize {
					let top = unsafe { *p00.add(c) as f64 } * (1.0 - fx) + unsafe { *p01.add(c) as f64 } * fx;
					let bot = unsafe { *p10.add(c) as f64 } * (1.0 - fx) + unsafe { *p11.add(c) as f64 } * fy;
					let v = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
					let dst_idx = ((oy as i64 + off_y) as usize * row_pitch) + ((ox as i64 + off_x) as usize * 4) + c;
					unsafe {
						*dst.add(dst_idx) = v;
					}
				}
			}
		}

		unsafe { device.unmap_memory(target_memory) };

		let wait = unsafe { device.wait_for_fences(&[fence], true, u64::MAX) };
		if let Err(e) = wait {
			return Err(format!("scaler: timed out waiting for previous frame: {e:?}"));
		}
		if let Err(e) = unsafe { device.reset_fences(&[fence]) } {
			tracing::warn!("scaler: failed to reset fence: {e:?}");
		}

		let barrier = vk::ImageMemoryBarrier::default()
			.old_layout(vk::ImageLayout::UNDEFINED)
			.new_layout(vk::ImageLayout::GENERAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(target_image)
			.subresource_range(vk::ImageSubresourceRange {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				base_mip_level: 0,
				level_count: 1,
				base_array_layer: 0,
				layer_count: 1,
			})
			.src_access_mask(vk::AccessFlags::empty())
			.dst_access_mask(vk::AccessFlags::SHADER_READ);

		if let Err(e) =
			unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES) }
		{
			return Err(format!("scaler: reset command buffer: {e:?}"));
		}
		if let Err(e) = unsafe { device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()) } {
			return Err(format!("scaler: begin command buffer: {e:?}"));
		}
		unsafe {
			device.cmd_pipeline_barrier(
				command_buffer,
				vk::PipelineStageFlags::TOP_OF_PIPE,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[barrier],
			);
			device
				.end_command_buffer(command_buffer)
				.map_err(|e| format!("scaler end command buffer: {e}"))?;
			device
				.queue_submit(
					transfer_queue,
					&[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
					fence,
				)
				.map_err(|e| format!("scaler submit: {e}"))?;
		}

		Ok(target_image)
	}
}

impl Drop for GpuScaler {
	fn drop(&mut self) {
		if let Some(t) = self.target.take() {
			destroy_target(&self.context, t);
		}
		let device = self.context.device();
		let fence = self.fence;
		let command_buffer = self.command_buffer;
		let command_pool = self.command_pool;
		if let Err(e) = unsafe { device.wait_for_fences(&[fence], true, u64::MAX) } {
			tracing::warn!("scaler drop: timed out waiting for fence: {e:?}");
		}
		unsafe {
			device.destroy_fence(fence, None);
			device.free_command_buffers(command_pool, &[command_buffer]);
			device.destroy_command_pool(command_pool, None);
		};
	}
}

pub(crate) struct GpuImageScaler {
	context: VideoContext,
	target: Option<TargetImage>,
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
					.queue_family_index(context.transfer_queue_family())
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

		let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
			.map_err(|e| format!("image scaler fence: {e}"))?;

		Ok(Self {
			context,
			target: None,
			command_pool,
			command_buffer,
			fence,
		})
	}

	fn ensure_target(&mut self, width: u32, height: u32) -> Result<&mut TargetImage, String> {
		let needs_create = self
			.target
			.as_ref()
			.is_none_or(|t| t.width != width || t.height != height);
		if needs_create {
			if let Some(old) = self.target.take() {
				destroy_target(&self.context, old);
			}
			self.target = Some(create_target_image(&self.context, width, height)?);
		}
		Ok(self.target.as_mut().expect("target just created"))
	}

	pub fn scale(
		&mut self,
		src: vk::Image,
		src_layout: vk::ImageLayout,
		src_w: u32,
		src_h: u32,
		target_w: u32,
		target_h: u32,
	) -> Result<vk::Image, String> {
		let transfer_queue = self.context.transfer_queue();
		let fence = self.fence;
		let command_buffer = self.command_buffer;
		let target_image = {
			let t = self.ensure_target(target_w, target_h)?;
			t.image
		};
		let device = self.context.device();

		let scale_x = target_w as f64 / src_w.max(1) as f64;
		let scale_y = target_h as f64 / src_h.max(1) as f64;
		let s = scale_x.min(scale_y);
		let out_w = (src_w as f64 * s).round().max(1.0) as u32;
		let out_h = (src_h as f64 * s).round().max(1.0) as u32;
		let off_x = ((target_w - out_w) / 2) as i32;
		let off_y = ((target_h - out_h) / 2) as i32;

		if let Err(e) = unsafe { device.wait_for_fences(&[fence], true, u64::MAX) } {
			return Err(format!("image scaler: timed out waiting for previous frame: {e:?}"));
		}
		if let Err(e) = unsafe { device.reset_fences(&[fence]) } {
			tracing::warn!("image scaler: failed to reset fence: {e:?}");
		}

		let target_barrier_in = vk::ImageMemoryBarrier::default()
			.old_layout(vk::ImageLayout::UNDEFINED)
			.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(target_image)
			.subresource_range(vk::ImageSubresourceRange {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				base_mip_level: 0,
				level_count: 1,
				base_array_layer: 0,
				layer_count: 1,
			})
			.src_access_mask(vk::AccessFlags::empty())
			.dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

		let target_barrier_out = vk::ImageMemoryBarrier::default()
			.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
			.new_layout(vk::ImageLayout::GENERAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(target_image)
			.subresource_range(vk::ImageSubresourceRange {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				base_mip_level: 0,
				level_count: 1,
				base_array_layer: 0,
				layer_count: 1,
			})
			.src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
			.dst_access_mask(vk::AccessFlags::SHADER_READ);

		let copy_region = vk::ImageCopy {
			src_subresource: vk::ImageSubresourceLayers {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				mip_level: 0,
				base_array_layer: 0,
				layer_count: 1,
			},
			src_offset: vk::Offset3D::default(),
			dst_subresource: vk::ImageSubresourceLayers {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				mip_level: 0,
				base_array_layer: 0,
				layer_count: 1,
			},
			dst_offset: vk::Offset3D {
				x: off_x,
				y: off_y,
				z: 0,
			},
			extent: vk::Extent3D {
				width: out_w,
				height: out_h,
				depth: 1,
			},
		};

		if let Err(e) =
			unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES) }
		{
			return Err(format!("image scaler: reset command buffer: {e:?}"));
		}
		if let Err(e) = unsafe { device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()) } {
			return Err(format!("image scaler: begin command buffer: {e:?}"));
		}
		unsafe {
			device.cmd_pipeline_barrier(
				command_buffer,
				vk::PipelineStageFlags::TOP_OF_PIPE,
				vk::PipelineStageFlags::TRANSFER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[target_barrier_in],
			);
			device.cmd_copy_image(
				command_buffer,
				src,
				src_layout,
				target_image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&[copy_region],
			);
			device.cmd_pipeline_barrier(
				command_buffer,
				vk::PipelineStageFlags::TRANSFER,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[target_barrier_out],
			);
			device
				.end_command_buffer(command_buffer)
				.map_err(|e| format!("image scaler end command buffer: {e}"))?;
			device
				.queue_submit(
					transfer_queue,
					&[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
					fence,
				)
				.map_err(|e| format!("image scaler submit: {e}"))?;
		}

		Ok(target_image)
	}
}

impl Drop for GpuImageScaler {
	fn drop(&mut self) {
		if let Some(t) = self.target.take() {
			destroy_target(&self.context, t);
		}
		let device = self.context.device();
		let fence = self.fence;
		let command_buffer = self.command_buffer;
		let command_pool = self.command_pool;
		if let Err(e) = unsafe { device.wait_for_fences(&[fence], true, u64::MAX) } {
			tracing::warn!("image scaler drop: timed out waiting for fence: {e:?}");
		}
		unsafe {
			device.destroy_fence(fence, None);
			device.free_command_buffers(command_pool, &[command_buffer]);
			device.destroy_command_pool(command_pool, None);
		};
	}
}
