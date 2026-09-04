use ash::vk;
use pixelforge::VideoContext;

struct CachedImage {
	image: vk::Image,
	memory: vk::DeviceMemory,
	width: u32,
	height: u32,
	format: vk::Format,
	row_pitch: u64,
	first_use: bool,
}

pub(crate) struct CpuUploader {
	context: VideoContext,
	cached: Option<CachedImage>,
}

impl CpuUploader {
	pub fn new(context: VideoContext) -> Self {
		Self { context, cached: None }
	}

	pub fn upload_or_reuse(
		&mut self,
		src: *const u8,
		src_size: usize,
		width: u32,
		height: u32,
		stride: u32,
		format: vk::Format,
	) -> Result<(vk::Image, bool), String> {
		let device = self.context.device();

		let needs_create = self.cached.as_ref().is_none_or(|c| {
			c.width != width || c.height != height || c.format != format
		});
		if needs_create {
			if let Some(old) = self.cached.take() {
				tracing::debug!(
					"CPU upload image changed {}x{:?} -> {}x{:?}, recreating",
					old.width,
					old.format,
					width,
					format,
				);
				unsafe {
					device.destroy_image(old.image, None);
					device.free_memory(old.memory, None);
				}
			}

			let create_info = vk::ImageCreateInfo::default()
				.image_type(vk::ImageType::TYPE_2D)
				.format(format)
				.extent(vk::Extent3D { width, height, depth: 1 })
				.mip_levels(1)
				.array_layers(1)
				.samples(vk::SampleCountFlags::TYPE_1)
				.tiling(vk::ImageTiling::LINEAR)
				.usage(vk::ImageUsageFlags::SAMPLED)
				.sharing_mode(vk::SharingMode::EXCLUSIVE)
				.initial_layout(vk::ImageLayout::UNDEFINED);
			let image = unsafe { device.create_image(&create_info, None) }
				.map_err(|e| format!("CPU upload image: {e}"))?;

			let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
			let mem_type = self
				.context
				.find_memory_type(
					mem_reqs.memory_type_bits,
					vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
				)
				.ok_or_else(|| "CPU upload: no HOST_VISIBLE memory type".to_string())?;
			let memory = unsafe {
				let alloc_info = vk::MemoryAllocateInfo::default()
					.allocation_size(mem_reqs.size)
					.memory_type_index(mem_type);
				device.allocate_memory(&alloc_info, None)
			}
			.map_err(|e| {
				unsafe { device.destroy_image(image, None) };
				format!("CPU upload image memory: {e}")
			})?;
			if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
				unsafe {
					device.free_memory(memory, None);
					device.destroy_image(image, None);
				}
				return Err(format!("CPU upload image bind: {e}"));
			}

			let subresource = vk::ImageSubresource::default()
				.aspect_mask(vk::ImageAspectFlags::COLOR)
				.mip_level(0)
				.array_layer(0);
			let layout = unsafe { device.get_image_subresource_layout(image, subresource) };
			tracing::debug!(
				"CPU upload image: {}x{} {:?}, row_pitch={}, size={}",
				width,
				height,
				format,
				layout.row_pitch,
				mem_reqs.size,
			);

			self.cached = Some(CachedImage {
				image,
				memory,
				width,
				height,
				format,
				row_pitch: layout.row_pitch,
				first_use: true,
			});
		}

		let cached = self.cached.as_mut().expect("CPU upload image just created");
		let dst_ptr = unsafe {
			device.map_memory(cached.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
		}
		.map_err(|e| format!("CPU upload map image: {e}"))?;

		let row_len = (stride as u64).min(cached.row_pitch) as usize;
		let mut src_off = 0usize;
		let mut dst_off = 0u64;
		for _ in 0..height {
			let remaining = src_size.saturating_sub(src_off);
			if remaining == 0 {
				break;
			}
			let len = row_len.min(remaining);
			unsafe {
				std::ptr::copy_nonoverlapping(
					src.add(src_off),
					(dst_ptr as *mut u8).add(dst_off as usize),
					len,
				);
			}
			src_off += stride as usize;
			dst_off += cached.row_pitch;
		}
		unsafe { device.unmap_memory(cached.memory) };

		let needs_transition = cached.first_use;
		cached.first_use = false;
		Ok((cached.image, needs_transition))
	}
}

impl Drop for CpuUploader {
	fn drop(&mut self) {
		if let Some(cached) = self.cached.take() {
			let device = self.context.device();
			unsafe {
				device.destroy_image(cached.image, None);
				device.free_memory(cached.memory, None);
			}
		}
	}
}
