use super::dmabuf::{DmaBufImporter, DmaBufPlane};
use super::gpu_copy::map_dmabuf_plane;
use super::gpu_scaler::{GpuImageScaler, GpuScaler};
use crate::session::desktop::frame::ExportedFrame;
use ash::vk;
use pixelforge::{InputFormat, VideoContext};
pub(crate) enum DesktopFrameResult {
	Ready {
		source_image: vk::Image,
		src_layout: vk::ImageLayout,
		input_format: InputFormat,
		buffer_released: bool,
	},
	Dropped,
	Fatal(String),
}
fn is_8bit_fourcc(fourcc: u32) -> bool {
	matches!(fourcc, 0x34324241 | 0x34324258 | 0x34325241 | 0x34325258)
}
fn is_bgrx_order(fourcc: u32) -> bool {
	matches!(fourcc, 0x34325241 | 0x34325258)
}
fn is_device_lost(e: &str) -> bool {
	e.to_lowercase().contains("device has been lost")
}
pub(crate) struct DesktopFrameHandler {
	context: VideoContext,
	target_width: u32,
	target_height: u32,
	scaler: Option<GpuScaler>,
	image_scaler: Option<GpuImageScaler>,
	dmabuf_importer: Option<DmaBufImporter>,
}
impl DesktopFrameHandler {
	pub(crate) fn new(context: VideoContext, target_width: u32, target_height: u32) -> Self {
		Self {
			context,
			target_width,
			target_height,
			scaler: None,
			image_scaler: None,
			dmabuf_importer: None,
		}
	}
	fn scaler(&mut self) -> Result<&mut GpuScaler, String> {
		if self.scaler.is_none() {
			self.scaler = Some(GpuScaler::new(self.context.clone())?);
		}
		Ok(self.scaler.as_mut().unwrap())
	}
	fn image_scaler(&mut self) -> Result<&mut GpuImageScaler, String> {
		if self.image_scaler.is_none() {
			self.image_scaler = Some(GpuImageScaler::new(self.context.clone())?);
		}
		Ok(self.image_scaler.as_mut().unwrap())
	}
	fn importer(&mut self) -> Result<&mut DmaBufImporter, String> {
		if self.dmabuf_importer.is_none() {
			self.dmabuf_importer = Some(DmaBufImporter::new(self.context.clone())?);
		}
		Ok(self.dmabuf_importer.as_mut().unwrap())
	}
	fn scale_image_to_target(
		&mut self,
		image: vk::Image,
		vk_format: vk::Format,
		src_layout: vk::ImageLayout,
		frame: &ExportedFrame,
	) -> Result<(vk::Image, vk::ImageLayout), String> {
		let (target_w, target_h) = (self.target_width, self.target_height);
		let scaled = self.image_scaler()?.scale(
			image,
			vk_format,
			src_layout,
			frame.width,
			frame.height,
			target_w,
			target_h,
		)?;
		Ok((scaled, vk::ImageLayout::GENERAL))
	}
	fn upload_pixels(
		&mut self,
		ptr: *const u8,
		frame: &ExportedFrame,
		stride: u32,
		mapped_size: usize,
		vk_format: vk::Format,
		bpp: usize,
	) -> Result<vk::Image, String> {
		self.scaler()?
			.upload(ptr, frame.width, frame.height, stride, vk_format, bpp, mapped_size)
	}
	pub(crate) fn handle_frame(
		&mut self,
		frame: &ExportedFrame,
		fourcc: u32,
	) -> (DesktopFrameResult, Option<vk::Format>) {
		let (input_format, import_vk_format) = super::drm_fourcc_to_input(fourcc);
		let Some(plane) = frame.planes.first() else {
			return (DesktopFrameResult::Dropped, None);
		};
		let mapped_ptr = plane.mapped_ptr.map(|p| p.0);
		let size_mismatch = frame.width != self.target_width || frame.height != self.target_height;
		let (target_w, target_h) = (self.target_width, self.target_height);
		let outcome: Result<(vk::Image, vk::ImageLayout, InputFormat), String> = 'frame: {
			if let Some(ptr) = mapped_ptr
				&& size_mismatch
				&& is_8bit_fourcc(fourcc)
			{
				let scaled = self.scaler().and_then(|s| {
					s.scale(
						ptr,
						frame.width,
						frame.height,
						plane.stride,
						target_w,
						target_h,
						is_bgrx_order(fourcc),
						plane.mapped_size,
					)
				});
				match scaled {
					Ok(image) => break 'frame Ok((image, vk::ImageLayout::GENERAL, InputFormat::RGBA)),
					Err(e) if is_device_lost(&e) => break 'frame Err(e),
					Err(e) => tracing::warn!("GPU scale failed: {e}; falling back to upload + image scale."),
				}
			}
			if let Some(ptr) = mapped_ptr {
				let image = match self.upload_pixels(
					ptr,
					frame,
					plane.stride,
					plane.mapped_size,
					import_vk_format,
					input_format.bytes_per_pixel(),
				) {
					Ok(image) => image,
					Err(e) if is_device_lost(&e) => break 'frame Err(format!("GPU upload failed: {e}")),
					Err(e) => break 'frame Err(format!("CPU upload failed: {e}")),
				};
				if size_mismatch {
					self.scale_image_to_target(image, import_vk_format, vk::ImageLayout::GENERAL, frame)
						.map(|(image, layout)| (image, layout, InputFormat::RGBA))
				} else {
					Ok((image, vk::ImageLayout::GENERAL, input_format))
				}
			} else {
				let plane_count = frame.planes.len().min(4);
				let mut planes_buf = [DmaBufPlane {
					fd: 0,
					offset: 0,
					stride: 0,
					modifier: 0,
				}; 4];
				for (i, p) in frame.planes.iter().take(4).enumerate() {
					planes_buf[i] = DmaBufPlane {
						fd: p.fd,
						offset: p.offset,
						stride: p.stride,
						modifier: frame.modifier,
					};
				}
				let planes = &planes_buf[..plane_count];
				let import = self
					.importer()
					.and_then(|i| i.import_or_reuse(planes[0].fd, frame.width, frame.height, import_vk_format, planes));
				match import {
					Ok((source_import, needs_transition)) => {
						let image = source_import.image();
						let imported_layout = if needs_transition {
							vk::ImageLayout::UNDEFINED
						} else {
							vk::ImageLayout::GENERAL
						};
						if size_mismatch {
							self.scale_image_to_target(image, import_vk_format, imported_layout, frame)
								.map(|(image, layout)| (image, layout, InputFormat::RGBA))
						} else {
							Ok((image, imported_layout, input_format))
						}
					},
					Err(e) => {
						tracing::warn!("Failed to import DMA-BUF: {e}; falling back to CPU upload.");
						let Some((ptr, size)) =
							map_dmabuf_plane(planes[0].fd, planes[0].offset, planes[0].stride, frame.height)
						else {
							break 'frame Err("DMA-BUF plane mmap unavailable".to_string());
						};
						let result = (|| -> Result<(vk::Image, vk::ImageLayout, InputFormat), String> {
							if size_mismatch && is_8bit_fourcc(fourcc) {
								let scaled = self.scaler()?.scale(
									ptr,
									frame.width,
									frame.height,
									planes[0].stride,
									target_w,
									target_h,
									is_bgrx_order(fourcc),
									size,
								)?;
								Ok((scaled, vk::ImageLayout::GENERAL, InputFormat::RGBA))
							} else {
								let image = self.upload_pixels(
									ptr,
									frame,
									planes[0].stride,
									size,
									import_vk_format,
									input_format.bytes_per_pixel(),
								)?;
								if size_mismatch {
									self.scale_image_to_target(image, import_vk_format, vk::ImageLayout::GENERAL, frame)
										.map(|(image, layout)| (image, layout, InputFormat::RGBA))
								} else {
									Ok((image, vk::ImageLayout::GENERAL, input_format))
								}
							}
						})();
						unsafe { libc::munmap(ptr as *mut libc::c_void, size) };
						result
					},
				}
			}
		};
		match outcome {
			Ok((source_image, src_layout, input_format)) => (
				DesktopFrameResult::Ready {
					source_image,
					src_layout,
					input_format,
					buffer_released: size_mismatch,
				},
				Some(import_vk_format),
			),
			Err(e) => {
				if is_device_lost(&e) {
					(DesktopFrameResult::Fatal(e), None)
				} else {
					tracing::warn!("Desktop frame dropped: {e}");
					(DesktopFrameResult::Dropped, None)
				}
			},
		}
	}
}
pub(crate) struct InvalidationCoalescer {
	last_invalidation_time: std::time::Instant,
}
impl InvalidationCoalescer {
	const COOLDOWN: std::time::Duration = std::time::Duration::from_millis(500);
	pub(crate) fn new() -> Self {
		Self {
			last_invalidation_time: std::time::Instant::now() - Self::COOLDOWN,
		}
	}
	pub(crate) fn should_honor(&mut self, first: Option<u64>) -> bool {
		if self.last_invalidation_time.elapsed() < Self::COOLDOWN {
			tracing::debug!(
				"Coalescing desktop reference-frame invalidation within {}ms cooldown (pending range: {first:?}); resync IDR already in flight",
				Self::COOLDOWN.as_millis(),
			);
			return false;
		}
		self.last_invalidation_time = std::time::Instant::now();
		true
	}
}
