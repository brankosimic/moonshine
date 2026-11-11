use super::dmabuf::{CachedImport, DmaBufImporter, DmaBufPlane};
use super::gpu_copy::map_dmabuf_plane;
use super::gpu_scaler::{GpuImageScaler, GpuScaler};
use crate::session::desktop::frame::ExportedFrame;
use ash::vk;
use pixelforge::{Handoff, RgbFormat, SourceImage, VideoContext};
use std::sync::Arc;
pub(crate) enum DesktopFrameResult {
	Ready {
		source: DesktopSource,
		input_format: RgbFormat,
	},
	Dropped,
	Fatal(String),
}
pub(crate) enum DesktopSource {
	Cached(Arc<CachedImport>),
	Owned(SourceImage),
}
impl DesktopSource {
	pub(crate) fn source(&self) -> &SourceImage {
		match self {
			Self::Cached(cached) => cached.source(),
			Self::Owned(source) => source,
		}
	}
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
	fn own(&self, image: vk::Image) -> Result<DesktopSource, String> {
		SourceImage::new(
			&self.context,
			image,
			Handoff::Device {
				layout: vk::ImageLayout::GENERAL,
			},
		)
		.map(DesktopSource::Owned)
		.map_err(|e| format!("Failed to register the scaled frame as an encoder source: {e}"))
	}
	fn scale_image_to_target(
		&mut self,
		image: vk::Image,
		vk_format: vk::Format,
		src_layout: vk::ImageLayout,
		frame: &ExportedFrame,
	) -> Result<vk::Image, String> {
		let (target_w, target_h) = (self.target_width, self.target_height);
		let scaler = self.image_scaler()?;
		let scaled = scaler.scale(
			image,
			vk_format,
			src_layout,
			frame.width,
			frame.height,
			target_w,
			target_h,
		)?;
		scaler.wait()?;
		Ok(scaled)
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
	fn scale_pixels_to_target(
		&mut self,
		ptr: *const u8,
		frame: &ExportedFrame,
		stride: u32,
		mapped_size: usize,
		fourcc: u32,
	) -> Result<vk::Image, String> {
		let (target_w, target_h) = (self.target_width, self.target_height);
		self.scaler()?.scale(
			ptr,
			frame.width,
			frame.height,
			stride,
			target_w,
			target_h,
			is_bgrx_order(fourcc),
			mapped_size,
		)
	}
	fn upload_and_maybe_scale(
		&mut self,
		ptr: *const u8,
		frame: &ExportedFrame,
		stride: u32,
		mapped_size: usize,
		fourcc: u32,
		import_vk_format: vk::Format,
		input_format: RgbFormat,
		size_mismatch: bool,
	) -> Result<(DesktopSource, RgbFormat), String> {
		if size_mismatch && is_8bit_fourcc(fourcc) {
			let image = self.scale_pixels_to_target(ptr, frame, stride, mapped_size, fourcc)?;
			return Ok((self.own(image)?, RgbFormat::RGBA));
		}
		let image = self.upload_pixels(
			ptr,
			frame,
			stride,
			mapped_size,
			import_vk_format,
			input_format.bytes_per_pixel(),
		)?;
		if size_mismatch {
			let scaled = self.scale_image_to_target(image, import_vk_format, vk::ImageLayout::GENERAL, frame)?;
			Ok((self.own(scaled)?, RgbFormat::RGBA))
		} else {
			Ok((self.own(image)?, input_format))
		}
	}
	pub(crate) fn handle_frame(&mut self, frame: &ExportedFrame, fourcc: u32) -> DesktopFrameResult {
		let (input_format, import_vk_format) = super::drm_fourcc_to_input(fourcc);
		let Some(plane) = frame.planes.first() else {
			return DesktopFrameResult::Dropped;
		};
		let mapped_ptr = plane.mapped_ptr.map(|p| p.0);
		let size_mismatch = frame.width != self.target_width || frame.height != self.target_height;
		let outcome: Result<(DesktopSource, RgbFormat), String> = 'frame: {
			if let Some(ptr) = mapped_ptr {
				match self.upload_and_maybe_scale(
					ptr,
					frame,
					plane.stride,
					plane.mapped_size,
					fourcc,
					import_vk_format,
					input_format,
					size_mismatch,
				) {
					Ok(v) => break 'frame Ok(v),
					Err(e) if is_device_lost(&e) => break 'frame Err(e),
					Err(e) => break 'frame Err(format!("CPU upload failed: {e}")),
				}
			}
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
				Ok(cached) => {
					if size_mismatch {
						let scaled = match self.scale_image_to_target(
							cached.source().image(),
							import_vk_format,
							vk::ImageLayout::UNDEFINED,
							frame,
						) {
							Ok(image) => image,
							Err(e) => break 'frame Err(e),
						};
						let owned = match self.own(scaled) {
							Ok(source) => source,
							Err(e) => break 'frame Err(e),
						};
						Ok((owned, RgbFormat::RGBA))
					} else {
						Ok((DesktopSource::Cached(cached), input_format))
					}
				},
				Err(e) => {
					tracing::warn!("Failed to import DMA-BUF: {e}; falling back to CPU upload.");
					let Some((ptr, size)) =
						map_dmabuf_plane(planes[0].fd, planes[0].offset, planes[0].stride, frame.height)
					else {
						break 'frame Err("DMA-BUF plane mmap unavailable".to_string());
					};
					let result = self.upload_and_maybe_scale(
						ptr,
						frame,
						planes[0].stride,
						size,
						fourcc,
						import_vk_format,
						input_format,
						size_mismatch,
					);
					unsafe { libc::munmap(ptr as *mut libc::c_void, size) };
					result
				},
			}
		};
		match outcome {
			Ok((source, input_format)) => DesktopFrameResult::Ready { source, input_format },
			Err(e) => {
				if is_device_lost(&e) {
					DesktopFrameResult::Fatal(e)
				} else {
					tracing::warn!("Desktop frame dropped: {e}");
					DesktopFrameResult::Dropped
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
