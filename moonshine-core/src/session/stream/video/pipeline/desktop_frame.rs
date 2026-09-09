use ash::vk;
use pixelforge::{InputFormat, VideoContext};

use super::dmabuf::{DmaBufImporter, DmaBufPlane};
use super::gpu_copy::map_dmabuf_plane;
use super::gpu_scaler::{GpuImageScaler, GpuScaler};
use crate::session::compositor::frame::ExportedFrame;

pub(crate) enum DesktopFrameResult {
	Ready {
		source_image: vk::Image,
		src_layout: vk::ImageLayout,
		input_format: InputFormat,
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

	pub(crate) fn handle_frame(
		&mut self,
		frame: &ExportedFrame,
		fourcc: u32,
	) -> (DesktopFrameResult, Option<vk::Format>) {
		let (mut input_format, import_vk_format) = super::drm_fourcc_to_input(fourcc);

		let plane = match frame.planes.first() {
			Some(p) => p,
			None => return (DesktopFrameResult::Dropped, None),
		};
		let mapped_ptr = plane.mapped_ptr.map(|p| p.0);

		let size_mismatch = frame.width != self.target_width || frame.height != self.target_height;

		let (source_image, src_layout);

		let scaler = match &mut self.scaler {
			Some(s) => s,
			None => match GpuScaler::new(self.context.clone()) {
				Ok(s) => {
					self.scaler = Some(s);
					self.scaler.as_mut().unwrap()
				},
				Err(e) => {
					tracing::warn!("Failed to create GPU scaler: {e}");
					return (DesktopFrameResult::Dropped, None);
				},
			},
		};

		if let Some(src_ptr) = mapped_ptr {
			if size_mismatch && is_8bit_fourcc(fourcc) {
				let swap_rb = is_bgrx_order(fourcc);
				let t_scale = std::time::Instant::now();
				match scaler.scale(
					src_ptr,
					frame.width,
					frame.height,
					plane.stride,
					self.target_width,
					self.target_height,
					swap_rb,
					plane.mapped_size,
				) {
					Ok(scaled_image) => {
						tracing::debug!(scale_us = t_scale.elapsed().as_micros() as u64, "scale done");
						source_image = scaled_image;
						src_layout = vk::ImageLayout::GENERAL;
						input_format = InputFormat::RGBA;
					},
					Err(e) => {
						if e.to_lowercase().contains("device has been lost") {
							return (DesktopFrameResult::Fatal(format!("GPU scale failed: {e}")), None);
						}
						tracing::warn!("GPU scale failed: {e}; encoding at native size.");
						match scaler.upload(
							src_ptr,
							frame.width,
							frame.height,
							plane.stride,
							import_vk_format,
							input_format.bytes_per_pixel(),
							plane.mapped_size,
						) {
							Ok(img) => {
								source_image = img;
								src_layout = vk::ImageLayout::GENERAL;
							},
							Err(e) => {
								tracing::warn!("CPU upload failed: {e}");
								return (DesktopFrameResult::Dropped, None);
							},
						}
					},
				}
			} else {
				match scaler.upload(
					src_ptr,
					frame.width,
					frame.height,
					plane.stride,
					import_vk_format,
					input_format.bytes_per_pixel(),
					plane.mapped_size,
				) {
					Ok(img) => {
						source_image = img;
						src_layout = vk::ImageLayout::GENERAL;
					},
					Err(e) => {
						if e.to_lowercase().contains("device has been lost") {
							return (DesktopFrameResult::Fatal(format!("GPU upload failed: {e}")), None);
						}
						tracing::warn!("CPU upload failed: {e}");
						return (DesktopFrameResult::Dropped, None);
					},
				}
			}
		} else {
			let importer = match &mut self.dmabuf_importer {
				Some(imp) => imp,
				None => match DmaBufImporter::new(self.context.clone()) {
					Ok(imp) => {
						self.dmabuf_importer = Some(imp);
						self.dmabuf_importer.as_mut().unwrap()
					},
					Err(e) => {
						tracing::warn!("Failed to create DMA-BUF importer: {e}");
						return (DesktopFrameResult::Dropped, None);
					},
				},
			};

			let mut planes_buf = [DmaBufPlane {
				fd: 0,
				offset: 0,
				stride: 0,
				modifier: 0,
			}; 4];
			let plane_count = frame.planes.len().min(4);
			for (i, p) in frame.planes.iter().take(4).enumerate() {
				planes_buf[i] = DmaBufPlane {
					fd: p.fd,
					offset: p.offset,
					stride: p.stride,
					modifier: frame.modifier,
				};
			}
			let planes = &planes_buf[..plane_count];

			match importer.import_or_reuse(planes[0].fd, frame.width, frame.height, import_vk_format, planes) {
				Ok((source_img, needs_transition)) => {
					let imported_layout = if needs_transition {
						vk::ImageLayout::UNDEFINED
					} else {
						vk::ImageLayout::GENERAL
					};
					if size_mismatch {
						let img_scaler = match &mut self.image_scaler {
							Some(s) => s,
							None => match GpuImageScaler::new(self.context.clone()) {
								Ok(s) => {
									self.image_scaler = Some(s);
									self.image_scaler.as_mut().unwrap()
								},
								Err(e) => {
									tracing::warn!("Failed to create GPU image scaler: {e}");
									return (DesktopFrameResult::Dropped, None);
								},
							},
						};
						match img_scaler.scale(
							source_img,
							import_vk_format,
							imported_layout,
							frame.width,
							frame.height,
							self.target_width,
							self.target_height,
						) {
							Ok(scaled_image) => {
								source_image = scaled_image;
								src_layout = vk::ImageLayout::GENERAL;
								input_format = InputFormat::RGBA;
							},
							Err(e) => {
								if e.to_lowercase().contains("device has been lost") {
									return (DesktopFrameResult::Fatal(format!("GPU scale failed: {e}")), None);
								}
								tracing::warn!("GPU scale failed: {e}; encoding at native size.");
								source_image = source_img;
								src_layout = imported_layout;
							},
						}
					} else {
						source_image = source_img;
						src_layout = imported_layout;
					}
				},
				Err(e) => {
					tracing::warn!("Failed to import DMA-BUF: {e}; falling back to CPU upload.");
					match map_dmabuf_plane(planes[0].fd, planes[0].offset, plane.stride, frame.height) {
						Some((ptr, size)) => {
							let result = if size_mismatch && is_8bit_fourcc(fourcc) {
								scaler
									.scale(
										ptr,
										frame.width,
										frame.height,
										plane.stride,
										self.target_width,
										self.target_height,
										is_bgrx_order(fourcc),
										size,
									)
									.map(|img| (img, InputFormat::RGBA))
							} else {
								scaler
									.upload(
										ptr,
										frame.width,
										frame.height,
										plane.stride,
										import_vk_format,
										input_format.bytes_per_pixel(),
										size,
									)
									.map(|img| (img, input_format))
							};
							unsafe { libc::munmap(ptr as *mut libc::c_void, size) };
							match result {
								Ok((img, fmt)) => {
									source_image = img;
									src_layout = vk::ImageLayout::GENERAL;
									input_format = fmt;
								},
								Err(e) => {
									if e.to_lowercase().contains("device has been lost") {
										return (
											DesktopFrameResult::Fatal(format!("DMA-BUF fallback upload failed: {e}")),
											None,
										);
									}
									tracing::warn!("DMA-BUF CPU upload failed: {e}");
									return (DesktopFrameResult::Dropped, None);
								},
							}
						},
						None => {
							tracing::warn!("DMA-BUF plane mmap unavailable; dropping frame.");
							return (DesktopFrameResult::Dropped, None);
						},
					}
				},
			}
		}

		(
			DesktopFrameResult::Ready {
				source_image,
				src_layout,
				input_format,
			},
			Some(import_vk_format),
		)
	}
}
