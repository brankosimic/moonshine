use crate::session::compositor::frame::{FrameColorSpace, HdrMetadata};
use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
#[derive(Debug, Clone)]
pub(crate) struct ExportedFrame {
	pub planes: Vec<ExportedPlane>,
	pub format: u32,
	pub modifier: u64,
	pub width: u32,
	pub height: u32,
	pub created_at: Instant,
	#[allow(dead_code)]
	pub buffer_index: usize,
	pub consumed: Arc<AtomicBool>,
	pub color_space: FrameColorSpace,
	pub hdr_metadata: Option<HdrMetadata>,
}
impl ExportedFrame {
	pub(crate) fn mark_consumed(&self) {
		self.consumed.store(true, Ordering::Release);
	}
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct MappedPtr(pub *const u8);
unsafe impl Send for MappedPtr {}
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExportedPlane {
	pub fd: RawFd,
	pub offset: u32,
	pub stride: u32,
	pub mapped_ptr: Option<MappedPtr>,
	pub mapped_size: usize,
}
