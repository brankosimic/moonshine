//! Frame types for the desktop (portal) capture path.
//!
//! Unlike ferriscope's [`ferriscope_core::Frame`], a desktop frame may be a
//! DMA-BUF *or* CPU-visible pixels, and it is not tied to a pool slot: the
//! portal owns the buffer lifetime, so there is no release guard. The video
//! pipeline consumes these in desktop mode instead of compositor frames.

use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::session::compositor::frame::{FrameColorSpace, HdrMetadata};

/// A frame produced by the desktop (portal) capture path.
#[derive(Debug, Clone)]
pub(crate) struct ExportedFrame {
	/// Per-plane DMA-BUF metadata.
	pub planes: Vec<ExportedPlane>,
	/// DRM format (e.g. Argb8888, Abgr2101010).
	pub format: u32,
	/// DRM modifier (e.g. Linear, tiled).
	pub modifier: u64,
	/// Frame width in pixels.
	pub width: u32,
	/// Frame height in pixels.
	pub height: u32,
	/// Timestamp when the frame was produced by the capture.
	pub created_at: Instant,
	/// Index of the pre-allocated buffer in the capture's pool.
	#[allow(dead_code)]
	pub buffer_index: usize,
	/// Shared flag set to `true` by the encoder after color conversion
	/// completes, signalling the capture that this buffer may be reused.
	pub consumed: Arc<AtomicBool>,
	/// Color space of the rendered frame.
	pub color_space: FrameColorSpace,
	/// Optional HDR metadata from the captured content.
	pub hdr_metadata: Option<HdrMetadata>,
}

impl ExportedFrame {
	/// Mark the frame as fully consumed so its source buffer may be recycled.
	pub(crate) fn mark_consumed(&self) {
		self.consumed.store(true, Ordering::Release);
	}
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MappedPtr(pub *const u8);
unsafe impl Send for MappedPtr {}

/// Metadata for a single DMA-BUF plane.
///
/// The fd is a borrowed reference (raw fd number) into the capture's buffer
/// pool. It is NOT owned — do not close it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExportedPlane {
	/// Raw file descriptor — borrowed from the capture's buffer pool.
	pub fd: RawFd,
	/// Byte offset into the DMA-BUF for this plane.
	pub offset: u32,
	/// Row stride in bytes.
	pub stride: u32,
	/// For software (memfd/memptr) buffers: a valid mmap of the data.
	/// `None` for DMA-BUF buffers.
	pub mapped_ptr: Option<MappedPtr>,
	/// Size in bytes of the mapped region (only meaningful when
	/// `mapped_ptr` is `Some`).
	pub mapped_size: usize,
}
