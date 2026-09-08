//! Fallback helpers for DMA-BUF frames the driver cannot import as external memory.
//!
//! Some drivers (e.g. NVIDIA at certain large resolutions) report no usable memory
//! type for a portal buffer's DMA-BUF fd, so zero-copy `vkImportMemoryFd` fails.
//! The caller uploads the mapped plane through the host-visible `CpuUploader`. A
//! GPU-side `vkCmdCopyBufferToImage` upload was tried here but faults inside
//! `libnvidia-eglcore` on `vkQueueSubmit` for this device, so it is not used.

use std::os::fd::RawFd;

/// Map a DMA-BUF plane fd read-only so the caller can feed it to the CPU uploader.
/// Returns `(ptr, size)` on success.
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
		tracing::warn!("DMA-BUF fallback: mmap failed fd={fd} size={size}");
		return None;
	}
	Some((ptr as *const u8, size))
}
