use std::os::fd::RawFd;
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
