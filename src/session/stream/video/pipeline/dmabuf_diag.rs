use std::os::fd::RawFd;

use ash::vk;
use pixelforge::VideoContext;

pub(crate) struct ImportDiag {
	pub fd: RawFd,
	pub width: u32,
	pub height: u32,
	pub format: vk::Format,
	pub modifier: u64,
	pub image_bits: u32,
	pub fd_bits: u32,
	pub size: u64,
}

pub(crate) fn diagnose_dmabuf_import_failure(context: &VideoContext, d: &ImportDiag) {
	static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
	if DONE.swap(true, std::sync::atomic::Ordering::SeqCst) || std::env::var("MOONSHINE_DMABUF_DIAG").is_err() {
		return;
	}

	let instance = context.instance();
	let phys = context.physical_device();
	let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };

	tracing::warn!(
		fd = d.fd, width = d.width, height = d.height, ?d.format, modifier = d.modifier,
		size = d.size, image_bits = d.image_bits, fd_bits = d.fd_bits,
		"DMA-BUF import failure diagnostic — memory type table:"
	);
	for (i, mt) in mem_props.memory_types.iter().enumerate() {
		let wanted_by_image = d.image_bits & (1 << i) != 0;
		let offered_by_fd = d.fd_bits & (1 << i) != 0;
		tracing::warn!(
			type_index = i,
			flags = ?mt.property_flags,
			heap_index = mt.heap_index,
			wanted_by_image,
			offered_by_fd,
			"  memory type"
		);
	}
	for (i, heap) in mem_props.memory_heaps.iter().enumerate() {
		tracing::warn!(heap_index = i, size = heap.size, "  memory heap");
	}
}
