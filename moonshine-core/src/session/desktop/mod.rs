use std::cell::RefCell;
use std::os::unix::io::{OwnedFd, RawFd};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ashpd::desktop::remote_desktop::{
	Axis, DeviceType, KeyState, RemoteDesktop,
};
use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};
use ashpd::desktop::{PersistMode, Session};
use async_shutdown::ShutdownManager;
use pipewire as pw;
use pw::spa;
use pw::spa::pod::serialize::PodSerializer;
use pw::spa::pod::PropertyFlags;
use pw::spa::utils::Id;

use crate::session::SessionContext;
use crate::session::compositor::frame::{ExportedFrame, ExportedPlane, FrameColorSpace};
use crate::session::compositor::input::CompositorInputEvent;
use crate::session::manager::SessionShutdownReason;

const FRAME_CHANNEL_CAPACITY: usize = 2;
const FORMAT_NEGOTIATION_TIMEOUT: Duration = Duration::from_secs(3);
const DESKTOP_LAUNCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Build a `SPA_PARAM_Buffers` pod declaring the acceptable buffer data
/// types (a bitmask of `spa_data_type`, e.g. `1 << SPA_DATA_DmaBuf`).
///
/// Responding to the negotiated Format param with a Buffers param via
/// `pw_stream_update_params` is required by the PipeWire negotiation
/// protocol: without it the node picks a default buffer type, which on
/// some screencast pipelines is a plain memfd (`SPA_DATA_MemFd`) that is
/// not importable into Vulkan as a DMA-BUF.
fn build_buffers_param(data_type_mask: i32) -> Vec<u8> {
	use std::io::Cursor;
	use pw::spa::pod::{Object, Property, Value};

	let buffers = Object {
		type_: pw::spa::sys::SPA_TYPE_OBJECT_ParamBuffers,
		id: pw::spa::sys::SPA_PARAM_Buffers,
		properties: vec![Property {
			key: pw::spa::sys::SPA_PARAM_BUFFERS_dataType,
			flags: PropertyFlags::empty(),
			value: Value::Int(data_type_mask),
		}],
	};
	let mut data = Vec::new();
	let (_cursor, _len) = PodSerializer::serialize(Cursor::new(&mut data), &Value::Object(buffers))
		.expect("serialize buffers object");
	data
}

pub(crate) struct DesktopHandles {
	pub frame_rx: Receiver<ExportedFrame>,
	pub input_tx: calloop::channel::Sender<CompositorInputEvent>,
}

pub(crate) struct DesktopReady {
	pub resolution: (u32, u32),
	#[allow(dead_code)]
	pub stream_id: u32,
	pub hdr: bool,
}

pub(crate) struct LaunchedDesktop {
	pub ready: DesktopReady,
}

pub(crate) struct Desktop {
	width: u32,
	height: u32,
	refresh_rate: u32,
	stop: ShutdownManager<SessionShutdownReason>,
	frame_tx: SyncSender<ExportedFrame>,
	input_rx: calloop::channel::Channel<CompositorInputEvent>,
	ready_tx: SyncSender<Result<DesktopReady, String>>,
	ready_rx: Receiver<Result<DesktopReady, String>>,
}

impl Desktop {
	pub(crate) fn new(
		context: &SessionContext,
		stop: ShutdownManager<SessionShutdownReason>,
	) -> (Self, DesktopHandles) {
		let (frame_tx, frame_rx) = sync_channel(FRAME_CHANNEL_CAPACITY);
		let (input_tx, input_rx) = calloop::channel::channel();
		let (ready_tx, ready_rx) = sync_channel(1);

		(
			Self {
				width: context.resolution.0,
				height: context.resolution.1,
				refresh_rate: context.refresh_rate.max(1),
				stop,
				frame_tx,
				input_rx,
				ready_tx,
				ready_rx,
			},
			DesktopHandles { frame_rx, input_tx },
		)
	}

	pub(crate) async fn launch(self) -> Result<LaunchedDesktop, ()> {
		let Self {
			width,
			height,
			refresh_rate,
			stop,
			frame_tx,
			input_rx,
			ready_tx,
			ready_rx,
		} = self;

		let remote_desktop = RemoteDesktop::new().await.map_err(|e| {
			tracing::error!("Failed to connect to the RemoteDesktop portal: {e}");
		})?;
		let screencast = Screencast::new().await.map_err(|e| {
			tracing::error!("Failed to connect to the ScreenCast portal: {e}");
		})?;

		let session = Arc::new(
			remote_desktop
				.create_session()
				.await
				.map_err(|e| tracing::error!("Failed to create portal session: {e}"))?,
		);

		remote_desktop
			.select_devices(
				&session,
				DeviceType::Keyboard | DeviceType::Pointer,
				None,
				PersistMode::DoNot,
			)
			.await
			.map_err(|e| tracing::error!("Failed to select input devices: {e}"))?;

		screencast
			.select_sources(
				&session,
				CursorMode::Embedded,
				SourceType::Monitor.into(),
				true,
				None,
				PersistMode::DoNot,
			)
			.await
			.map_err(|e| tracing::error!("Failed to select capture sources: {e}"))?;

		let selected = remote_desktop
			.start(&session, None)
			.await
			.map_err(|e| tracing::error!("Failed to start remote desktop session: {e}"))?
			.response()
			.map_err(|e| tracing::error!("Remote desktop session start rejected: {e}"))?;

		let devices = selected.devices();
		tracing::info!(?devices, "Remote desktop devices granted.");
		if !devices.contains(DeviceType::Keyboard) {
			tracing::warn!("Portal did not grant keyboard input.");
		}
		if !devices.contains(DeviceType::Pointer) {
			tracing::warn!("Portal did not grant pointer input.");
		}

		let stream = selected
			.streams()
			.ok_or_else(|| tracing::error!("Remote desktop session returned no capture streams."))?;
		let stream_id = stream[0].pipe_wire_node_id();
		let resolution = stream[0]
			.size()
			.map(|(w, h)| (w.max(1) as u32, h.max(1) as u32))
			.unwrap_or((width.max(1), height.max(1)));
		tracing::info!(stream_id, ?resolution, "Desktop capture stream ready.");

		let fd = screencast
			.open_pipe_wire_remote(&session)
			.await
			.map_err(|e| tracing::error!("Failed to open PipeWire remote: {e}"))?;

		let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
		{
			let stop = stop.clone();
			tokio::spawn(async move {
				stop.wait_shutdown_triggered().await;
				let _ = stop_tx.send(());
			});
		}

		let input_remote = remote_desktop;
		let input_session = session.clone();
		let input_stop = stop.clone();
		tokio::spawn(async move {
			run_input(input_remote, input_session, input_rx, input_stop, resolution, stream_id).await;
		});

		let capture_stop = stop.clone();
		std::thread::Builder::new()
			.name("desktop-capture".to_string())
			.spawn(move || {
				run_capture(
					fd,
					stream_id,
					resolution,
					refresh_rate,
					frame_tx,
					stop_rx,
					capture_stop,
					ready_tx,
				);
			})
			.map_err(|e| tracing::error!("Failed to spawn desktop capture thread: {e}"))?;

		let ready = ready_rx
			.recv_timeout(DESKTOP_LAUNCH_TIMEOUT)
			.map_err(|e| tracing::error!("Timed out waiting for desktop capture to start: {e}"))?
			.map_err(|e| tracing::error!("Desktop capture failed: {e}"))?;

		tracing::info!(?ready.resolution, "Desktop capture live.");
		Ok(LaunchedDesktop { ready })
	}
}

struct Shared {
	frame_tx: SyncSender<ExportedFrame>,
	/// Pending buffers: (pw_buffer pointer, consumed flag, optional (mmap ptr, size)).
	/// The mmap is for MemFd buffers and must be unmapped when the buffer is recycled.
	pending: Vec<(
		*mut pw::sys::pw_buffer,
		Arc<AtomicBool>,
		Option<(*mut libc::c_void, usize)>,
	)>,
	next_index: u64,
	info: Option<spa::param::video::VideoInfoRaw>,
}

#[allow(clippy::too_many_arguments)]
fn run_capture(
	fd: OwnedFd,
	stream_id: u32,
	resolution: (u32, u32),
	refresh_rate: u32,
	frame_tx: SyncSender<ExportedFrame>,
	stop_rx: std::sync::mpsc::Receiver<()>,
	stop: ShutdownManager<SessionShutdownReason>,
	ready_tx: SyncSender<Result<DesktopReady, String>>,
) {
	let result = run_capture_inner(
		fd,
		stream_id,
		resolution,
		refresh_rate,
		frame_tx,
		stop_rx,
		ready_tx.clone(),
	);
	if let Err(e) = result {
		tracing::error!("Desktop capture stopped: {e}");
		let _ = ready_tx.send(Err(e.clone()));
		if stop.trigger_shutdown(SessionShutdownReason::DesktopStreamStopped).is_ok() {
			tracing::warn!("Triggered session shutdown after desktop capture failure.");
		}
	}
}

fn run_capture_inner(
	fd: OwnedFd,
	stream_id: u32,
	resolution: (u32, u32),
	refresh_rate: u32,
	frame_tx: SyncSender<ExportedFrame>,
	stop_rx: std::sync::mpsc::Receiver<()>,
	ready_tx: SyncSender<Result<DesktopReady, String>>,
) -> Result<(), String> {
	pw::init();

	let mainloop = pw::main_loop::MainLoopBox::new(None).map_err(|e| format!("main loop: {e}"))?;
	let context =
		pw::context::ContextBox::new(mainloop.loop_(), None).map_err(|e| format!("context: {e}"))?;
	let core = context
		.connect_fd(fd, None)
		.map_err(|e| format!("connect to PipeWire remote: {e}"))?;

	let stream = pw::stream::StreamBox::new(
		&core,
		"moonshine-desktop",
		pw::properties::properties! {
			*pw::keys::MEDIA_TYPE => "Video",
			*pw::keys::MEDIA_CATEGORY => "Capture",
			*pw::keys::MEDIA_ROLE => "Screen",
		},
	)
	.map_err(|e| format!("stream: {e}"))?;

	let shared = Rc::new(RefCell::new(Shared {
		frame_tx: frame_tx.clone(),
		pending: Vec::new(),
		next_index: 0,
		info: None,
	}));

	let _listener = stream
		.add_local_listener_with_user_data(shared.clone())
		.state_changed(|_, _, old, new| {
			tracing::debug!(?old, ?new, "Desktop capture stream state changed");
		})
		.param_changed(|stream, shared, id, param| {
			let Some(param) = param else {
				tracing::debug!(param_id = id, "Desktop capture param cleared");
				return;
			};
			let (media_type, media_subtype) = spa::param::format_utils::parse_format(param)
				.map(|(mt, ms)| (mt.as_raw(), ms.as_raw()))
				.unwrap_or((u32::MAX, u32::MAX));
			tracing::debug!(
				param_id = id,
				media_type,
				media_subtype,
				"Desktop capture param changed"
			);
			if id != spa::param::ParamType::Format.as_raw() {
				return;
			}
			if media_type != spa::param::format::MediaType::Video.as_raw()
				|| media_subtype != spa::param::format::MediaSubtype::Raw.as_raw()
			{
				tracing::warn!(
					media_type,
					media_subtype,
					"Desktop capture format has unsupported media type/subtype"
				);
				return;
			}
			let mut info = spa::param::video::VideoInfoRaw::new();
			if info.parse(param).is_err() {
				return;
			}
			tracing::info!(
				format = ?info.format(),
				size = ?info.size(),
				modifier = info.modifier(),
				"Desktop capture format negotiated"
			);

			// Complete the PipeWire buffer negotiation: ack the buffer data
			// type we can consume. Request DMA-BUF buffers when the format
			// advertises a modifier, otherwise fall back to plain memory.
			// Without this ack the node defaults to memfd buffers, which
			// cannot be imported into Vulkan as DMA-BUF.
			let buffer_types = if param
				.as_object()
				.ok()
				.and_then(|obj| obj.find_prop(Id(pw::spa::sys::SPA_FORMAT_VIDEO_modifier)))
				.is_some()
			{
				tracing::info!("Requesting DMA-BUF capture buffers");
				1 << pw::spa::sys::SPA_DATA_DmaBuf
			} else {
				tracing::info!("Requesting memory capture buffers");
				1 << pw::spa::sys::SPA_DATA_MemPtr
			};
			let pod_bytes = build_buffers_param(buffer_types);
			let Some(buffers_pod) = pw::spa::pod::Pod::from_bytes(&pod_bytes) else {
				tracing::warn!("Failed to parse serialized buffers param");
				return;
			};
			let mut params = [buffers_pod];
			if let Err(e) = stream.update_params(&mut params) {
				tracing::warn!("Failed to update capture buffer params: {e}");
			} else {
				tracing::info!(buffer_types, "Capture buffer type negotiated");
			}

			shared.borrow_mut().info = Some(info);
		})
		.process(|stream, shared| {
			let mut shared = shared.borrow_mut();

			shared.pending.retain_mut(|entry| {
				let (buf, consumed, mmap_info) = entry;
				if consumed.load(Ordering::Acquire) {
					// Unmap MemFd buffers before recycling.
					if let Some((ptr, size)) = mmap_info.take() {
						unsafe { libc::munmap(ptr, size); }
					}
					unsafe { stream.queue_raw_buffer(*buf) };
					false
				} else {
					true
				}
			});

			let Some(info) = shared.info.as_ref() else { return; };
			let Some((fourcc, width, height, modifier)) = captured_format(info)
				.map(|(fourcc, width, height)| (fourcc, width, height, info.modifier()))
			else {
				return;
			};

			loop {
				let buf = unsafe { stream.dequeue_raw_buffer() };
				if buf.is_null() {
					break;
				}

				let Some((fd, offset, stride, mmap_ptr, mmap_size)) =
					(unsafe { buffer_plane(buf) })
				else {
					unsafe { stream.queue_raw_buffer(buf) };
					continue;
				};

				let consumed = Arc::new(AtomicBool::new(false));
				let frame = ExportedFrame {
					planes: vec![ExportedPlane {
						fd,
						offset,
						stride,
						mapped_ptr: mmap_ptr.map(|p| crate::session::compositor::frame::MappedPtr(p as *const u8)),
						mapped_size: mmap_size,
					}],
					format: fourcc,
					modifier,
					width,
					height,
					created_at: Instant::now(),
					buffer_index: shared.next_index as usize,
					consumed: consumed.clone(),
					color_space: FrameColorSpace::Srgb,
					hdr_metadata: None,
				};
				shared.next_index = shared.next_index.wrapping_add(1);

				match shared.frame_tx.try_send(frame) {
					Ok(()) => shared.pending.push((buf, consumed, mmap_ptr.map(|p| (p, mmap_size)))),
					Err(_) => unsafe { stream.queue_raw_buffer(buf) },
				}
			}
		})
		.register()
		.map_err(|e| format!("stream listener: {e}"))?;

	let (w, h) = resolution;
	let fps = refresh_rate.min(1000);
	let obj = pw::spa::pod::object!(
		pw::spa::utils::SpaTypes::ObjectParamFormat,
		pw::spa::param::ParamType::EnumFormat,
		pw::spa::pod::property!(
			pw::spa::param::format::FormatProperties::MediaType,
			Id,
			pw::spa::param::format::MediaType::Video
		),
		pw::spa::pod::property!(
			pw::spa::param::format::FormatProperties::MediaSubtype,
			Id,
			pw::spa::param::format::MediaSubtype::Raw
		),
		pw::spa::pod::property!(
			pw::spa::param::format::FormatProperties::VideoFormat,
			Choice,
			Enum,
			Id,
			pw::spa::param::video::VideoFormat::BGRx,
			pw::spa::param::video::VideoFormat::BGRx,
			pw::spa::param::video::VideoFormat::RGBx,
			pw::spa::param::video::VideoFormat::BGRA,
			pw::spa::param::video::VideoFormat::RGBA,
		),
		pw::spa::pod::property!(
			pw::spa::param::format::FormatProperties::VideoSize,
			Choice,
			Range,
			Rectangle,
			pw::spa::utils::Rectangle { width: w, height: h },
			pw::spa::utils::Rectangle { width: 1, height: 1 },
			pw::spa::utils::Rectangle { width: 16384, height: 16384 }
		),
		pw::spa::pod::property!(
			pw::spa::param::format::FormatProperties::VideoFramerate,
			Choice,
			Range,
			Fraction,
			pw::spa::utils::Fraction { num: fps, denom: 1 },
			pw::spa::utils::Fraction { num: 0, denom: 1 },
			pw::spa::utils::Fraction { num: 1000, denom: 1 }
		),
	);
	let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
		std::io::Cursor::new(Vec::new()),
		&pw::spa::pod::Value::Object(obj),
	)
	.map_err(|e| format!("format pod: {e}"))?
	.0
	.into_inner();
	let pod = pw::spa::pod::Pod::from_bytes(&values).ok_or("format pod: invalid bytes")?;
	let mut params = [pod];

	stream
		.connect(
			spa::utils::Direction::Input,
			Some(stream_id),
			pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
			&mut params,
		)
		.map_err(|e| format!("connect stream: {e}"))?;
	stream
		.set_active(true)
		.map_err(|e| format!("activate stream: {e}"))?;

	let deadline = Instant::now() + FORMAT_NEGOTIATION_TIMEOUT;
	loop {
		if shared.borrow().info.is_some() {
			break;
		}
		if stop_rx.try_recv().is_ok() {
			return Err("stopped before capture format negotiation".to_string());
		}
		if Instant::now() > deadline {
			return Err("timed out waiting for capture format negotiation".to_string());
		}
		mainloop.loop_().iterate(pw::loop_::Timeout::Finite(Duration::from_millis(50)));
	}

	let _ = ready_tx.send(Ok(DesktopReady { resolution, stream_id, hdr: false }));

	loop {
		if stop_rx.try_recv().is_ok() {
			break;
		}
		mainloop.loop_().iterate(pw::loop_::Timeout::Finite(Duration::from_millis(100)));
	}

	for (buf, _, mmap_info) in shared.borrow_mut().pending.drain(..) {
		if let Some((ptr, size)) = mmap_info {
			unsafe { libc::munmap(ptr, size); }
		}
		unsafe { stream.queue_raw_buffer(buf) };
	}

	Ok(())
}

fn captured_format(info: &spa::param::video::VideoInfoRaw) -> Option<(u32, u32, u32)> {
	use spa::param::video::VideoFormat;

	let fourcc = match info.format() {
		VideoFormat::BGRA => 0x34325241,
		VideoFormat::BGRx => 0x34325258,
		VideoFormat::RGBA => 0x34324241,
		VideoFormat::RGBx => 0x34324258,
		VideoFormat::ABGR_210LE => 0x30334241,
		other => {
			tracing::warn!(?other, "Unsupported desktop capture format, dropping frames.");
			return None;
		},
	};
	let width = info.size().width;
	let height = info.size().height;
	Some((fourcc, width, height))
}

unsafe fn buffer_plane(buf: *mut pw::sys::pw_buffer) -> Option<(RawFd, u32, u32, Option<*mut libc::c_void>, usize)> {
	let spa_buf = unsafe { (*buf).buffer };
	if spa_buf.is_null()
		|| unsafe { (*spa_buf).n_datas == 0 || (*spa_buf).datas.is_null() }
	{
		return None;
	}
	let data = unsafe { &*((*spa_buf).datas) };
	tracing::debug!(
		data_type = data.type_,
		data_fd = data.fd,
		data_flags = data.flags,
		"Desktop capture buffer data"
	);
	let fd = data.fd as RawFd;
	if fd < 0 {
		return None;
	}

	let is_memfd = data.type_ == pw::spa::sys::SPA_DATA_MemFd;
	let chunk = unsafe { &*data.chunk };

	if is_memfd {
		// MemFd buffer — mmap it so the pipeline can CPU-copy into a VkImage.
		let size = chunk.size as usize;
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
				chunk.offset as libc::off_t,
			)
		};
		if ptr == libc::MAP_FAILED {
			tracing::warn!("Failed to mmap MemFd buffer fd={fd} size={size}");
			return None;
		}
		Some((fd, chunk.offset, chunk.stride as u32, Some(ptr), size))
	} else {
		Some((fd, chunk.offset, chunk.stride as u32, None, 0))
	}
}

async fn run_input(
	remote_desktop: RemoteDesktop<'_>,
	session: Arc<Session<'_, RemoteDesktop<'_>>>,
	input_rx: calloop::channel::Channel<CompositorInputEvent>,
	stop: ShutdownManager<SessionShutdownReason>,
	capture_size: (u32, u32),
	stream_id: u32,
) {
	let (bridge_tx, mut bridge_rx) = tokio::sync::mpsc::unbounded_channel::<CompositorInputEvent>();
	let bridge = tokio::task::spawn_blocking(move || {
		while let Ok(event) = input_rx.recv() {
			if bridge_tx.send(event).is_err() {
				break;
			}
		}
	});

	let forward = async {
		while let Some(event) = bridge_rx.recv().await {
			forward_input(&remote_desktop, &session, &event, capture_size, stream_id).await;
		}
	};

	tokio::select! {
		_ = stop.wait_shutdown_triggered() => {
			tracing::info!("Desktop input task stopping (session shutdown).");
		},
		_ = forward => {
			tracing::info!("Desktop input channel closed.");
		},
	}

	bridge.abort();

	if let Err(e) = session.close().await {
		tracing::debug!("Failed to close portal session: {e}");
	}
	tracing::info!("Desktop portal session closed.");
}

async fn forward_input(
	remote_desktop: &RemoteDesktop<'_>,
	session: &Session<'_, RemoteDesktop<'_>>,
	event: &CompositorInputEvent,
	capture_size: (u32, u32),
	stream_id: u32,
) {
	let result = match event {
		CompositorInputEvent::KeyDown { keycode } => remote_desktop
			.notify_keyboard_keycode(session, *keycode as i32, KeyState::Pressed)
			.await,
		CompositorInputEvent::KeyUp { keycode } => remote_desktop
			.notify_keyboard_keycode(session, *keycode as i32, KeyState::Released)
			.await,
		CompositorInputEvent::MouseMoveAbsolute {
			x,
			y,
			screen_width,
			screen_height,
		} => {
			let (w, h) = capture_size;
			let nx = if *screen_width > 0 {
				*x as f64 / *screen_width as f64 * w as f64
			} else {
				*x as f64
			};
			let ny = if *screen_height > 0 {
				*y as f64 / *screen_height as f64 * h as f64
			} else {
				*y as f64
			};
			remote_desktop
				.notify_pointer_motion_absolute(session, stream_id, nx, ny)
				.await
		},
		CompositorInputEvent::MouseMoveRelative { dx, dy } => remote_desktop
			.notify_pointer_motion(session, *dx as f64, *dy as f64)
			.await,
		CompositorInputEvent::MouseButtonDown { button } => remote_desktop
			.notify_pointer_button(session, *button as i32, KeyState::Pressed)
			.await,
		CompositorInputEvent::MouseButtonUp { button } => remote_desktop
			.notify_pointer_button(session, *button as i32, KeyState::Released)
			.await,
		CompositorInputEvent::ScrollVertical { amount } => remote_desktop
			.notify_pointer_axis_discrete(session, Axis::Vertical, *amount as i32)
			.await,
		CompositorInputEvent::ScrollHorizontal { amount } => remote_desktop
			.notify_pointer_axis_discrete(session, Axis::Horizontal, *amount as i32)
			.await,
	};

	if let Err(e) = result {
		tracing::debug!(target: "input", "Failed to forward input event to portal: {e}");
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::desktop_streaming::{detect_desktop_session, desktop_application};

	#[test]
	fn detects_kde_wayland_session() {
		unsafe { std::env::remove_var("XDG_CURRENT_DESKTOP") };
		unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
		assert!(!detect_desktop_session());

		unsafe { std::env::set_var("XDG_CURRENT_DESKTOP", "KDE") };
		assert!(!detect_desktop_session());

		unsafe { std::env::set_var("WAYLAND_DISPLAY", "wayland-0") };
		assert!(detect_desktop_session());

		unsafe { std::env::set_var("XDG_CURRENT_DESKTOP", "kde-plasma") };
		assert!(detect_desktop_session());

		unsafe { std::env::set_var("XDG_CURRENT_DESKTOP", "GNOME") };
		assert!(!detect_desktop_session());
	}

	#[test]
	fn maps_negotiated_formats_to_drm_fourcc() {
		fn info_for(format: spa::param::video::VideoFormat) -> spa::param::video::VideoInfoRaw {
			let mut info = spa::param::video::VideoInfoRaw::new();
			info.set_format(format);
			info
		}

		let cases = [
			(spa::param::video::VideoFormat::BGRA, 0x34325241),
			(spa::param::video::VideoFormat::BGRx, 0x34325258),
			(spa::param::video::VideoFormat::RGBA, 0x34324241),
			(spa::param::video::VideoFormat::RGBx, 0x34324258),
			(spa::param::video::VideoFormat::ABGR_210LE, 0x30334241),
		];
		for (format, fourcc) in cases {
			let (got_fourcc, _, _) = captured_format(&info_for(format)).expect("format must map");
			assert_eq!(got_fourcc, fourcc);
		}

		assert!(
			captured_format(&info_for(spa::param::video::VideoFormat::A420)).is_none(),
		);
	}

	#[test]
	fn desktop_application_entry_is_marked_desktop() {
		let app = desktop_application();
		assert_eq!(app.title, "Desktop");
		assert!(app.desktop);
		assert!(app.command.is_empty());
	}
}
