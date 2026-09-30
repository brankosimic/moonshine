use std::cell::RefCell;
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::io::{BorrowedFd, OwnedFd, RawFd};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::{Duration, Instant};

use ashpd::desktop::remote_desktop::{Axis, DeviceType, KeyState, RemoteDesktop};
use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};
use ashpd::desktop::{PersistMode, Session};
use async_shutdown::ShutdownManager;
use pipewire as pw;
use pw::spa;
use pw::spa::pod::PropertyFlags;
use pw::spa::pod::serialize::PodSerializer;

use crate::session::SessionContext;
use crate::session::compositor::frame::{ExportedFrame, ExportedPlane, FrameColorSpace};
use crate::session::compositor::input::CompositorInputEvent;
use crate::session::manager::SessionShutdownReason;

const FRAME_CHANNEL_CAPACITY: usize = 2;
const FORMAT_NEGOTIATION_TIMEOUT: Duration = Duration::from_secs(3);
const DESKTOP_LAUNCH_TIMEOUT: Duration = Duration::from_secs(10);
const CAPTURE_LOOP_TIMEOUT: Duration = Duration::from_millis(500);

fn build_buffers_param(data_type_mask: i32, prefer_linear: bool) -> Vec<u8> {
	use pw::spa::pod::{Object, Property, Value};
	use std::io::Cursor;

	let mut properties = vec![Property {
		key: pw::spa::sys::SPA_PARAM_BUFFERS_dataType,
		flags: PropertyFlags::empty(),
		value: Value::Int(data_type_mask),
	}];
	if prefer_linear {
		properties.push(Property {
			key: pw::spa::sys::SPA_FORMAT_VIDEO_modifier,
			flags: PropertyFlags::empty(),
			value: Value::Long(0),
		});
	}

	let buffers = Object {
		type_: pw::spa::sys::SPA_TYPE_OBJECT_ParamBuffers,
		id: pw::spa::sys::SPA_PARAM_Buffers,
		properties,
	};
	let mut data = Vec::new();
	let (_cursor, _len) =
		PodSerializer::serialize(Cursor::new(&mut data), &Value::Object(buffers)).expect("serialize buffers object");
	data
}

/// Serialize the EnumFormat POD advertising raw video capture capabilities:
/// BGRx/RGBx/BGRA/RGBA at `resolution` and up to `refresh_rate` fps.
///
/// No `modifier` property is advertised: on producers that cannot export
/// DMA-BUF (kwin with the NVIDIA GBM backend) a DONT_FIXATE modifier choice
/// makes the fixate step loop and no buffer is ever delivered. Producers
/// then fixate to linear with a shared-memory (MemFd) copy, which the
/// CPU-upload path handles.
fn build_enum_format_pod(resolution: (u32, u32), refresh_rate: u32) -> Result<Vec<u8>, String> {
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
			pw::spa::utils::Rectangle {
				width: 16384,
				height: 16384
			}
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
	Ok(
		PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &pw::spa::pod::Value::Object(obj))
			.map_err(|e| format!("{e}"))?
			.0
			.into_inner(),
	)
}

pub(crate) struct DesktopHandles {
	pub frame_rx: Receiver<ExportedFrame>,
	pub input_tx: calloop::channel::Sender<CompositorInputEvent>,
}

pub(crate) struct DesktopReady {
	pub resolution: (u32, u32),
	#[allow(dead_code)]
	pub stream_id: u32,
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

		let stream_ref = selected
			.streams()
			.and_then(|streams| streams.first())
			.ok_or_else(|| tracing::error!("Remote desktop session returned no capture streams."))?;
		let stream_id = stream_ref.pipe_wire_node_id();
		let resolution = stream_ref
			.size()
			.map(|(w, h)| (w.max(1) as u32, h.max(1) as u32))
			.unwrap_or((width.max(1), height.max(1)));
		tracing::info!(stream_id, ?resolution, "Desktop capture stream ready.");

		let fd = screencast
			.open_pipe_wire_remote(&session)
			.await
			.map_err(|e| tracing::error!("Failed to open PipeWire remote: {e}"))?;

		let wake_fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
		if wake_fd < 0 {
			return Err(());
		}
		let wake_owned: OwnedFd = unsafe { std::os::unix::io::FromRawFd::from_raw_fd(wake_fd) };

		let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
		{
			let stop = stop.clone();
			let wake = wake_owned.try_clone().map_err(|e| {
				tracing::error!("Failed to clone capture wakeup fd: {e}");
			})?;
			tokio::spawn(async move {
				stop.wait_shutdown_triggered().await;
				let _ = stop_tx.send(());
				write_eventfd(wake.as_raw_fd());
			});
		}

		let input_remote = remote_desktop;
		let input_session = session.clone();
		let input_stop = stop.clone();
		tokio::spawn(async move {
			run_input(
				input_remote,
				input_session,
				input_rx,
				input_stop,
				resolution,
				stream_id,
				(width.max(1), height.max(1)),
			)
			.await;
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
					wake_owned,
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

/// A frame handed off to the encoder but not yet consumed. Holds the raw
/// PipeWire buffer (returned on consume), the consumed flag shared with the
/// encode thread, and any `munmap` bookkeeping for memfd mappings.
struct PendingFrame {
	buf: *mut pw::sys::pw_buffer,
	consumed: Arc<AtomicBool>,
	unmap: Option<(*mut libc::c_void, usize)>,
}

struct Shared {
	frame_tx: SyncSender<ExportedFrame>,
	pending: Vec<PendingFrame>,
	next_index: u64,
	info: Option<spa::param::video::VideoInfoRaw>,
	last_handoff: Option<Instant>,
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
	wake_fd: OwnedFd,
) {
	let result = run_capture_inner(
		fd,
		stream_id,
		resolution,
		refresh_rate,
		frame_tx,
		stop_rx,
		ready_tx.clone(),
		wake_fd,
	);
	if let Err(e) = result {
		tracing::error!("Desktop capture stopped: {e}");
		let _ = ready_tx.send(Err(e.clone()));
		if stop
			.trigger_shutdown(SessionShutdownReason::DesktopStreamStopped)
			.is_ok()
		{
			tracing::warn!("Triggered session shutdown after desktop capture failure.");
		}
	}
}

#[allow(clippy::too_many_arguments)]
fn run_capture_inner(
	fd: OwnedFd,
	stream_id: u32,
	resolution: (u32, u32),
	refresh_rate: u32,
	frame_tx: SyncSender<ExportedFrame>,
	stop_rx: std::sync::mpsc::Receiver<()>,
	ready_tx: SyncSender<Result<DesktopReady, String>>,
	wake_fd: OwnedFd,
) -> Result<(), String> {
	pw::init();

	let mainloop = pw::main_loop::MainLoopBox::new(None).map_err(|e| format!("main loop: {e}"))?;
	let context = pw::context::ContextBox::new(mainloop.loop_(), None).map_err(|e| format!("context: {e}"))?;
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

	let wake_raw = wake_fd.as_raw_fd();

	let shared = Rc::new(RefCell::new(Shared {
		frame_tx: frame_tx.clone(),
		pending: Vec::new(),
		next_index: 0,
		info: None,
		last_handoff: None,
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

			let buffer_types = (1 << pw::spa::sys::SPA_DATA_DmaBuf)
				| (1 << pw::spa::sys::SPA_DATA_MemFd)
				| (1 << pw::spa::sys::SPA_DATA_MemPtr);
			tracing::info!(buffer_types, "Advertising capture buffer types (DMA-BUF preferred)");
			let prefer_linear = true;
			let pod_bytes = build_buffers_param(buffer_types, prefer_linear);
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
		.process(move |stream, shared| {
			let mut shared = shared.borrow_mut();

			shared.pending.retain_mut(|frame| {
				if frame.consumed.load(Ordering::Acquire) {
					if let Some((ptr, size)) = frame.unmap.take() {
						unsafe {
							libc::munmap(ptr, size);
						}
					}
					unsafe { stream.queue_raw_buffer(frame.buf) };
					false
				} else {
					true
				}
			});

			let Some(info) = shared.info.as_ref() else {
				return;
			};
			let Some((fourcc, width, height, modifier)) =
				captured_format(info).map(|(fourcc, width, height)| (fourcc, width, height, info.modifier()))
			else {
				return;
			};

			let min_interval = Duration::from_secs_f64(0.95 / (refresh_rate.max(1) as f64));

			loop {
				let buf = unsafe { stream.dequeue_raw_buffer() };
				if buf.is_null() {
					break;
				}

				if shared.last_handoff.is_some_and(|last| last.elapsed() < min_interval) {
					unsafe { stream.queue_raw_buffer(buf) };
					continue;
				}
				let Some(plane) = (unsafe { buffer_plane(buf) }) else {
					unsafe { stream.queue_raw_buffer(buf) };
					continue;
				};
				let CapturePlane {
					fd,
					offset,
					stride,
					mapped_ptr,
					mapped_size,
					unmap,
				} = plane;

				let consumed = Arc::new(AtomicBool::new(false));
				let frame = ExportedFrame {
					planes: vec![ExportedPlane {
						fd,
						offset,
						stride,
						mapped_ptr: mapped_ptr.map(|p| crate::session::compositor::frame::MappedPtr(p as *const u8)),
						mapped_size,
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
					Ok(()) => {
						shared.last_handoff = Some(Instant::now());
						shared.pending.push(PendingFrame { buf, consumed, unmap });
						write_eventfd(wake_raw);
					},
					Err(_) => {
						if let Some((ptr, size)) = unmap {
							unsafe { libc::munmap(ptr, size) };
						}
						unsafe { stream.queue_raw_buffer(buf) };
					},
				}
			}
		})
		.register()
		.map_err(|e| format!("stream listener: {e}"))?;

	// The EnumFormat POD advertising the capture capabilities: raw RGB(A)
	// formats at the negotiated size and refresh rate.
	let values = build_enum_format_pod(resolution, refresh_rate).map_err(|e| format!("format pod: {e}"))?;
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
	stream.set_active(true).map_err(|e| format!("activate stream: {e}"))?;

	let _wake_source = mainloop.loop_().add_io(
		wake_fd.as_fd(),
		spa::support::system::IoFlags::IN,
		move |io: &mut BorrowedFd<'_>| {
			drain_eventfd(io.as_raw_fd());
		},
	);

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
		mainloop
			.loop_()
			.iterate(pw::loop_::Timeout::Finite(Duration::from_millis(50)));
	}

	let _ = ready_tx.send(Ok(DesktopReady { resolution, stream_id }));

	loop {
		if stop_rx.try_recv().is_ok() {
			break;
		}
		mainloop
			.loop_()
			.iterate(pw::loop_::Timeout::Finite(CAPTURE_LOOP_TIMEOUT));

		shared.borrow_mut().pending.retain_mut(|frame| {
			if frame.consumed.load(Ordering::Acquire) {
				if let Some((ptr, size)) = frame.unmap.take() {
					unsafe {
						libc::munmap(ptr, size);
					}
				}
				unsafe { stream.queue_raw_buffer(frame.buf) };
				false
			} else {
				true
			}
		});
	}

	for frame in shared.borrow_mut().pending.drain(..) {
		if let Some((ptr, size)) = frame.unmap {
			unsafe {
				libc::munmap(ptr, size);
			}
		}
	}

	Ok(())
}

fn write_eventfd(fd: RawFd) {
	let val: u64 = 1;
	unsafe {
		let _ = libc::write(
			fd,
			&val as *const u64 as *const libc::c_void,
			std::mem::size_of::<u64>(),
		);
	}
}

fn drain_eventfd(fd: RawFd) {
	let mut buf = [0u8; 8];
	unsafe {
		let _ = libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len());
	}
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

/// A single captured buffer plane, either DMA-BUF (zero-copy import) or
/// CPU-visible memory (memfd-mapped or inline memptr).
struct CapturePlane {
	fd: RawFd,
	offset: u32,
	stride: u32,
	mapped_ptr: Option<*mut libc::c_void>,
	mapped_size: usize,
	/// munmap bookkeeping; `None` for DMA-BUF and memptr (nothing to unmap).
	unmap: Option<(*mut libc::c_void, usize)>,
}

unsafe fn buffer_plane(buf: *mut pw::sys::pw_buffer) -> Option<CapturePlane> {
	if buf.is_null() {
		return None;
	}
	let spa_buf = unsafe { (*buf).buffer };
	if spa_buf.is_null() || unsafe { (*spa_buf).n_datas == 0 || (*spa_buf).datas.is_null() } {
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

	let is_memfd = data.type_ == pw::spa::sys::SPA_DATA_MemFd;
	let is_memptr = data.type_ == pw::spa::sys::SPA_DATA_MemPtr;
	let chunk = unsafe { &*data.chunk };

	if is_memptr {
		if data.data.is_null() {
			return None;
		}
		let offset = chunk.offset as usize;
		let size = (chunk.size as usize).saturating_sub(offset);
		if size == 0 {
			return None;
		}
		let ptr = unsafe { (data.data as *const u8).add(offset) as *mut libc::c_void };
		return Some(CapturePlane {
			fd: -1,
			offset: 0,
			stride: chunk.stride as u32,
			mapped_ptr: Some(ptr),
			mapped_size: size,
			unmap: None,
		});
	}

	if fd < 0 {
		return None;
	}

	if is_memfd {
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
		Some(CapturePlane {
			fd,
			offset: chunk.offset,
			stride: chunk.stride as u32,
			mapped_ptr: Some(ptr),
			mapped_size: size,
			unmap: Some((ptr, size)),
		})
	} else {
		Some(CapturePlane {
			fd,
			offset: chunk.offset,
			stride: chunk.stride as u32,
			mapped_ptr: None,
			mapped_size: 0,
			unmap: None,
		})
	}
}

/// Map a point from stream coordinates (what the client sees) to capture
/// coordinates (the portal's screen), accounting for the letterbox the video
/// pipeline applies when the capture aspect ratio differs from the stream's.
///
/// `(x, y)` is expressed in pixels over `(screen_w, screen_h)` (the client's
/// reported screen size); normalized coordinates pass `1.0, 1.0`. The
/// letterbox math mirrors `gpu_scaler::letterbox_push`.
fn map_stream_to_capture(
	x: f64,
	y: f64,
	screen_w: f64,
	screen_h: f64,
	capture_size: (u32, u32),
	stream_size: (u32, u32),
) -> (f64, f64) {
	// Normalize to [0, 1] within the stream.
	let mut u = if screen_w > 0.0 { x / screen_w } else { x };
	let mut v = if screen_h > 0.0 { y / screen_h } else { y };

	let (cap_w, cap_h) = (capture_size.0.max(1) as f64, capture_size.1.max(1) as f64);
	let (str_w, str_h) = (stream_size.0.max(1) as f64, stream_size.1.max(1) as f64);

	// Undo the letterbox: stream space maps onto the fitted rect, centered,
	// with black bars outside it.
	let scale = (str_w / cap_w).min(str_h / cap_h);
	let fit_w = (cap_w * scale).round().max(1.0);
	let fit_h = (cap_h * scale).round().max(1.0);
	let off_x = (str_w - fit_w) / 2.0;
	let off_y = (str_h - fit_h) / 2.0;

	u = (u * str_w - off_x) / fit_w;
	v = (v * str_h - off_y) / fit_h;

	// Clamp to the capture; clicks in the black bars land on the edge.
	let nx = u.clamp(0.0, 1.0) * cap_w;
	let ny = v.clamp(0.0, 1.0) * cap_h;
	(nx, ny)
}

async fn run_input(
	remote_desktop: RemoteDesktop<'_>,
	session: Arc<Session<'_, RemoteDesktop<'_>>>,
	input_rx: calloop::channel::Channel<CompositorInputEvent>,
	stop: ShutdownManager<SessionShutdownReason>,
	capture_size: (u32, u32),
	stream_id: u32,
	stream_size: (u32, u32),
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
			forward_input(&remote_desktop, &session, &event, capture_size, stream_id, stream_size).await;
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
	stream_size: (u32, u32),
) {
	let result = match event {
		CompositorInputEvent::KeyDown { keycode } => {
			remote_desktop
				.notify_keyboard_keycode(session, *keycode as i32, KeyState::Pressed)
				.await
		},
		CompositorInputEvent::KeyUp { keycode } => {
			remote_desktop
				.notify_keyboard_keycode(session, *keycode as i32, KeyState::Released)
				.await
		},
		CompositorInputEvent::MouseMoveAbsolute {
			x,
			y,
			screen_width,
			screen_height,
		} => {
			let (nx, ny) = map_stream_to_capture(
				*x as f64,
				*y as f64,
				*screen_width as f64,
				*screen_height as f64,
				capture_size,
				stream_size,
			);
			remote_desktop
				.notify_pointer_motion_absolute(session, stream_id, nx, ny)
				.await
		},
		CompositorInputEvent::MouseMoveRelative { dx, dy } => {
			remote_desktop
				.notify_pointer_motion(session, *dx as f64, *dy as f64)
				.await
		},
		CompositorInputEvent::MouseButtonDown { button } => {
			remote_desktop
				.notify_pointer_button(session, *button as i32, KeyState::Pressed)
				.await
		},
		CompositorInputEvent::MouseButtonUp { button } => {
			remote_desktop
				.notify_pointer_button(session, *button as i32, KeyState::Released)
				.await
		},
		CompositorInputEvent::ScrollVertical { amount } => {
			remote_desktop
				.notify_pointer_axis_discrete(session, Axis::Vertical, *amount as i32)
				.await
		},
		CompositorInputEvent::ScrollHorizontal { amount } => {
			remote_desktop
				.notify_pointer_axis_discrete(session, Axis::Horizontal, *amount as i32)
				.await
		},
		CompositorInputEvent::TouchDown { slot, x, y } => {
			let (nx, ny) = map_stream_to_capture(*x as f64, *y as f64, 1.0, 1.0, capture_size, stream_size);
			remote_desktop
				.notify_touch_down(session, stream_id, *slot, nx, ny)
				.await
		},
		CompositorInputEvent::TouchMove { slot, x, y } => {
			let (nx, ny) = map_stream_to_capture(*x as f64, *y as f64, 1.0, 1.0, capture_size, stream_size);
			remote_desktop
				.notify_touch_motion(session, stream_id, *slot, nx, ny)
				.await
		},
		CompositorInputEvent::TouchUp { slot } => remote_desktop.notify_touch_up(session, *slot).await,
		CompositorInputEvent::TouchCancelAll
		| CompositorInputEvent::Pen { .. }
		| CompositorInputEvent::TypeText { .. } => {
			tracing::debug!(target: "input", "Ignoring unsupported portal input event: {event:?}");
			Ok(())
		},
	};

	if let Err(e) = result {
		tracing::debug!(target: "input", "Failed to forward input event to portal: {e}");
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::desktop_streaming::desktop_application;
	use pw::spa::utils::Id;

	#[test]
	fn desktop_application_entry_is_marked_desktop() {
		let app = desktop_application();
		assert_eq!(app.title, "Desktop");
		assert!(app.desktop);
		assert!(app.command.is_empty());
		assert!(app.boxart.is_some(), "Desktop entry must ship a boxart");
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

		assert!(captured_format(&info_for(spa::param::video::VideoFormat::A420)).is_none(),);
	}

	#[test]
	fn letterbox_input_identity_when_sizes_match() {
		// 1920x1080 capture streamed at 1920x1080: no letterbox, direct mapping.
		let (nx, ny) = map_stream_to_capture(960.0, 540.0, 1920.0, 1080.0, (1920, 1080), (1920, 1080));
		assert!((nx - 960.0).abs() < 1e-6, "nx = {nx}");
		assert!((ny - 540.0).abs() < 1e-6, "ny = {ny}");
	}

	#[test]
	fn letterbox_input_maps_center_and_corners() {
		// 4:3 capture (1024x768) letterboxed into a 16:9 stream (1280x720):
		// fit height, bars left/right. The fitted rect is 960x720 centered.
		let (cx, cy) = map_stream_to_capture(640.0, 360.0, 1280.0, 720.0, (1024, 768), (1280, 720));
		assert!((cx - 512.0).abs() < 1e-6, "center x = {cx}");
		assert!((cy - 384.0).abs() < 1e-6, "center y = {cy}");

		// Left edge of the fitted rect maps to capture x=0.
		let (x0, _) = map_stream_to_capture(160.0, 360.0, 1280.0, 720.0, (1024, 768), (1280, 720));
		assert!((x0).abs() < 1e-6, "left edge = {x0}");

		// A click in the left black bar clamps to the capture edge.
		let (bar, _) = map_stream_to_capture(80.0, 360.0, 1280.0, 720.0, (1024, 768), (1280, 720));
		assert!(bar.abs() < 1e-6, "bar click = {bar}");
	}

	#[test]
	fn letterbox_input_normalized_touch_coordinates() {
		// Touch events arrive normalized (0..1): center stays center.
		let (nx, ny) = map_stream_to_capture(0.5, 0.5, 1.0, 1.0, (1024, 768), (1280, 720));
		assert!((nx - 512.0).abs() < 1e-6, "nx = {nx}");
		assert!((ny - 384.0).abs() < 1e-6, "ny = {ny}");
	}

	#[test]
	fn enum_format_pod_serializes_and_roundtrips() {
		let values = build_enum_format_pod((1920, 1080), 60).expect("pod must serialize");
		let pod = pw::spa::pod::Pod::from_bytes(&values).expect("serialized pod must parse");
		let (media_type, media_subtype) =
			spa::param::format_utils::parse_format(pod).expect("pod must parse as format");
		assert_eq!(media_type.as_raw(), spa::param::format::MediaType::Video.as_raw());
		assert_eq!(media_subtype.as_raw(), spa::param::format::MediaSubtype::Raw.as_raw());

		// The stream connect path parses it back into a VideoInfoRaw; ensure
		// the advertised size survives a full negotiation round-trip.
		let mut info = spa::param::video::VideoInfoRaw::new();
		info.parse(pod).expect("VideoInfoRaw must parse the EnumFormat pod");
	}

	#[test]
	fn enum_format_pod_clamps_excessive_refresh_rate() {
		// 10_000 fps must clamp to the advertised maximum (1000), not overflow
		// the Fraction property.
		let values = build_enum_format_pod((640, 480), 10_000).expect("pod must serialize");
		let pod = pw::spa::pod::Pod::from_bytes(&values).expect("pod must parse");
		let mut info = spa::param::video::VideoInfoRaw::new();
		info.parse(pod).expect("clamped pod must still parse");
	}

	#[test]
	fn buffers_param_contains_datatype_and_optional_linear_modifier() {
		let dma_mask = 1 << pw::spa::sys::SPA_DATA_DmaBuf;
		let mem_mask = 1 << pw::spa::sys::SPA_DATA_MemPtr;

		// Memory buffers: only the dataType property, no modifier request.
		let mem_bytes = build_buffers_param(mem_mask, false);
		let mem_pod = pw::spa::pod::Pod::from_bytes(&mem_bytes).expect("mem buffers pod must parse");
		assert!(mem_pod.as_object().is_ok());

		// DMA-BUF with linear preference: the modifier=0 property rides along.
		let dma_bytes = build_buffers_param(dma_mask, true);
		let dma_pod = pw::spa::pod::Pod::from_bytes(&dma_bytes).expect("dma buffers pod must parse");
		let obj = dma_pod.as_object().expect("buffers param is an object");
		assert!(
			obj.find_prop(Id(pw::spa::sys::SPA_FORMAT_VIDEO_modifier)).is_some(),
			"linear preference must request modifier 0"
		);
	}

	#[test]
	fn eventfd_wakeup_roundtrip() {
		let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
		assert!(fd >= 0);

		// Non-blocking, nothing written: read drains EAGAIN harmlessly.
		drain_eventfd(fd);

		write_eventfd(fd);
		write_eventfd(fd);
		// eventfd accumulates: a single drain reads the accumulated count.
		drain_eventfd(fd);
		// Fully drained: a second drain is a no-op.
		drain_eventfd(fd);

		unsafe { libc::close(fd) };
	}

	/// Build a `pw_buffer` around fully-owned `spa_buffer`/`spa_data`/`spa_chunk`
	/// values for `buffer_plane` tests. The closure receives the raw buffer ptr.
	fn with_pw_buffer<R>(n_datas: u32, f: impl FnOnce(*mut pw::sys::pw_buffer) -> R) -> R {
		let mut buffer = pw::sys::pw_buffer {
			buffer: std::ptr::null_mut(),
			user_data: std::ptr::null_mut(),
			size: 0,
			requested: 0,
			time: 0,
		};
		let mut spa_buffer = pw::spa::sys::spa_buffer {
			n_metas: 0,
			n_datas,
			metas: std::ptr::null_mut(),
			datas: std::ptr::null_mut(),
		};
		let mut datas: Vec<pw::spa::sys::spa_data> = (0..n_datas.max(1) as usize)
			.map(|_| pw::spa::sys::spa_data {
				type_: 0,
				flags: 0,
				fd: -1,
				mapoffset: 0,
				maxsize: 0,
				data: std::ptr::null_mut(),
				chunk: std::ptr::null_mut(),
			})
			.collect();
		let mut chunks: Vec<pw::spa::sys::spa_chunk> = (0..n_datas.max(1) as usize)
			.map(|_| pw::spa::sys::spa_chunk {
				offset: 0,
				size: 0,
				stride: 0,
				flags: 0,
			})
			.collect();
		for (data, chunk) in datas.iter_mut().zip(chunks.iter_mut()) {
			data.chunk = chunk;
		}
		spa_buffer.datas = datas.as_mut_ptr();
		buffer.buffer = &mut spa_buffer;

		f(std::ptr::from_mut(&mut buffer))
	}

	#[test]
	fn buffer_plane_rejects_null_and_empty_buffers() {
		// Null buffer.
		assert!(unsafe { buffer_plane(std::ptr::null_mut()) }.is_none());

		// Buffer with zero data members.
		let plane = with_pw_buffer(0, |buf| unsafe { buffer_plane(buf) });
		assert!(plane.is_none(), "no datas must yield no plane");
	}

	#[test]
	fn buffer_plane_extracts_dmabuf_plane() {
		// A DMA-BUF plane: type DmaBuf, valid fd, no mapped pointer.
		let plane = with_pw_buffer(1, |buf| unsafe {
			let spa_buffer = (*buf).buffer;
			let data = &mut *(*spa_buffer).datas;
			data.type_ = pw::spa::sys::SPA_DATA_DmaBuf;
			data.fd = 42;
			(*data.chunk).offset = 256;
			(*data.chunk).stride = 7680;
			buffer_plane(buf)
		})
		.expect("dmabuf plane must extract");
		assert_eq!(plane.fd, 42);
		assert_eq!(plane.offset, 256);
		assert_eq!(plane.stride, 7680);
		assert!(plane.mapped_ptr.is_none(), "dmabuf is imported, not mapped");
		assert!(plane.unmap.is_none(), "dmabuf needs no munmap");
	}

	#[test]
	fn buffer_plane_rejects_dmabuf_without_fd() {
		let plane = with_pw_buffer(1, |buf| unsafe {
			let spa_buffer = (*buf).buffer;
			let data = &mut *(*spa_buffer).datas;
			data.type_ = pw::spa::sys::SPA_DATA_DmaBuf;
			data.fd = -1;
			buffer_plane(buf)
		});
		assert!(plane.is_none(), "dmabuf without fd must be rejected");
	}

	#[test]
	fn buffer_plane_extracts_memptr_plane() {
		// Inline memory: type MemPtr with a data pointer and a sized chunk.
		let sample = [1u8, 2, 3, 4];
		let plane = with_pw_buffer(1, |buf| unsafe {
			let spa_buffer = (*buf).buffer;
			let data = &mut *(*spa_buffer).datas;
			data.type_ = pw::spa::sys::SPA_DATA_MemPtr;
			data.data = sample.as_ptr() as *mut std::os::raw::c_void;
			(*data.chunk).offset = 0;
			(*data.chunk).size = 4;
			(*data.chunk).stride = 4;
			buffer_plane(buf)
		})
		.expect("memptr plane must extract");
		assert_eq!(plane.fd, -1, "memptr carries no fd");
		assert!(plane.mapped_ptr.is_some(), "memptr must expose its pointer");
		assert_eq!(plane.mapped_size, 4);
		assert!(plane.unmap.is_none(), "memptr needs no munmap");
	}

	#[test]
	fn buffer_plane_rejects_zero_sized_memptr() {
		let plane = with_pw_buffer(1, |buf| unsafe {
			let spa_buffer = (*buf).buffer;
			let data = &mut *(*spa_buffer).datas;
			data.type_ = pw::spa::sys::SPA_DATA_MemPtr;
			data.data = std::ptr::dangling_mut::<std::os::raw::c_void>(); // non-null
			(*data.chunk).size = 0; // but empty
			buffer_plane(buf)
		});
		assert!(plane.is_none(), "empty memptr must be rejected");
	}
}
