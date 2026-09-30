//! Desktop audio capture: records the system audio playing on the user's real
//! sound server and feeds it to the stream encoder.
//!
//! Regular sessions point the launched application at Moonshine's private
//! PulseAudio-compatible server (see `pulse_server`), so the app's audio
//! arrives through that socket. A desktop stream has no such application —
//! the desktop plays audio through the user's system daemon (PulseAudio or
//! PipeWire's pulse compat layer). This module connects to that daemon as a
//! client and records the **monitor source of the default sink**, which
//! carries the post-mix audio of everything playing on the host.

use std::os::unix::net::UnixStream;
use std::time::Instant;

use async_shutdown::ShutdownManager;
use pulseaudio::protocol::{self as pulse};
use pulseaudio::{Client, RecordBuffer};

use crate::session::manager::SessionShutdownReason;

use super::pulse_server::{AudioFrame, CAPTURE_SAMPLE_RATE, capture_channel_map};

/// Target bytes buffered ahead by the record stream: ~20ms of f32 stereo @48kHz.
const TARGET_BUFFERED_BYTES: usize = 48000 * 2 * 4 / 50;

pub(crate) struct DesktopAudioCapture {}

impl DesktopAudioCapture {
	/// Connect to the user's sound server and record the default sink's
	/// monitor at the negotiated sample spec, feeding frames to `frame_tx`.
	///
	/// Runs its own thread with a single-threaded tokio runtime for the
	/// pulseaudio client's async reactor.
	///
	/// Desktop audio is best-effort: if the sound server is unreachable or
	/// has no monitor source, the session continues in silence rather than
	/// being torn down (video keeps streaming).
	pub(crate) fn spawn(
		channels: u8,
		packet_duration_ms: u32,
		frame_tx: crossbeam_channel::Sender<AudioFrame>,
		frame_recycle_rx: crossbeam_channel::Receiver<AudioFrame>,
		stop: ShutdownManager<SessionShutdownReason>,
	) -> Result<(), ()> {
		let socket_path = user_pulse_socket_path().ok_or_else(|| {
			tracing::warn!("No user PulseAudio/PipeWire socket found; desktop audio disabled.");
		})?;
		tracing::info!(path = %socket_path.display(), "Connecting to system sound server for desktop audio.");

		std::thread::Builder::new()
			.name("desktop-audio-capture".to_string())
			.spawn(move || {
				// Keep the shutdown waiting for us while we run; capture
				// failure must NOT end the session (video keeps streaming).
				let _delay_stop = stop.delay_shutdown_token();

				let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
					Ok(rt) => rt,
					Err(e) => {
						tracing::warn!("Failed to build runtime for desktop audio capture: {e}");
						return;
					},
				};
				if rt
					.block_on(capture_loop(
						socket_path,
						channels,
						packet_duration_ms,
						frame_tx,
						frame_recycle_rx,
						stop,
					))
					.is_err()
				{
					tracing::warn!("Desktop audio capture stopped; continuing without audio.");
				}
			})
			.map_err(|e| tracing::error!("Failed to spawn desktop audio capture thread: {e}"))?;

		Ok(())
	}
}

/// The socket of the *user's* sound server. Deliberately ignores
/// `PULSE_SERVER`, which Moonshine sets for launched applications to point at
/// its private server; the desktop capture must reach the real daemon.
fn user_pulse_socket_path() -> Option<std::path::PathBuf> {
	let runtime_dir = env_or("XDG_RUNTIME_DIR").or_else(|| env_or("PULSE_RUNTIME_PATH"))?;
	let path = std::path::PathBuf::from(runtime_dir).join("pulse/native");
	std::fs::metadata(&path).is_ok().then_some(path)
}

fn env_or(key: &str) -> Option<String> {
	std::env::var(key).ok().filter(|v| !v.is_empty())
}

async fn capture_loop(
	socket_path: std::path::PathBuf,
	channels: u8,
	packet_duration_ms: u32,
	frame_tx: crossbeam_channel::Sender<AudioFrame>,
	frame_recycle_rx: crossbeam_channel::Receiver<AudioFrame>,
	stop: ShutdownManager<SessionShutdownReason>,
) -> Result<(), ()> {
	let socket = UnixStream::connect(&socket_path)
		.map_err(|e| tracing::warn!("Failed to connect to sound server at {}: {e}", socket_path.display()))?;
	let cookie = std::env::var("PULSE_COOKIE")
		.ok()
		.map(std::path::PathBuf::from)
		.or_else(|| {
			#[allow(deprecated)]
			std::env::home_dir().map(|home| home.join(".config/pulse/cookie"))
		})
		.filter(|path| path.is_file())
		.and_then(|path| std::fs::read(path).ok());
	let client = Client::new_unix(c"moonshine-desktop-capture", socket, cookie)
		.map_err(|e| tracing::warn!("PulseAudio handshake with system server failed: {e}"))?;

	// The monitor of the default sink carries all desktop audio.
	let server_info = client
		.server_info()
		.await
		.map_err(|e| tracing::warn!("Failed to query system sound server: {e}"))?;
	let default_sink = server_info
		.default_sink_name
		.ok_or_else(|| tracing::warn!("System sound server reported no default sink."))?;
	let sink_info = client
		.sink_info_by_name(default_sink.clone())
		.await
		.map_err(|e| tracing::warn!("Failed to query default sink {default_sink:?}: {e}"))?;
	let monitor = sink_info
		.monitor_source_name
		.ok_or_else(|| tracing::warn!("Default sink {default_sink:?} has no monitor source."))?;
	tracing::info!(?monitor, "Recording desktop audio from default sink monitor.");

	// One AudioFrame per packet: the server paces us at exactly the encoder's
	// cadence via fragsize, so no local timer is needed.
	let frame_ms = packet_duration_ms.clamp(1, 100);
	let samples_per_frame = CAPTURE_SAMPLE_RATE * frame_ms / 1000;
	let bytes_per_frame = (samples_per_frame * channels as u32 * 4) as usize;

	let buffer = RecordBuffer::new(TARGET_BUFFERED_BYTES.max(bytes_per_frame * 4));
	let capture_spec = pulse::SampleSpec {
		format: pulse::SampleFormat::Float32Le,
		channels,
		sample_rate: CAPTURE_SAMPLE_RATE,
	};
	let params = pulse::RecordStreamParams {
		sample_spec: capture_spec,
		channel_map: capture_channel_map(channels),
		source_name: Some(monitor),
		buffer_attr: pulse::stream::BufferAttr {
			max_length: (bytes_per_frame * 4) as u32,
			fragment_size: bytes_per_frame as u32,
			..Default::default()
		},
		flags: pulse::stream::StreamFlags {
			adjust_latency: true,
			..Default::default()
		},
		..Default::default()
	};

	let _stream = client
		.create_record_stream(params, buffer.as_record_sink())
		.await
		.map_err(|e| tracing::warn!("Failed to create desktop audio record stream: {e}"))?;

	pump_frames(buffer, bytes_per_frame, frame_tx, frame_recycle_rx, stop).await
}

/// Read captured bytes off the record buffer and emit encoder frames at the
/// cadence the sound server delivers them.
async fn pump_frames(
	mut buffer: RecordBuffer,
	bytes_per_frame: usize,
	frame_tx: crossbeam_channel::Sender<AudioFrame>,
	frame_recycle_rx: crossbeam_channel::Receiver<AudioFrame>,
	stop: ShutdownManager<SessionShutdownReason>,
) -> Result<(), ()> {
	use futures_util::io::AsyncReadExt;

	let epoch = Instant::now();
	let mut bytes_scratch = vec![0u8; bytes_per_frame];

	loop {
		if stop.is_shutdown_triggered() {
			tracing::info!("Desktop audio capture stopping (session shutdown).");
			return Ok(());
		}

		// Read exactly one frame's worth; the server delivers at wall-clock
		// pace, so this naturally blocks until the next chunk is ready.
		match stop.wrap_cancel(buffer.read_exact(&mut bytes_scratch)).await {
			// Cancelled by shutdown: stop the pump.
			Err(_) => {
				tracing::info!("Desktop audio capture stopping (session shutdown).");
				return Ok(());
			},
			// The read itself failed (server gone / EOF): stop the pump. An
			// EOF-after-data would otherwise spin-read Ok(0) forever.
			Ok(Err(e)) => {
				tracing::warn!("Desktop audio stream ended: {e}");
				return Err(());
			},
			Ok(Ok(())) => (),
		}

		let mut frame = match frame_recycle_rx.try_recv() {
			Ok(mut frame) => {
				frame.buf.clear();
				frame
			},
			Err(crossbeam_channel::TryRecvError::Empty) => AudioFrame {
				buf: Vec::with_capacity(bytes_scratch.len() / 4),
				capture_ts_ms: 0,
			},
			Err(crossbeam_channel::TryRecvError::Disconnected) => {
				tracing::debug!("Encoder gone; desktop audio capture stopping.");
				return Ok(());
			},
		};

		frame.buf.reserve(bytes_scratch.len() / 4);
		let (samples, _rem) = bytes_scratch.as_chunks::<4>();
		for sample in samples {
			frame.buf.push(f32::from_le_bytes(*sample));
		}
		frame.capture_ts_ms = epoch.elapsed().as_millis() as u64;

		if let Err(crossbeam_channel::SendError(_)) = frame_tx.send(frame) {
			tracing::debug!("Encoder gone; desktop audio capture stopping.");
			return Ok(());
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Serialize env-mutating tests across the whole crate (see `test_support`).
	#[allow(clippy::significant_drop_tightening)]
	fn lock_env() -> std::sync::MutexGuard<'static, ()> {
		crate::test_support::env_lock()
	}

	fn restore(var: &str, previous: Option<std::ffi::OsString>) {
		match previous {
			Some(value) => unsafe { std::env::set_var(var, value) },
			None => unsafe { std::env::remove_var(var) },
		}
	}

	#[test]
	fn socket_path_resolves_from_runtime_dir() {
		let _lock = lock_env();
		let prev_runtime = std::env::var_os("XDG_RUNTIME_DIR");
		let prev_pulse = std::env::var_os("PULSE_RUNTIME_PATH");

		let scratch = std::env::temp_dir().join("moonshine-test-pulse-socket");
		std::fs::create_dir_all(scratch.join("pulse")).unwrap();
		std::fs::write(scratch.join("pulse/native"), "").unwrap();
		unsafe { std::env::set_var("XDG_RUNTIME_DIR", &scratch) };
		let path = user_pulse_socket_path().expect("socket must resolve");
		assert_eq!(path, scratch.join("pulse/native"));

		restore("XDG_RUNTIME_DIR", prev_runtime);
		restore("PULSE_RUNTIME_PATH", prev_pulse);
	}

	#[test]
	fn socket_path_requires_existing_socket() {
		let _lock = lock_env();
		let prev_runtime = std::env::var_os("XDG_RUNTIME_DIR");
		let prev_pulse = std::env::var_os("PULSE_RUNTIME_PATH");

		// Runtime dir exists but has no pulse socket.
		let scratch = std::env::temp_dir().join("moonshine-test-no-pulse-socket");
		std::fs::create_dir_all(&scratch).unwrap();
		unsafe { std::env::set_var("XDG_RUNTIME_DIR", &scratch) };
		assert!(user_pulse_socket_path().is_none());

		restore("XDG_RUNTIME_DIR", prev_runtime);
		restore("PULSE_RUNTIME_PATH", prev_pulse);
	}

	#[test]
	fn socket_path_falls_back_to_pulse_runtime_path() {
		let _lock = lock_env();
		let prev_runtime = std::env::var_os("XDG_RUNTIME_DIR");
		let prev_pulse = std::env::var_os("PULSE_RUNTIME_PATH");

		let scratch = std::env::temp_dir().join("moonshine-test-pulse-runtime");
		std::fs::create_dir_all(scratch.join("pulse")).unwrap();
		std::fs::write(scratch.join("pulse/native"), "").unwrap();
		unsafe { std::env::remove_var("XDG_RUNTIME_DIR") };
		unsafe { std::env::set_var("PULSE_RUNTIME_PATH", &scratch) };
		let path = user_pulse_socket_path().expect("socket must resolve via PULSE_RUNTIME_PATH");
		assert_eq!(path, scratch.join("pulse/native"));

		restore("XDG_RUNTIME_DIR", prev_runtime);
		restore("PULSE_RUNTIME_PATH", prev_pulse);
	}

	#[test]
	fn env_or_rejects_empty_values() {
		assert_eq!(env_or("MOONSHINE_DEFINITELY_UNSET_VAR"), None);
	}
}

#[cfg(test)]
mod fake_server_tests {
	//! Deterministic tests for `capture_loop`'s setup failure branches,
	//! driven by an in-process fake sound server that speaks the PulseAudio
	//! protocol via the `pulseaudio` crate's own codec.

	use super::*;
	use std::io::{BufReader, Write as _};
	use std::os::unix::net::{UnixListener, UnixStream};

	/// What the fake server should answer for the default-sink query.
	#[derive(Clone, Copy, PartialEq, Eq)]
	enum SinkScenario {
		/// `GetServerInfo` reports no default sink.
		NoDefaultSink,
		/// Default sink exists but has no monitor source.
		NoMonitor,
		/// Full happy path: monitor exists, record stream accepted, one frame
		/// of audio data pushed, then the connection closes.
		StreamOneFrame,
	}

	/// Handle one pulse protocol client until it disconnects or the scenario
	/// plays out. Uses blocking std IO on its own thread.
	fn serve(mut stream: UnixStream, scenario: SinkScenario) {
		let mut reader = BufReader::new(match stream.try_clone() {
			Ok(r) => r,
			Err(_) => return,
		});
		let mut protocol_version = pulse::MAX_VERSION;

		loop {
			let Ok((seq, command)) = pulse::read_command_message(&mut reader, protocol_version) else {
				return;
			};
			match command {
				pulse::Command::Auth(auth) => {
					protocol_version = auth.version.min(pulse::MAX_VERSION);
					let reply = pulse::AuthReply {
						version: pulse::MAX_VERSION,
						..Default::default()
					};
					let _ = pulse::write_reply_message(&mut stream, seq, &reply, protocol_version);
				},
				pulse::Command::SetClientName(_) => {
					let reply = pulse::SetClientNameReply { client_id: 42 };
					let _ = pulse::write_reply_message(&mut stream, seq, &reply, protocol_version);
				},
				pulse::Command::GetServerInfo => {
					let reply = pulse::ServerInfo {
						default_sink_name: match scenario {
							SinkScenario::NoDefaultSink => None,
							SinkScenario::NoMonitor | SinkScenario::StreamOneFrame => {
								Some(std::ffi::CString::new("fake-sink").unwrap())
							},
						},
						..Default::default()
					};
					let _ = pulse::write_reply_message(&mut stream, seq, &reply, protocol_version);
				},
				pulse::Command::GetSinkInfo(_) => {
					let reply = pulse::SinkInfo {
						name: std::ffi::CString::new("fake-sink").unwrap(),
						monitor_source_name: match scenario {
							SinkScenario::StreamOneFrame => Some(std::ffi::CString::new("fake-monitor").unwrap()),
							_ => None,
						},
						..Default::default()
					};
					let _ = pulse::write_reply_message(&mut stream, seq, &reply, protocol_version);
				},
				pulse::Command::CreateRecordStream(params) if scenario == SinkScenario::StreamOneFrame => {
					let reply = pulse::CreateRecordStreamReply {
						channel: 7,
						stream_index: 7,
						buffer_attr: params.buffer_attr,
						sample_spec: params.sample_spec,
						channel_map: params.channel_map,
						stream_latency: 0,
						sink_index: 0,
						sink_name: None,
						suspended: false,
						format: pulse::FormatInfo::new(pulse::FormatEncoding::Pcm),
					};
					let _ = pulse::write_reply_message(&mut stream, seq, &reply, protocol_version);

					// Push exactly one frame of stream data on channel 7, then
					// close: `pump_frames` must deliver the frame and then see
					// EOF once the buffer drains.
					let payload: Vec<u8> = (0..480).flat_map(|_| 0.25f32.to_le_bytes()).collect();
					let desc = pulse::Descriptor {
						length: payload.len() as u32,
						channel: 7,
						offset: 0,
						flags: pulse::DescriptorFlags::empty(),
					};
					let mut framed = Vec::new();
					pulse::write_descriptor(&mut framed, &desc).unwrap();
					framed.extend_from_slice(&payload);
					let _ = stream.write_all(&framed);
					let _ = stream.flush();
					// Keep the socket open briefly so the client can read the
					// frame before EOF.
					std::thread::sleep(std::time::Duration::from_millis(200));
					return;
				},
				// Anything else: reject, ending the scenario.
				_ => {
					let _ = pulse::write_error(&mut stream, seq, &pulse::PulseError::NoEntity);
					return;
				},
			}
		}
	}

	/// Bind a fake server on a unique temp path, serving `scenario`, returning
	/// the socket path. Panics on failure (test-only).
	fn spawn_fake_server(scenario: SinkScenario) -> std::path::PathBuf {
		use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
		static NEXT_ID: AtomicU32 = AtomicU32::new(0);

		let id = NEXT_ID.fetch_add(1, AtomicOrdering::SeqCst);
		let dir = std::env::temp_dir().join(format!("moonshine-fake-pulse-{id}-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("native");
		let _ = std::fs::remove_file(&path);
		let listener = UnixListener::bind(&path).expect("bind fake pulse socket");

		std::thread::spawn(move || {
			if let Ok((stream, _)) = listener.accept() {
				serve(stream, scenario);
			}
		});
		path
	}

	fn test_channels() -> u8 {
		2
	}

	/// Drive `capture_loop` against a fake server; returns its Result along
	/// with the receiving end of the frame channel.
	async fn run_capture_loop(
		socket_path: std::path::PathBuf,
	) -> (Result<(), ()>, crossbeam_channel::Receiver<AudioFrame>) {
		let stop = ShutdownManager::<SessionShutdownReason>::new();
		let (frame_tx, frame_rx) = crossbeam_channel::bounded(3);
		let (_recycle_tx, frame_recycle_rx) = crossbeam_channel::bounded(3);
		let result = capture_loop(socket_path, test_channels(), 5, frame_tx, frame_recycle_rx, stop).await;
		(result, frame_rx)
	}

	#[tokio::test]
	async fn capture_fails_when_no_default_sink() {
		let path = spawn_fake_server(SinkScenario::NoDefaultSink);
		let (result, _frames) = run_capture_loop(path.clone()).await;
		assert!(result.is_err(), "no default sink must fail capture setup");
		let _ = std::fs::remove_file(&path);
	}

	#[tokio::test]
	async fn capture_fails_when_sink_has_no_monitor() {
		let path = spawn_fake_server(SinkScenario::NoMonitor);
		let (result, _frames) = run_capture_loop(path.clone()).await;
		assert!(result.is_err(), "sink without monitor must fail capture setup");
		let _ = std::fs::remove_file(&path);
	}

	#[tokio::test]
	async fn capture_fails_on_unreachable_socket() {
		let dir = std::env::temp_dir().join(format!("moonshine-fake-pulse-unreachable-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("definitely-not-listening");
		let _ = std::fs::remove_file(&path);
		let (result, _frames) = run_capture_loop(path.clone()).await;
		assert!(result.is_err(), "unreachable socket must fail capture setup");
	}

	/// A server that closes the connection mid-handshake (EOF) must surface as
	/// an error, not a hang or panic.
	#[tokio::test]
	async fn capture_fails_on_server_eof() {
		let dir = std::env::temp_dir().join(format!("moonshine-fake-pulse-eof-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("eof-server");
		let _ = std::fs::remove_file(&path);
		let listener = UnixListener::bind(&path).unwrap();
		std::thread::spawn(move || {
			let (stream, _) = listener.accept().unwrap();
			// Drop immediately: client reads EOF during Auth.
			drop(stream);
		});
		let (result, _frames) = run_capture_loop(path.clone()).await;
		assert!(result.is_err(), "EOF during handshake must fail capture setup");
		let _ = std::fs::remove_file(&path);
	}

	/// End-to-end over a fake server: the record stream is created, one frame
	/// of audio is pushed, `pump_frames` converts it to an `AudioFrame` and
	/// the capture ends gracefully when the server EOFs.
	#[tokio::test]
	async fn capture_streams_frame_and_ends_gracefully_on_eof() {
		let path = spawn_fake_server(SinkScenario::StreamOneFrame);
		let (result, frame_rx) = run_capture_loop(path.clone()).await;
		let _ = std::fs::remove_file(&path);

		// The single pushed frame must arrive as a proper AudioFrame: 5ms of
		// stereo f32 @48kHz = 240 samples per channel = 480 total.
		let frame = frame_rx
			.recv_timeout(std::time::Duration::from_secs(2))
			.expect("captured frame must be delivered");
		assert_eq!(frame.buf.len(), 480);
		assert!(
			frame.buf.iter().all(|s| (*s - 0.25).abs() < 1e-6),
			"samples must round-trip"
		);

		// Server EOF after the frame: capture exits with an error (stream
		// ended), never a hang — the caller treats this as best-effort stop.
		assert!(result.is_err(), "EOF after stream must end the capture, not hang");
	}
}
