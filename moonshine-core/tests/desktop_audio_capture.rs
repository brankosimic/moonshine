//! Integration test for desktop audio capture against the real user sound
//! server (PulseAudio or PipeWire's pulse compat layer).
//!
//! Skips automatically when no server is reachable (e.g. CI or headless).

#![cfg(target_os = "linux")]

use std::os::unix::net::UnixStream;
use std::time::Duration;

use pulseaudio::protocol::{self as pulse};
use pulseaudio::{Client, RecordBuffer};

fn user_pulse_socket_path() -> Option<std::path::PathBuf> {
	let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok().filter(|v| !v.is_empty())?;
	let path = std::path::PathBuf::from(runtime_dir).join("pulse/native");
	std::fs::metadata(&path).is_ok().then_some(path)
}

#[tokio::test]
async fn records_monitor_of_default_sink() {
	let Some(socket_path) = user_pulse_socket_path() else {
		eprintln!("skipping: no user pulse socket");
		return;
	};

	let socket = UnixStream::connect(&socket_path).expect("connect");
	let cookie = std::env::var("PULSE_COOKIE")
		.ok()
		.map(std::path::PathBuf::from)
		.or_else(|| {
			#[allow(deprecated)]
			std::env::home_dir().map(|home| home.join(".config/pulse/cookie"))
		})
		.filter(|path| path.is_file())
		.and_then(|path| std::fs::read(path).ok());
	let client = Client::new_unix(c"moonshine-capture-test", socket, cookie).expect("handshake");

	let server_info = client.server_info().await.expect("server info");
	let default_sink = server_info.default_sink_name.expect("default sink");
	let sink_info = client.sink_info_by_name(default_sink).await.expect("sink info");
	let monitor = sink_info.monitor_source_name.expect("monitor source");
	assert!(!monitor.to_bytes().is_empty());

	const RATE: u32 = 48000;
	let channels: u8 = 2;
	let frame_ms: u32 = 5;
	let bytes_per_frame: usize = (RATE * frame_ms / 1000 * channels as u32 * 4) as usize;

	let mut buffer = RecordBuffer::new(bytes_per_frame * 20);
	let params = pulse::RecordStreamParams {
		sample_spec: pulse::SampleSpec {
			format: pulse::SampleFormat::Float32Le,
			channels,
			sample_rate: RATE,
		},
		channel_map: pulse::ChannelMap::stereo(),
		source_name: Some(monitor),
		buffer_attr: pulse::stream::BufferAttr {
			max_length: (bytes_per_frame * 8) as u32,
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
		.expect("create record stream");

	use futures_util::io::AsyncReadExt;

	// Read ~500ms of audio in frame-sized chunks.
	let mut collected = Vec::<u8>::new();
	let frames: usize = 100;
	let mut chunk = vec![0u8; bytes_per_frame];
	for _ in 0..frames {
		buffer.read_exact(&mut chunk).await.expect("read captured audio");
		collected.extend_from_slice(&chunk);
	}

	assert_eq!(collected.len(), frames * bytes_per_frame);

	// The stream must deliver valid f32 samples (finite values, no NaN).
	let samples: Vec<f32> = collected
		.as_chunks::<4>()
		.0
		.iter()
		.map(|c| f32::from_le_bytes(*c))
		.collect();
	assert!(samples.iter().all(|s| s.is_finite()));
}

/// The record buffer must block (not spin/EOF) when no data has arrived yet.
#[tokio::test]
async fn record_buffer_blocks_until_data() {
	let Some(socket_path) = user_pulse_socket_path() else {
		eprintln!("skipping: no user pulse socket");
		return;
	};

	let socket = UnixStream::connect(&socket_path).expect("connect");
	let cookie = std::env::var("PULSE_COOKIE")
		.ok()
		.map(std::path::PathBuf::from)
		.or_else(|| {
			#[allow(deprecated)]
			std::env::home_dir().map(|home| home.join(".config/pulse/cookie"))
		})
		.filter(|path| path.is_file())
		.and_then(|path| std::fs::read(path).ok());
	let client = Client::new_unix(c"moonshine-block-test", socket, cookie).expect("handshake");

	let mut buffer = RecordBuffer::new(4096);
	let params = pulse::RecordStreamParams {
		sample_spec: pulse::SampleSpec {
			format: pulse::SampleFormat::Float32Le,
			channels: 2,
			sample_rate: 48000,
		},
		..Default::default()
	};

	let _stream = client
		.create_record_stream(params, buffer.as_record_sink())
		.await
		.expect("create record stream");

	use futures_util::io::AsyncReadExt;

	let mut out = [0u8; 16];
	let start = std::time::Instant::now();
	buffer.read_exact(&mut out).await.expect("read");
	// The default record stream (default sink source) delivers at wall-clock
	// pace; ensure we actually waited for server-pushed data rather than
	// returning immediately.
	assert!(start.elapsed() >= Duration::from_millis(1));
}
