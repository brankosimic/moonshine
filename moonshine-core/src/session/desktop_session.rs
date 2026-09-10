use std::sync::Arc;

use async_shutdown::ShutdownManager;
use tokio::sync::{Notify, watch};

use super::SessionContext;
use super::SessionShutdownReason;
use super::compositor::frame::HdrModeState;
use super::desktop::{Desktop, LaunchedDesktop};
use super::inhibit::SleepInhibitor;
use super::stream::audio::{AudioStream, AudioStreamConfig, AudioStreamContext};
use super::stream::control::{ControlStream, ControlStreamConfig, ControlStreamContext};
use super::stream::video::{FrameStats, VideoStream, VideoStreamConfig, VideoStreamContext, VideoStreamHandle};

/// Wrapper for launched session states (regular or desktop).
pub(crate) enum LaunchedState {
	Regular(Box<super::LaunchedSession>),
	Desktop(Box<DesktopLaunchedSession>),
}

impl LaunchedState {
	pub(crate) async fn start(
		self,
		video_config: VideoStreamConfig,
		stream_timeout: u64,
		video_ctx: VideoStreamContext,
		audio_ctx: AudioStreamContext,
		stop: ShutdownManager<SessionShutdownReason>,
		inhibit_sleep: bool,
	) -> Result<(ActiveState, Arc<Notify>, Arc<Notify>), ()> {
		match self {
			Self::Regular(session) => {
				let (active, video_notify, audio_notify) = session
					.start(video_config, stream_timeout, video_ctx, audio_ctx, stop, inhibit_sleep)
					.await?;
				Ok((ActiveState::Regular(Box::new(active)), video_notify, audio_notify))
			},
			Self::Desktop(session) => {
				let (active, video_notify, audio_notify) = session
					.start(video_config, stream_timeout, video_ctx, audio_ctx, stop, inhibit_sleep)
					.await?;
				Ok((ActiveState::Desktop(Box::new(active)), video_notify, audio_notify))
			},
		}
	}
}

/// Wrapper for active session states (regular or desktop).
pub(crate) enum ActiveState {
	Regular(Box<super::ActiveSession>),
	Desktop(Box<DesktopActiveSession>),
}

pub(crate) struct DesktopInitializedSession {
	context: SessionContext,
	desktop: Desktop,
	audio_stream: AudioStream,
	video_stream: VideoStream,
	control_stream: ControlStream,
	hdr_metadata_rx: watch::Receiver<HdrModeState>,
	stop: ShutdownManager<SessionShutdownReason>,
}

impl DesktopInitializedSession {
	pub(crate) fn context(&self) -> &SessionContext {
		&self.context
	}

	pub(crate) async fn new(
		context: SessionContext,
		address: String,
		stop: ShutdownManager<SessionShutdownReason>,
		stats_tx: tokio::sync::broadcast::Sender<FrameStats>,
	) -> Result<Self, ()> {
		let (hdr_metadata_tx, hdr_metadata_rx) = watch::channel(HdrModeState::new(context.hdr));

		let (desktop, handles) = Desktop::new(&context, stop.clone());

		let audio = AudioStream::new(AudioStreamConfig::default(), address.clone(), stop.clone())
			.await
			.map_err(|()| tracing::error!("Failed to create audio stream for desktop session"))?;

		let video_stream = VideoStream::new(
			VideoStreamConfig::default(),
			address.clone(),
			handles.frame_rx,
			hdr_metadata_tx,
			stop.clone(),
			stats_tx,
		)
		.await
		.map_err(|()| tracing::error!("Failed to create video stream for desktop session"))?;

		let control_stream =
			ControlStream::new(ControlStreamConfig::default(), address, handles.input_tx, stop.clone())
				.map_err(|()| tracing::error!("Failed to create control stream for desktop session"))?;

		Ok(Self {
			context,
			desktop,
			audio_stream: audio,
			video_stream,
			control_stream,
			hdr_metadata_rx,
			stop,
		})
	}

	pub(crate) async fn launch(self) -> Result<DesktopLaunchedSession, ()> {
		let Self {
			context,
			desktop,
			audio_stream: audio,
			video_stream,
			control_stream,
			hdr_metadata_rx,
			stop,
		} = self;

		let launched_desktop = desktop.launch().await?;

		Ok(DesktopLaunchedSession {
			context,
			launched_desktop,
			video_stream,
			audio,
			control_stream,
			hdr_metadata_rx,
			stop,
		})
	}
}

pub(crate) struct DesktopLaunchedSession {
	context: SessionContext,
	launched_desktop: LaunchedDesktop,
	video_stream: VideoStream,
	audio: AudioStream,
	control_stream: ControlStream,
	hdr_metadata_rx: watch::Receiver<HdrModeState>,
	stop: ShutdownManager<SessionShutdownReason>,
}

impl DesktopLaunchedSession {
	pub(crate) fn context(&self) -> &SessionContext {
		&self.context
	}

	pub(crate) async fn start(
		self,
		video_config: VideoStreamConfig,
		stream_timeout: u64,
		video_ctx: VideoStreamContext,
		audio_ctx: AudioStreamContext,
		stop: ShutdownManager<SessionShutdownReason>,
		inhibit_sleep: bool,
	) -> Result<(DesktopActiveSession, Arc<Notify>, Arc<Notify>), ()> {
		let Self {
			context,
			launched_desktop,
			video_stream,
			audio,
			control_stream,
			hdr_metadata_rx,
			stop: _session_stop,
		} = self;

		let (negotiated_w, negotiated_h) = (video_ctx.width, video_ctx.height);
		let (captured_w, captured_h) = launched_desktop.ready.resolution;
		tracing::info!(
			negotiated_w,
			negotiated_h,
			captured_w,
			captured_h,
			"Desktop stream using negotiated resolution; capture will be scaled if needed."
		);

		let mut video_ctx = video_ctx;
		let mut audio_ctx = audio_ctx;
		mark_desktop_contexts(&mut video_ctx, &mut audio_ctx);

		let keys_rx = context.keys.clone_rx().ok_or_else(|| {
			tracing::error!("Session keys not initialized");
		})?;

		let video_handle = video_stream
			.start(video_config, video_ctx, keys_rx.clone(), stop.clone())
			.map_err(|()| tracing::error!("Failed to start video stream"))?;

		let audio_trigger = audio
			.start(audio_ctx, keys_rx)
			.map_err(|()| tracing::error!("Failed to start audio stream"))?;

		let video_start_notify = video_handle.clone_start_notify();
		let audio_start_notify = audio_trigger.clone_start_notify();

		let video_handle_for_resume = video_handle.clone();

		let control_ctx = ControlStreamContext::new(&context, false);
		control_stream.start(
			stream_timeout,
			control_ctx,
			video_handle,
			audio_trigger,
			hdr_metadata_rx,
		);

		let sleep_inhibitor = if inhibit_sleep {
			SleepInhibitor::acquire().await
		} else {
			None
		};

		Ok((
			DesktopActiveSession {
				context,
				video_handle: video_handle_for_resume,
				sleep_inhibitor,
			},
			video_start_notify,
			audio_start_notify,
		))
	}
}

pub(crate) struct DesktopActiveSession {
	context: SessionContext,
	video_handle: VideoStreamHandle,
	#[allow(dead_code)]
	sleep_inhibitor: Option<SleepInhibitor>,
}

impl DesktopActiveSession {
	pub(crate) fn context(&self) -> &SessionContext {
		&self.context
	}

	pub(crate) fn reset_video_stream(&self) {
		self.video_handle.request_reset();
	}
}

/// Mark the stream contexts as belonging to a desktop session: the video
/// pipeline takes the desktop frame path, and the audio stream swaps its
/// private PulseAudio server for system-audio monitor capture.
pub(crate) fn mark_desktop_contexts(
	video_ctx: &mut super::stream::video::VideoStreamContext,
	audio_ctx: &mut super::stream::audio::AudioStreamContext,
) {
	video_ctx.desktop_mode = true;
	audio_ctx.desktop = true;
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn desktop_contexts_enable_desktop_mode_and_system_audio() {
		let mut video_ctx = VideoStreamContext::default();
		let mut audio_ctx = AudioStreamContext::default();
		assert!(
			!video_ctx.desktop_mode,
			"regular sessions must not take the desktop frame path"
		);
		assert!(
			!audio_ctx.desktop,
			"regular sessions must use the private PulseAudio server"
		);

		mark_desktop_contexts(&mut video_ctx, &mut audio_ctx);
		assert!(
			video_ctx.desktop_mode,
			"desktop sessions must take the desktop frame path"
		);
		assert!(audio_ctx.desktop, "desktop sessions must capture system audio");
	}

	#[test]
	fn default_contexts_are_regular_sessions() {
		// Guards against someone flipping a default and silently turning
		// every regular session into a desktop session.
		let video_ctx = VideoStreamContext::default();
		let audio_ctx = AudioStreamContext::default();
		assert!(!video_ctx.desktop_mode);
		assert!(!audio_ctx.desktop);
	}
}
