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
use super::stream::video::{VideoStream, VideoStreamConfig, VideoStreamContext, VideoStreamHandle, FrameStats};

/// Wrapper for launched session states (regular or desktop).
pub(crate) enum LaunchedState {
	Regular(super::LaunchedSession),
	Desktop(DesktopLaunchedSession),
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
				Ok((ActiveState::Regular(active), video_notify, audio_notify))
			},
			Self::Desktop(session) => {
				let (active, video_notify, audio_notify) = session
					.start(video_config, stream_timeout, video_ctx, audio_ctx, stop, inhibit_sleep)
					.await?;
				Ok((ActiveState::Desktop(active), video_notify, audio_notify))
			},
		}
	}
}

/// Wrapper for active session states (regular or desktop).
pub(crate) enum ActiveState {
	Regular(super::ActiveSession),
	Desktop(DesktopActiveSession),
}

impl ActiveState {
	fn context(&self) -> &SessionContext {
		match self {
			Self::Regular(s) => s.context(),
			Self::Desktop(s) => s.context(),
		}
	}

	fn reset_video_stream(&self) {
		match self {
			Self::Regular(s) => s.reset_video_stream(),
			Self::Desktop(s) => s.reset_video_stream(),
		}
	}
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

		let audio = AudioStream::new(
			AudioStreamConfig::default(),
			address.clone(),
			stop.clone(),
		)
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

		let control_stream = ControlStream::new(
			ControlStreamConfig::default(),
			address,
			handles.input_tx,
			stop.clone(),
		)
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
		_video_config: VideoStreamConfig,
		_stream_timeout: u64,
		mut video_ctx: VideoStreamContext,
		audio_ctx: AudioStreamContext,
		_stop: ShutdownManager<SessionShutdownReason>,
		inhibit_sleep: bool,
	) -> Result<(DesktopActiveSession, Arc<Notify>, Arc<Notify>), ()> {
		let Self {
			context,
			launched_desktop,
			video_stream,
			audio,
			control_stream,
			hdr_metadata_rx,
			stop,
		} = self;

		let (width, height) = launched_desktop.ready.resolution;
		video_ctx.width = width;
		video_ctx.height = height;

		let keys_rx = context.keys.clone_rx().ok_or_else(|| {
			tracing::error!("Session keys not initialized");
		})?;

		let video_handle = video_stream
			.start(
				VideoStreamConfig::default(),
				video_ctx,
				keys_rx.clone(),
				stop.clone(),
			)
			.map_err(|()| tracing::error!("Failed to start video stream"))?;

		let audio_trigger = audio
			.start(audio_ctx, keys_rx)
			.map_err(|()| tracing::error!("Failed to start audio stream"))?;

		let video_start_notify = video_handle.clone_start_notify();
		let audio_start_notify = audio_trigger.clone_start_notify();

		let video_handle_for_resume = video_handle.clone();

		let control_ctx = ControlStreamContext::new(&context, false);
		control_stream.start(60, control_ctx, video_handle, audio_trigger, hdr_metadata_rx);

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
