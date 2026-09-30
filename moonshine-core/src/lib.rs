// TODO: Remove this when a proper error type is implemented for all functions that return `Result<(), ()>`.
#![allow(clippy::result_unit_err)]

pub mod app_scanner;
pub mod clients;
pub mod config;
pub(crate) mod crypto;
pub mod desktop_streaming;
pub mod discovery;
pub mod healthcheck;
pub mod rtsp;
pub mod session;
pub(crate) mod state;
pub mod tls;
pub mod webserver;

/// Shared lock for tests that mutate process-global environment variables.
///
/// Rust runs unit tests as parallel threads of a single process; env-var
/// mutations in different test modules must be serialized against each
/// other, not just within one module. Env-mutating tests across the crate
/// (`desktop_streaming`, `stream::audio::desktop_capture`,
/// `app_scanner::desktop`) take this lock for the duration of their body.
#[cfg(test)]
pub mod test_support {
	use std::sync::Mutex;
	use std::sync::MutexGuard;

	static ENV_LOCK: Mutex<()> = Mutex::new(());

	/// Acquire the crate-wide env-var lock. Hold the guard for the whole
	/// test body, including the restore of previous values.
	#[allow(clippy::significant_drop_tightening)]
	pub fn env_lock() -> MutexGuard<'static, ()> {
		ENV_LOCK.lock().unwrap()
	}
}

/// Reasons for initiating a global shutdown.
///
/// Used as the type parameter for `ShutdownManager<ShutdownReason>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShutdownReason {
	/// Application quit signal (Ctrl+C or SIGTERM).
	AppQuit = 1,
	/// HTTP webserver is shutting down.
	HttpShutdown = 2,
	/// HTTPS webserver is shutting down.
	HttpsShutdown = 3,
	/// RTSP server is shutting down.
	RtspShutdown = 4,
	/// Session manager guard token (trigger_shutdown_token, not a shutdown trigger).
	SessionManagerShutdown = 5,
}
