//! Desktop stream (KDE Plasma screen capture) wiring.
//!
//! Everything desktop-streaming specific lives in this module and in
//! `session/desktop/`, `session/desktop_session.rs` and
//! `session/stream/video/pipeline/desktop_frame.rs`; the wider codebase is
//! only touched where strictly needed (app entry injection, session state
//! branching, and the desktop frame path in the video pipeline).
//!
//! The "Desktop" entry is exposed only when the host actually runs a KDE
//! Plasma Wayland session *and* the xdg-desktop-portal services needed for
//! RemoteDesktop/ScreenCast capture are reachable on the session bus.

use std::env;
use std::time::Duration;

use crate::session::application::ApplicationConfig;
use crate::session::stream::DesktopMode;

/// Boxart served for the "Desktop" app entry (KDE Plasma branded cover).
const DESKTOP_BOXART: &[u8] = include_bytes!("../../assets/desktop-boxart.png");

/// D-Bus name owned by KWin in a running Plasma Wayland session.
const KWIN_DBUS_NAME: &str = "org.kde.KWin";
/// xdg-desktop-portal frontend used for RemoteDesktop/ScreenCast sessions.
const PORTAL_DBUS_NAME: &str = "org.freedesktop.portal.Desktop";
/// Portal backend that implements ScreenCast/RemoteDesktop for Plasma.
const KDE_PORTAL_DBUS_NAME: &str = "org.freedesktop.impl.portal.desktop.kde";

/// Whether the environment looks like a KDE Plasma session.
///
/// `XDG_CURRENT_DESKTOP` is authoritative when set, but it is empty in
/// systemd-user services (the standard Moonshine deployment), so absence of
/// the variable must not be treated as "not KDE".
fn environment_suggests_kde() -> bool {
	let desktop = env::var("XDG_CURRENT_DESKTOP").unwrap_or_default();
	if !desktop.is_empty() {
		return desktop.to_lowercase().contains("kde");
	}
	// systemd-user service: no desktop vars set. Fall back to looking for
	// Plasma's runtime sockets in XDG_RUNTIME_DIR (kwin-wayland creates the
	// `kwin_wayland` socket in every Plasma Wayland session).
	let Some(runtime_dir) = env::var_os("XDG_RUNTIME_DIR").map(std::path::PathBuf::from) else {
		return false;
	};
	runtime_dir.join("kwin_wayland").exists()
}

/// Whether the Plasma session is reachable on the session bus and the portal
/// services needed for desktop capture are available.
///
/// `None` means the check could not run (no session bus, D-Bus error); the
/// caller then treats the capability as absent rather than guessing.
async fn desktop_capture_supported() -> Option<bool> {
	let connection = zbus::Connection::session().await.ok()?;
	async fn has_name(connection: &zbus::Connection, name: &str) -> Option<bool> {
		let proxy = zbus::Proxy::new(
			connection,
			"org.freedesktop.DBus",
			"/org/freedesktop/DBus",
			"org.freedesktop.DBus",
		)
		.await
		.ok()?;
		let owned: bool = proxy.call("NameHasOwner", &(name,)).await.ok()?;
		Some(owned)
	}
	let kwin = has_name(&connection, KWIN_DBUS_NAME).await?;
	let portal = has_name(&connection, PORTAL_DBUS_NAME).await?;
	let kde_portal = has_name(&connection, KDE_PORTAL_DBUS_NAME).await?;
	Some(kwin && (portal || kde_portal))
}

/// Synchronous wrapper: probe D-Bus on a short-lived runtime.
///
/// `inject_desktop_app` runs during startup (before the async runtime is
/// necessarily up), so this spins a minimal single-thread runtime just for
/// the probe. A D-Bus timeout or missing bus maps to `false`.
pub fn detect_desktop_session() -> bool {
	let Ok(runtime) = tokio::runtime::Builder::new_current_thread().enable_all().build() else {
		return false;
	};
	runtime.block_on(async {
		match tokio::time::timeout(Duration::from_secs(2), desktop_capture_supported()).await {
			Ok(Some(true)) => true,
			Ok(Some(false)) | Ok(None) | Err(_) => false,
		}
	})
}

fn desktop_boxart_path() -> Option<std::path::PathBuf> {
	let dir = env::temp_dir().join(format!("moonshine-desktop-boxart-{}", std::process::id()));
	std::fs::create_dir_all(&dir).ok()?;
	let path = dir.join("desktop.png");
	if !path.exists() {
		std::fs::write(&path, DESKTOP_BOXART).ok()?;
	}
	Some(path)
}

pub fn desktop_application() -> ApplicationConfig {
	ApplicationConfig {
		title: "Desktop".to_string(),
		boxart: desktop_boxart_path(),
		command: Vec::new(),
		pre_command: Vec::new(),
		post_command: Vec::new(),
		stdout: None,
		stderr: None,
		launch_timeout_secs: 2,
		desktop: true,
	}
}

pub fn inject_desktop_app(config: &mut crate::config::Config) {
	inject_desktop_app_with(config, detect_desktop_session);
}

/// Testable core of `inject_desktop_app`: `detect` reports whether the
/// desktop capture stack (Plasma session + portals) is currently reachable,
/// so unit tests can substitute the live D-Bus probe.
pub fn inject_desktop_app_with(config: &mut crate::config::Config, detect: impl Fn() -> bool) {
	match config.stream.desktop {
		DesktopMode::On => {
			if detect() {
				tracing::info!("Desktop streaming enabled by configuration.");
				config.applications.push(desktop_application());
			} else {
				tracing::warn!(
					"Desktop streaming is enabled ('On') but no KDE Plasma session with the \
					 capture portal was found; the Desktop entry is not exposed."
				);
			}
		},
		DesktopMode::Auto => {
			if desktop_streaming_supported_with(detect) {
				tracing::info!("KDE Plasma Wayland session detected — exposing Desktop streaming.");
				config.applications.push(desktop_application());
			} else {
				tracing::debug!("No KDE Plasma Wayland session detected — Desktop streaming disabled.");
			}
		},
		DesktopMode::Off => {},
	}
}

/// Testable core of `desktop_streaming_supported`.
fn desktop_streaming_supported_with(detect: impl Fn() -> bool) -> bool {
	environment_suggests_kde() && detect()
}

#[cfg(test)]
mod tests {
	use super::*;

	// Env-mutating tests take the crate-wide env lock (see `test_support`).
	#[test]
	fn environment_detection_uses_xdg_current_desktop_when_set() {
		let _lock = crate::test_support::env_lock();
		let previous = env::var_os("XDG_CURRENT_DESKTOP");
		let previous_runtime = env::var_os("XDG_RUNTIME_DIR");

		unsafe { env::set_var("XDG_CURRENT_DESKTOP", "KDE") };
		assert!(environment_suggests_kde());
		unsafe { env::set_var("XDG_CURRENT_DESKTOP", "kde-plasma") };
		assert!(environment_suggests_kde());
		unsafe { env::set_var("XDG_CURRENT_DESKTOP", "GNOME") };
		assert!(!environment_suggests_kde());
		unsafe { env::set_var("XDG_CURRENT_DESKTOP", "") };
		// Empty desktop var with no kwin socket in a scratch runtime dir.
		let scratch = env::temp_dir().join("moonshine-test-no-kwin");
		std::fs::create_dir_all(&scratch).unwrap();
		unsafe { env::set_var("XDG_RUNTIME_DIR", &scratch) };
		assert!(!environment_suggests_kde());

		match previous {
			Some(value) => unsafe { env::set_var("XDG_CURRENT_DESKTOP", value) },
			None => unsafe { env::remove_var("XDG_CURRENT_DESKTOP") },
		}
		match previous_runtime {
			Some(value) => unsafe { env::set_var("XDG_RUNTIME_DIR", value) },
			None => unsafe { env::remove_var("XDG_RUNTIME_DIR") },
		}
	}

	#[test]
	fn environment_detection_falls_back_to_kwin_socket() {
		let _lock = crate::test_support::env_lock();
		let previous_desktop = env::var_os("XDG_CURRENT_DESKTOP");
		let previous_runtime = env::var_os("XDG_RUNTIME_DIR");

		// Simulate the systemd-user service environment: no XDG_CURRENT_DESKTOP,
		// but Plasma's KWin socket exists in the runtime dir.
		let scratch = env::temp_dir().join("moonshine-test-kwin-socket");
		std::fs::create_dir_all(&scratch).unwrap();
		std::fs::write(scratch.join("kwin_wayland"), "").unwrap();
		unsafe { env::remove_var("XDG_CURRENT_DESKTOP") };
		unsafe { env::set_var("XDG_RUNTIME_DIR", &scratch) };
		assert!(environment_suggests_kde());

		match previous_desktop {
			Some(value) => unsafe { env::set_var("XDG_CURRENT_DESKTOP", value) },
			None => unsafe { env::remove_var("XDG_CURRENT_DESKTOP") },
		}
		match previous_runtime {
			Some(value) => unsafe { env::set_var("XDG_RUNTIME_DIR", value) },
			None => unsafe { env::remove_var("XDG_RUNTIME_DIR") },
		}
	}

	#[test]
	fn desktop_boxart_is_written_and_readable() {
		let path = desktop_boxart_path().expect("boxart path");
		assert!(path.is_file());
		let bytes = std::fs::read(&path).expect("boxart readable");
		// PNG magic.
		assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
		assert_eq!(bytes.len(), DESKTOP_BOXART.len());
	}

	fn test_config(mode: DesktopMode) -> crate::config::Config {
		crate::config::Config {
			applications: Vec::new(),
			application_scanners: Vec::new(),
			stream: crate::session::stream::StreamConfig {
				desktop: mode,
				..Default::default()
			},
			..Default::default()
		}
	}

	#[test]
	fn inject_off_mode_never_exposes_desktop() {
		for detect in [true, false] {
			let mut config = test_config(DesktopMode::Off);
			inject_desktop_app_with(&mut config, move || detect);
			assert!(
				config.applications.is_empty(),
				"Off must not expose the Desktop entry (detect={detect})"
			);
		}
	}

	#[test]
	fn inject_on_mode_requires_detection() {
		let mut config = test_config(DesktopMode::On);
		inject_desktop_app_with(&mut config, || false);
		assert!(config.applications.is_empty(), "On without detection must not expose");

		let mut config = test_config(DesktopMode::On);
		inject_desktop_app_with(&mut config, || true);
		assert_eq!(config.applications.len(), 1, "On with detection must expose");
		assert!(config.applications[0].desktop);
		assert_eq!(config.applications[0].title, "Desktop");
	}

	#[test]
	fn inject_auto_mode_needs_kde_environment_and_detection() {
		let _lock = crate::test_support::env_lock();
		let previous_desktop = env::var_os("XDG_CURRENT_DESKTOP");
		let previous_runtime = env::var_os("XDG_RUNTIME_DIR");

		// Scratch runtime dir without a kwin socket: environment not KDE.
		let scratch = env::temp_dir().join("moonshine-test-auto-mode");
		std::fs::create_dir_all(&scratch).unwrap();
		unsafe { env::set_var("XDG_RUNTIME_DIR", &scratch) };

		// Non-KDE environment: even a working capture stack must not expose.
		unsafe { env::set_var("XDG_CURRENT_DESKTOP", "GNOME") };
		let mut config = test_config(DesktopMode::Auto);
		inject_desktop_app_with(&mut config, || true);
		assert!(
			config.applications.is_empty(),
			"Auto with non-KDE environment must not expose"
		);

		// KDE environment + working detection: expose.
		unsafe { env::set_var("XDG_CURRENT_DESKTOP", "KDE") };
		let mut config = test_config(DesktopMode::Auto);
		inject_desktop_app_with(&mut config, || true);
		assert_eq!(config.applications.len(), 1, "Auto with KDE + detection must expose");

		// KDE environment + failed detection: do not expose.
		let mut config = test_config(DesktopMode::Auto);
		inject_desktop_app_with(&mut config, || false);
		assert!(
			config.applications.is_empty(),
			"Auto with KDE but failed detection must not expose"
		);

		match previous_desktop {
			Some(value) => unsafe { env::set_var("XDG_CURRENT_DESKTOP", value) },
			None => unsafe { env::remove_var("XDG_CURRENT_DESKTOP") },
		}
		match previous_runtime {
			Some(value) => unsafe { env::set_var("XDG_RUNTIME_DIR", value) },
			None => unsafe { env::remove_var("XDG_RUNTIME_DIR") },
		}
	}

	#[test]
	fn inject_appends_to_existing_application_list() {
		let mut config = test_config(DesktopMode::On);
		config
			.applications
			.push(crate::session::application::ApplicationConfig {
				title: "Existing".to_string(),
				desktop: false,
				..Default::default()
			});
		inject_desktop_app_with(&mut config, || true);
		assert_eq!(config.applications.len(), 2);
		assert_eq!(config.applications[0].title, "Existing");
		assert_eq!(config.applications[1].title, "Desktop");
	}
}
