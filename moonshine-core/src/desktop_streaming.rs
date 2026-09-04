use std::env;

use crate::session::application::ApplicationConfig;
use crate::session::stream::DesktopMode;

pub fn detect_desktop_session() -> bool {
	let desktop = env::var("XDG_CURRENT_DESKTOP").unwrap_or_default();
	let wayland_display = env::var("WAYLAND_DISPLAY").unwrap_or_default();
	desktop.to_lowercase().contains("kde") && !wayland_display.is_empty()
}

pub fn desktop_application() -> ApplicationConfig {
	ApplicationConfig {
		title: "Desktop".to_string(),
		boxart: None,
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
	match config.stream.desktop {
		DesktopMode::On => {
			tracing::info!("Desktop streaming enabled by configuration.");
			config.applications.push(desktop_application());
		},
		DesktopMode::Auto => {
			if detect_desktop_session() {
				tracing::info!("KDE Plasma Wayland session detected — exposing Desktop streaming.");
				config.applications.push(desktop_application());
			} else {
				tracing::debug!("No KDE Plasma Wayland session detected — Desktop streaming disabled.");
			}
		},
		DesktopMode::Off => {},
	}
}
