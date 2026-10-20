#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum DesktopInputEvent {
	KeyDown {
		keycode: u32,
	},
	KeyUp {
		keycode: u32,
	},
	MouseMoveAbsolute {
		x: i16,
		y: i16,
		screen_width: i16,
		screen_height: i16,
	},
	MouseMoveRelative {
		dx: i16,
		dy: i16,
	},
	MouseButtonDown {
		button: u32,
	},
	MouseButtonUp {
		button: u32,
	},
	ScrollVertical {
		amount: i16,
	},
	ScrollHorizontal {
		amount: i16,
	},
	TouchDown {
		slot: u32,
		x: f32,
		y: f32,
	},
	TouchMove {
		slot: u32,
		x: f32,
		y: f32,
	},
	TouchUp {
		slot: u32,
	},
	TouchCancelAll,
	Pen {
		x: f32,
		y: f32,
		pressure: f32,
		tilt_x: f32,
		tilt_y: f32,
		down: bool,
	},
	TypeText {
		text: String,
	},
}
