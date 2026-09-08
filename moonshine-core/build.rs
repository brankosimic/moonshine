use std::process::Command;

fn main() {
	let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
	let shader = "shader/scaler.comp";
	let spv = format!("{out_dir}/scaler.spv");

	let status = Command::new("glslc")
		.args(["--target-env=vulkan1.2", "-o", &spv, shader])
		.status()
		.expect("failed to run glslc (is it installed?)");
	assert!(status.success(), "glslc failed to compile {shader}");

	println!("cargo:rerun-if-changed={shader}");
}
