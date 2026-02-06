use std::env;
use std::fs;
use std::path::PathBuf;

fn compile_wgsl(shader_path: &str, spv_name: &str) {
    let shader_path = PathBuf::from(shader_path);
    let source = fs::read_to_string(&shader_path).expect("read WGSL shader");

    let module = naga::front::wgsl::parse_str(&source).expect("parse WGSL");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validate WGSL");

    let spv = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        None,
    )
    .expect("emit SPIR-V");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let spv_bytes: Vec<u8> = spv.into_iter().flat_map(u32::to_le_bytes).collect();
    fs::write(out_dir.join(spv_name), spv_bytes).expect("write SPIR-V");

    println!("cargo:rerun-if-changed={}", shader_path.display());
}

fn main() {
    compile_wgsl("shaders/add.wgsl", "add.spv");
    compile_wgsl("shaders/fft_stage.wgsl", "fft_stage.spv");
}
