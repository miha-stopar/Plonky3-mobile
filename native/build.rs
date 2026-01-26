use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let shader_path = PathBuf::from("shaders/add.wgsl");
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
    fs::write(out_dir.join("add.spv"), spv).expect("write SPIR-V");

    println!("cargo:rerun-if-changed=shaders/add.wgsl");
}
