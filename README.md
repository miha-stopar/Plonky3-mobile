# Plonky3 Android Vulkan

Android app scaffolding with a Rust JNI library and a Vulkan compute proof-of-concept.

## Requirements
- Android Studio + SDK (compileSdk/targetSdk 34)
- NDK (API 26+)
- Rust toolchain
- `cargo-ndk`

## Build native library
```
cargo ndk -t arm64-v8a -o app/src/main/jniLibs build --release --manifest-path native/Cargo.toml
```

## Run app
Open the repo in Android Studio and run on a device (Galaxy A55 or any Vulkan-capable device).

## Current GPU behavior
The Vulkan compute shader runs a simple `+1` transformation over the input buffer. This is a stand-in for the Poseidon2 permutation kernel.

## Next steps
- Replace `native/shaders/add.wgsl` with a Poseidon2 permutation kernel.
- Wire the kernel into the JNI call to match `poseidon2-air/examples/prove_poseidon2_baby_bear_keccak_zk.rs`.
