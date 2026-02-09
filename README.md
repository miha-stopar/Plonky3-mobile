# Plonky3 Mobile (GPU Backends)

Mobile-focused scaffolding for Plonky3 with a Rust core and planned Vulkan/Metal/WebGPU GPU backends. The current app target is Android, with iOS/Metal intended for parity.

## Requirements
- Android Studio + SDK (compileSdk/targetSdk 34)
- NDK (API 26+)
- Rust toolchain
- `cargo-ndk`
Optional for Metal backend development (macOS):
- Xcode + Metal SDK

## Build native library
```
cargo ndk -t arm64-v8a -o app/src/main/jniLibs build --release --manifest-path native/Cargo.toml --features plonky3
```

## Run app
Open the repo in Android Studio and run on a device (Galaxy A55 or any Vulkan-capable device).

## Current behavior
- Runs a `fib_air` zk proof (modeled after `uni-stark/tests/fib_air.rs::test_zk`) in Rust.
- The Android UI calls the Rust JNI entry point and displays `fib_air zk ok` if proof+verify succeed.

## Enabling GPU path (placeholder)
```
cargo ndk -t arm64-v8a -o app/src/main/jniLibs build --release --manifest-path native/Cargo.toml --features "plonky3 gpu"
```

## Backend selection
The Android app can select a backend at runtime via JNI:
- `setBackend("cpu")`
- `setBackend("vulkan")`
- `setBackend("metal")`
- `setBackend("webgpu")`

Currently all GPU backends fall back to CPU until implemented.

## Next steps
- Implement the Vulkan FFT/LDE kernel (first target).
- Add Metal and WebGPU backends in parallel using the shared interface.
