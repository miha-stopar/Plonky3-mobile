# Plonky3 Android Vulkan

Android app scaffolding with a Rust JNI library and a Plonky3 `fib_air` zk proof-of-concept.

## Requirements
- Android Studio + SDK (compileSdk/targetSdk 34)
- NDK (API 26+)
- Rust toolchain
- `cargo-ndk`

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

## Next steps
- Identify which parts of `fib_air` we want to offload first (e.g., LDE/FFT, Merkle hashing).
- Implement a GPU-backed path behind the same JNI entry point.
