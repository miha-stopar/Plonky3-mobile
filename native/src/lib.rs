use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;

#[cfg(feature = "plonky3")]
mod fib_air;
#[cfg(feature = "plonky3")]
mod gpu_dft;
#[cfg(feature = "plonky3")]
mod backend_vulkan;
#[cfg(feature = "plonky3")]
mod backend_metal;
#[cfg(feature = "plonky3")]
mod backend_webgpu;

#[no_mangle]
pub extern "system" fn Java_com_plonky3_android_MainActivity_runFibAirZk(
    mut env: JNIEnv,
    _class: JClass,
) -> jstring {
    let message = {
        #[cfg(feature = "plonky3")]
        {
            match fib_air::run_fib_air_zk() {
                Ok(value) => value,
                Err(err) => format!("fib_air zk failed: {err}"),
            }
        }
        #[cfg(not(feature = "plonky3"))]
        {
            "plonky3 feature disabled in native build".to_string()
        }
    };

    match env.new_string(message) {
        Ok(value) => value.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_plonky3_android_MainActivity_setBackend(
    mut env: JNIEnv,
    _class: JClass,
    backend: jstring,
) {
    let backend = unsafe { JString::from_raw(backend) };
    let value = match env.get_string(&backend) {
        Ok(value) => value,
        Err(_) => return,
    };
    let value = value.to_string_lossy();
    let _ = gpu_dft::set_backend_kind_from_str(&value);
}

#[no_mangle]
pub extern "system" fn Java_com_plonky3_android_MainActivity_setBAckend(
    env: JNIEnv,
    class: JClass,
    backend: jstring,
) {
    Java_com_plonky3_android_MainActivity_setBackend(env, class, backend);
}

#[no_mangle]
pub extern "system" fn Java_com_plonk3_android_MainActivity_setBackend(
    env: JNIEnv,
    class: JClass,
    backend: jstring,
) {
    Java_com_plonky3_android_MainActivity_setBackend(env, class, backend);
}
