use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;
use std::ffi::CString;
use std::os::raw::c_char;

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

#[cfg(target_os = "android")]
const ANDROID_LOG_INFO: i32 = 4;
#[cfg(target_os = "android")]
const ANDROID_LOG_ERROR: i32 = 6;

#[cfg(target_os = "android")]
extern "C" {
    fn __android_log_write(prio: i32, tag: *const c_char, text: *const c_char) -> i32;
}

#[cfg(target_os = "android")]
fn log_android(prio: i32, message: &str) {
    let tag = CString::new("plonky3").unwrap_or_else(|_| CString::new("plonky3").unwrap());
    let text = CString::new(message).unwrap_or_else(|_| CString::new("log message contains NUL").unwrap());
    unsafe {
        let _ = __android_log_write(prio, tag.as_ptr(), text.as_ptr());
    }
}

#[no_mangle]
pub extern "system" fn Java_com_plonky3_android_MainActivity_runFibAirZk(
    mut env: JNIEnv,
    _class: JClass,
) -> jstring {
    let message = {
        #[cfg(feature = "plonky3")]
        {
            let run = std::panic::catch_unwind(|| fib_air::run_fib_air_zk());
            let mut result = match run {
                Ok(Ok(value)) => value,
                Ok(Err(err)) => format!("fib_air zk failed: {err}"),
                Err(payload) => {
                    let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    format!("fib_air zk panicked: {msg}")
                }
            };
            if let Some(err) = gpu_dft::take_last_vulkan_error() {
                result.push_str("\nVulkan error: ");
                result.push_str(&err);
            }
            #[cfg(target_os = "android")]
            {
                let prio = if result.contains("failed") || result.contains("panicked") {
                    ANDROID_LOG_ERROR
                } else {
                    ANDROID_LOG_INFO
                };
                log_android(prio, &result);
            }
            result
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
pub extern "system" fn Java_com_plonky3_android_MainActivity_runDftBenchmark(
    mut env: JNIEnv,
    _class: JClass,
) -> jstring {
    let message = {
        #[cfg(feature = "plonky3")]
        {
            let run = std::panic::catch_unwind(|| fib_air::run_dft_benchmark());
            let result = match run {
                Ok(Ok(value)) => value,
                Ok(Err(err)) => format!("dft benchmark failed: {err}"),
                Err(payload) => {
                    let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    format!("dft benchmark panicked: {msg}")
                }
            };
            #[cfg(target_os = "android")]
            {
                let prio = if result.contains("failed") || result.contains("panicked") {
                    ANDROID_LOG_ERROR
                } else {
                    ANDROID_LOG_INFO
                };
                log_android(prio, &result);
            }
            result
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

#[no_mangle]
pub extern "system" fn Java_com_plonky3_android_MainActivity_isVulkanAvailable(
    mut env: JNIEnv,
    _class: JClass,
) -> jstring {
    let message = match backend_vulkan::is_vulkan_available() {
        Ok(()) => "Vulkan available".to_string(),
        Err(err) => format!("Vulkan unavailable: {err}"),
    };
    match env.new_string(message) {
        Ok(value) => value.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}
