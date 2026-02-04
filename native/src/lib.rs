use jni::objects::JClass;
use jni::sys::jstring;
use jni::JNIEnv;

#[cfg(feature = "plonky3")]
mod fib_air;

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
