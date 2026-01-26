use jni::objects::JClass;
use jni::sys::jintArray;
use jni::JNIEnv;

mod poseidon_cpu;
mod vulkan;

#[no_mangle]
pub extern "system" fn Java_com_plonky3_android_MainActivity_runPoseidonTest(
    mut env: JNIEnv,
    _class: JClass,
    input: jintArray,
) -> jintArray {
    let input_vec = match read_int_array(&mut env, input) {
        Ok(values) => values,
        Err(_) => Vec::new(),
    };

    let output = match vulkan::try_run(&input_vec) {
        Ok(values) => values,
        Err(_) => poseidon_cpu::fallback(&input_vec),
    };

    write_int_array(&mut env, &output)
}

fn read_int_array(env: &mut JNIEnv, input: jintArray) -> jni::errors::Result<Vec<i32>> {
    let len = env.get_array_length(input)? as usize;
    let mut buffer = vec![0i32; len];
    env.get_int_array_region(input, 0, &mut buffer)?;
    Ok(buffer)
}

fn write_int_array(env: &mut JNIEnv, output: &[i32]) -> jintArray {
    let array = match env.new_int_array(output.len() as i32) {
        Ok(array) => array,
        Err(_) => return std::ptr::null_mut(),
    };

    if env.set_int_array_region(array, 0, output).is_err() {
        return std::ptr::null_mut();
    }

    array
}
