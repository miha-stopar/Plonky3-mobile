@group(0) @binding(0)
var<storage, read> input_data: array<i32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<i32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&input_data)) {
        return;
    }
    output_data[idx] = input_data[idx] + 1;
}
