struct Params {
    width: u32,
    height: u32,
    stage: u32,
    log_n: u32,
    twiddle_base: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@group(0) @binding(1)
var<uniform> params: Params;

fn add_mod(a: u32, b: u32) -> u32 {
    return a + b;
}

fn sub_mod(a: u32, b: u32) -> u32 {
    return a - b;
}

fn mul_mod(a: u32, b: u32) -> u32 {
    return a * b;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let j = gid.x;

    if (row >= params.height) {
        return;
    }

    let m = 1u << (params.stage + 1u);
    let half = m >> 1u;
    let width = params.width;

    let block = j / half;
    let offset = j % half;
    let base = block * m + offset;

    if (base + half >= width) {
        return;
    }

    let idx0 = row * width + base;
    let idx1 = idx0 + half;

    let a = data[idx0];
    let b = data[idx1];

    let twiddle = params.twiddle_base;
    let t = mul_mod(b, twiddle);

    data[idx0] = add_mod(a, t);
    data[idx1] = sub_mod(a, t);
}
