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

@group(0) @binding(2)
var<storage, read> twiddles: array<u32>;

const PRIME: u32 = 0x78000001u;
const MONTY_MU: u32 = 0x88000001u;
const MONTY_MASK: u32 = 0xffffffffu;

fn add_mod(a: u32, b: u32) -> u32 {
    let sum = a + b;
    if (sum >= PRIME) {
        return sum - PRIME;
    }
    return sum;
}

fn sub_mod(a: u32, b: u32) -> u32 {
    if (a >= b) {
        return a - b;
    }
    return (a + PRIME) - b;
}

fn monty_reduce(x: u64) -> u32 {
    let t = (x * u64(MONTY_MU)) & u64(MONTY_MASK);
    let u = t * u64(PRIME);
    let x_sub_u = x - u;
    let over = x < u;
    let x_sub_u_hi = u32(x_sub_u >> 32u);
    if (over) {
        return x_sub_u_hi + PRIME;
    }
    return x_sub_u_hi;
}

fn mul_mod(a: u32, b: u32) -> u32 {
    let prod = u64(a) * u64(b);
    return monty_reduce(prod);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let j = gid.y;

    if (col >= params.width) {
        return;
    }

    let m = 1u << (params.stage + 1u);
    let half = m >> 1u;
    let height = params.height;

    let block = j / half;
    let offset = j % half;
    let base = block * m + offset;

    if (base + half >= height) {
        return;
    }

    let idx0 = base * params.width + col;
    let idx1 = (base + half) * params.width + col;

    let a = data[idx0];
    let b = data[idx1];

    let twiddle = twiddles[offset];
    let t = mul_mod(b, twiddle);

    data[idx0] = add_mod(a, t);
    data[idx1] = sub_mod(a, t);
}
