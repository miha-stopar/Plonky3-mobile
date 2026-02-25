struct Params {
    // Number of columns in the batched matrix.
    width: u32,
    // Number of rows in each column (FFT length, power of two).
    height: u32,
    // Current FFT stage [0, log_n - 1].
    stage: u32,
    // log2(height), used for parameter consistency/debugging.
    log_n: u32,
    // Base offset into packed twiddle table for this stage.
    twiddle_base: u32,
    // Explicit padding to keep uniform layout predictable across backends.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
// In-place FFT data buffer.
// Layout is row-major matrix: index = row * width + col.
var<storage, read_write> data: array<u32>;

// Stage parameters are passed via Vulkan push constants.
var<push_constant> params: Params;

@group(0) @binding(2)
// Packed twiddles for all stages.
// Stage-local twiddle index = params.twiddle_base + offset.
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
    // One invocation handles exactly one butterfly lane for one column:
    //   (col, j) = (gid.x, gid.y)
    // where:
    //   col = which independent FFT column in the batch
    //   j   = which butterfly lane for the current stage
    let col = gid.x;
    let j = gid.y;

    // Over-dispatch in X is allowed; inactive lanes return immediately.
    if (col >= params.width) {
        return;
    }

    // Stage geometry:
    // m    = butterfly span at this stage
    // half = distance between pair elements
    //
    // Example for height=8:
    // stage 0: m=2, half=1  -> pairs (0,1), (2,3), ...
    // stage 1: m=4, half=2  -> pairs (0,2), (1,3), ...
    // stage 2: m=8, half=4  -> pairs (0,4), (1,5), ...
    let m = 1u << (params.stage + 1u);
    let half = m >> 1u;
    let height = params.height;

    // Map j to one butterfly pair within a block of size m.
    let block = j / half;
    let offset = j % half;
    let base = block * m + offset;

    // Over-dispatch in Y is allowed; out-of-range lanes return immediately.
    if (base + half >= height) {
        return;
    }

    // Row-major element addresses for this butterfly pair in this column.
    let idx0 = base * params.width + col;
    let idx1 = (base + half) * params.width + col;

    // Read pair.
    let a = data[idx0];
    let b = data[idx1];

    // Stage-specific twiddle selection.
    let twiddle = twiddles[params.twiddle_base + offset];
    let t = mul_mod(b, twiddle);

    // In-place radix-2 butterfly outputs.
    data[idx0] = add_mod(a, t);
    data[idx1] = sub_mod(a, t);
}
