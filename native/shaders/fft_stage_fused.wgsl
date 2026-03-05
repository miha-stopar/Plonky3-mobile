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
var<storage, read_write> dst_data: array<u32>;

@group(0) @binding(1)
var<storage, read> src_data: array<u32>;

var<push_constant> params: Params;

@group(0) @binding(2)
var<storage, read> twiddles: array<u32>;

const PRIME: u32 = 0x78000001u;
const MONTY_MU: u32 = 0x88000001u;
const MONTY_MASK: u32 = 0xffffffffu;

const WG_X: u32 = 8u;
const WG_Y: u32 = 8u;
const MAX_TILE: u32 = 32u;

// One column tile per local x lane; max tile is 32 rows.
var<workgroup> shared_tile: array<u32, 256u>;

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

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let col = wid.x * WG_X + lid.x;
    if (col >= params.width) {
        return;
    }
    // This shader fuses exactly two consecutive stages starting at `stage`.
    if (params.stage + 1u >= params.log_n || params.stage > 3u) {
        return;
    }

    let tile_size = 1u << (params.stage + 2u); // 4 * 2^stage
    if (tile_size > MAX_TILE) {
        return;
    }

    let block = wid.y;
    let num_blocks = params.height / tile_size;
    if (block >= num_blocks) {
        return;
    }

    let base_row = block * tile_size;
    let tile_off = lid.x * MAX_TILE;

    // Load one column tile from global storage into workgroup memory.
    for (var lane = lid.y; lane < tile_size; lane = lane + WG_Y) {
        let global_idx = (base_row + lane) * params.width + col;
        shared_tile[tile_off + lane] = src_data[global_idx];
    }
    workgroupBarrier();

    // Stage s inside the tile.
    let half0 = 1u << params.stage;
    let m0 = half0 << 1u;
    let j_count = tile_size >> 1u;
    for (var j = lid.y; j < j_count; j = j + WG_Y) {
        let block0 = j / half0;
        let offset0 = j % half0;
        let base0 = block0 * m0 + offset0;
        let idx0 = tile_off + base0;
        let idx1 = idx0 + half0;
        let a = shared_tile[idx0];
        let b = shared_tile[idx1];
        let twiddle0 = twiddles[params.twiddle_base + offset0];
        let t0 = mul_mod(b, twiddle0);
        shared_tile[idx0] = add_mod(a, t0);
        shared_tile[idx1] = sub_mod(a, t0);
    }
    workgroupBarrier();

    // Stage s+1 inside the same tile (keeps intermediate in workgroup memory).
    let half1 = half0 << 1u;
    let m1 = half1 << 1u;
    let twiddle_base1 = params.twiddle_base + half0;
    for (var j = lid.y; j < j_count; j = j + WG_Y) {
        let block1 = j / half1;
        let offset1 = j % half1;
        let base1 = block1 * m1 + offset1;
        let idx0 = tile_off + base1;
        let idx1 = idx0 + half1;
        let a = shared_tile[idx0];
        let b = shared_tile[idx1];
        let twiddle1 = twiddles[twiddle_base1 + offset1];
        let t1 = mul_mod(b, twiddle1);
        shared_tile[idx0] = add_mod(a, t1);
        shared_tile[idx1] = sub_mod(a, t1);
    }
    workgroupBarrier();

    // Store tile result once back to global storage.
    for (var lane = lid.y; lane < tile_size; lane = lane + WG_Y) {
        let global_idx = (base_row + lane) * params.width + col;
        dst_data[global_idx] = shared_tile[tile_off + lane];
    }
}
