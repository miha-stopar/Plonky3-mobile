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

const WG_X: u32 = 4u;
const WG_Y: u32 = 32u;
const TILE_ROWS: u32 = 256u;
const MAX_FUSED_STAGE: u32 = 7u; // stages with m <= TILE_ROWS

// One row tile per local x lane (column in the workgroup's column block).
var<workgroup> shared_tile: array<u32, 1024u>;

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

@compute @workgroup_size(4, 32, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let col = wid.x * WG_X + lid.x;
    if (col >= params.width) {
        return;
    }
    // This shader fuses a stage window inside one TILE_ROWS chunk.
    // Host only routes here when at least two stages can be fused.
    if (params.stage >= params.log_n || params.stage > MAX_FUSED_STAGE) {
        return;
    }

    let block = wid.y;
    let num_blocks = (params.height + TILE_ROWS - 1u) / TILE_ROWS;
    if (block >= num_blocks) {
        return;
    }

    let base_row = block * TILE_ROWS;
    let valid_rows = min(TILE_ROWS, params.height - base_row);
    if (valid_rows < 2u) {
        return;
    }
    let tile_off = lid.x * TILE_ROWS;

    // Load one column tile from global storage into workgroup memory.
    for (var lane = lid.y; lane < valid_rows; lane = lane + WG_Y) {
        let global_idx = (base_row + lane) * params.width + col;
        shared_tile[tile_off + lane] = src_data[global_idx];
    }
    workgroupBarrier();

    var s = params.stage;
    loop {
        if (s >= params.log_n || s > MAX_FUSED_STAGE) {
            break;
        }
        let half = 1u << s;
        let m = half << 1u;
        if (m > valid_rows) {
            break;
        }
        let twiddle_base = (1u << s) - 1u;
        let j_count = valid_rows >> 1u;
        for (var j = lid.y; j < j_count; j = j + WG_Y) {
            let block_s = j / half;
            let offset = j % half;
            let base = block_s * m + offset;
            if (base + half >= valid_rows) {
                continue;
            }
            let idx0 = tile_off + base;
            let idx1 = idx0 + half;
            let a = shared_tile[idx0];
            let b = shared_tile[idx1];
            let tw = twiddles[twiddle_base + offset];
            let t = mul_mod(b, tw);
            shared_tile[idx0] = add_mod(a, t);
            shared_tile[idx1] = sub_mod(a, t);
        }
        workgroupBarrier();
        s = s + 1u;
    }

    // Store tile result once back to global storage.
    for (var lane = lid.y; lane < valid_rows; lane = lane + WG_Y) {
        let global_idx = (base_row + lane) * params.width + col;
        dst_data[global_idx] = shared_tile[tile_off + lane];
    }
}
