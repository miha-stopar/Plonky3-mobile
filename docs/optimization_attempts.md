# Optimization Attempts Log

This file tracks optimization experiments for `Plonky3-android` Vulkan FFT/DFT.

## How to add a new attempt

Copy this template and fill it in:

```md
## OPT-XXXX: <short title>

- Date range: YYYY-MM-DD to YYYY-MM-DD
- Status: proposed | in-progress | paused | reverted | landed
- Goal:
- Hypothesis:

### Code changes
- Files:
  - `native/src/...`
  - `native/shaders/...`
- Gating/flags:
  - How it was enabled/disabled.

### Benchmark protocol
- Device:
- Build:
- Workload sizes:
- Runs/repeats:

### Results (key numbers)
- Case A:
- Case B:

### Decision
- Keep / Revert / Gate / Follow-up
- Why:

### Follow-ups
- [ ] Item 1
- [ ] Item 2
```

---

## OPT-0001: Fused early-stage FFT shader (`fft_stage_fused.wgsl`)

- Date range: 2026-03-04 to 2026-03-05
- Status: reverted (runtime-disabled, code kept)
- Goal: reduce per-stage global-memory traffic and dispatch overhead for early FFT stages.
- Hypothesis: fuse two consecutive early stages (`s` and `s+1`) in one dispatch using workgroup shared memory to improve kernel time.

### Code changes

- Added fused shader:
  - `native/shaders/fft_stage_fused.wgsl`
- Added compilation in build script:
  - `native/build.rs` (`compile_wgsl("shaders/fft_stage_fused.wgsl", "fft_stage_fused.spv")`)
- Added second Vulkan compute pipeline:
  - `native/src/backend_vulkan.rs` pipeline creation for `fused_shader_module` and `fused_pipeline`
- Added fused dispatch path:
  - `dispatch_dims_fused(...)`
  - stage loop chooses `pipeline` vs `fused_pipeline`
  - stage increment becomes `+2` when fused, `+1` otherwise
- Current runtime gate:
  - `can_use_fused_stage(...)` is hard-disabled (`false`) in `native/src/backend_vulkan.rs`

### What was executed

At runtime, the intended fused path was:

1. Bind fused pipeline for eligible stages.
2. Dispatch fused kernel that computes two stages in one pass.
3. Advance stage index by 2.

When not eligible, fallback path:

1. Bind normal stage pipeline.
2. Dispatch one stage.
3. Advance stage index by 1.

### Benchmark protocol

- Device: Samsung Galaxy A55 (Vulkan driver string in logs: `24.3.9`, hash `e5014ade39`)
- Bench output source: app log `dft benchmark (repeats=10, warmup=1, stats=avg/median/p95)`
- Metrics used:
  - `vk_e2e` (upload + compute + readback + sync)
  - `vk_e2e_batched`
  - `vk_kernel` (GPU-only timed compute path)

### Results (representative)

Key workload: `h=16384, w=32` (largest tested matrix)

- Baseline before fused activation (2026-03-03):
  - `vk_e2e=11.990ms`, `vk_e2e_batched=9.747ms`, `vk_kernel=6.760ms`
- With fused shader active (2026-03-04):
  - `vk_e2e=21.681ms`, `vk_e2e_batched=16.800ms`, `vk_kernel=11.301ms`
- After runtime disable + follow-up host-side cleanup (2026-03-05):
  - `vk_e2e=11.614ms`, `vk_e2e_batched=8.985ms`, `vk_kernel=5.396ms`

Interpretation:
- Fused attempt caused a strong regression (both kernel and e2e).
- Disabling fused path recovered/improved performance.

### Decision

- Decision: keep fused infrastructure in tree but disable runtime usage.
- Reason:
  - Current fused design increased kernel time on target hardware.
  - Risk too high to keep it active until a narrower/fixed variant is validated.

### Follow-ups

- [ ] Reintroduce fused path only behind a stricter predicate (small/tile-friendly shapes, specific stage windows).
- [ ] Add explicit perf guardrails: automatically compare fused vs non-fused in one run and report deltas.
- [ ] Investigate fused regression causes (occupancy, barriers, LDS/shared-memory pressure, extra instructions).
- [ ] Keep logs for each new attempt in this file using the template above.

---

## OPT-0002: Tiled multi-stage fused FFT kernel + dynamic stage-span dispatch

- Date range: 2026-03-10 to 2026-03-10
- Status: landed
- Goal: make the fused path beneficial on wider column batches by increasing tile locality and fusing more early stages per dispatch.
- Hypothesis: replacing fixed 2-stage fusion with a tile-resident stage loop (up to stage constraints) reduces global-memory traffic enough to improve kernel time, especially for wide `w`.

### Code changes

- Shader redesign (`native/shaders/fft_stage_fused.wgsl`):
  - Workgroup shape changed from `8x8` to `4x32`.
  - Shared tile changed from fixed `MAX_TILE=32` rows to `TILE_ROWS=256`.
  - Removed fixed “exactly two stages (`s`, `s+1`)” logic.
  - Added in-shader loop over stages while `m <= valid_rows` and `s <= MAX_FUSED_STAGE`.
- Host policy updates (`native/src/backend_vulkan.rs`):
  - Added `fused_stage_span(...)` to compute how many stages are fused for current `stage`.
  - `can_use_fused_stage(...)` now enables fused path when span > 1.
  - Stage advance changed from fixed `stage += 2` to `stage += fused_stage_span(...)`.
  - Fused dispatch geometry changed to fixed tile blocks:
    - `x = ceil(width / 4)`, `y = ceil(height / 256)`.
- Benchmark coverage widened (`native/src/fib_air.rs`):
  - Added `(4096,64)`, `(4096,128)`, `(16384,64)`, `(16384,128)`, `(256,16000)`.

### Benchmark protocol

- Device: Samsung Galaxy A55 (driver log: `24.3.9`, hash `e5014ade39`).
- Benchmark source: app log from `run_dft_benchmark()` (`repeats=10`, `warmup=1`).
- Metrics:
  - `cpu` (reference CPU path)
  - `vk_e2e` (upload + compute + readback + sync)
  - `vk_e2e_batched`
  - `vk_kernel` (GPU kernel timing)

### Results (full comparison from 2026-03-10 run)

| h,w | CPU avg ms | VK e2e avg ms | e2e speedup | VK e2e batched avg ms | batched speedup | VK kernel avg ms | kernel speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256,8 | 0.013 | 0.746 | 0.02x | 0.252 | 0.05x | 0.176 | 0.07x |
| 1024,8 | 0.052 | 0.979 | 0.05x | 0.319 | 0.16x | 0.452 | 0.12x |
| 4096,8 | 0.240 | 1.497 | 0.16x | 0.800 | 0.30x | 1.216 | 0.20x |
| 16384,8 | 1.409 | 3.966 | 0.36x | 2.480 | 0.57x | 1.913 | 0.74x |
| 4096,32 | 0.963 | 3.090 | 0.31x | 2.302 | 0.42x | 1.886 | 0.51x |
| 16384,32 | 7.022 | 10.708 | 0.66x | 9.593 | 0.73x | 6.833 | 1.03x |
| 4096,64 | 2.255 | 6.241 | 0.36x | 5.630 | 0.40x | 6.026 | 0.37x |
| 4096,128 | 4.661 | 14.764 | 0.32x | 11.255 | 0.41x | 7.066 | 0.66x |
| 16384,64 | 12.057 | 25.450 | 0.47x | 21.543 | 0.56x | 14.274 | 0.84x |
| 16384,128 | 21.170 | 47.550 | 0.45x | 34.401 | 0.62x | 15.576 | 1.36x |
| 256,16000 | 22.926 | 51.023 | 0.45x | 44.702 | 0.51x | 27.341 | 0.84x |

### Decision

- Decision: keep landed changes; this is better than the previous fused attempt and improves kernel competitiveness on large/wide cases.
- Why:
  - Kernel-only wins now appear on large shapes (for example `h=16384,w=32` and `h=16384,w=128`).
  - Wide stress case `h=256,w=16000` improved substantially versus the earlier fused design.
  - End-to-end path remains CPU-faster due to transfer/sync overhead.

### Follow-ups

- [ ] Add automatic A/B benchmark mode: non-fused vs fused policy in one run, same process, same thermal window.
- [ ] Tune fused gate (`width`, stage span, and possibly height-based constraints) per device profile.
- [ ] Improve e2e overlap and transfer strategy (persistent buffers / async pipelining) to close the e2e gap.
