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

