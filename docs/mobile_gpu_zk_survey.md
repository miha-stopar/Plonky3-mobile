# Mobile GPU Optimizations for SNARKs and STARKs

This document is a research-and-engineering survey focused on zero-knowledge proving on phone-class GPUs: Android Vulkan/WebGPU devices, Apple Metal devices, and browser-exposed mobile GPUs.

## Executive Summary

Public ZK GPU acceleration is still weighted toward server-oriented CUDA work, with representative examples including [ICICLE](https://github.com/ingonyama-zk/icicle), [cuZK](https://github.com/speakspeak/cuZK), and [PipeZK](https://www.microsoft.com/en-us/research/publication/pipezk-accelerating-zero-knowledge-proof-with-a-pipelined-architecture/). Mobile and browser-facing efforts are fewer, but they are growing, especially around Vulkan, Metal, and WebGPU.

For STARK-like and post-quantum-transparent systems, mobile GPUs are attractive because much of the workload is hashes, field arithmetic, and FFT/NTT-like operations rather than elliptic-curve pairings. But this does not automatically make them easy targets. The bottlenecks move toward memory bandwidth, trace size, Merkle commitment traffic, and sustained thermally-limited throughput.

The practical lesson is that phone-friendly ZK systems should be designed around:

- 32-bit-friendly finite fields and reductions.
- Tiled or fused FFT/NTT kernels with minimal global permutations.
- End-to-end GPU residency where possible, because CPU<->GPU synchronization often dominates.
- Memory-aware protocol choices, especially for FRI-family and other transparent/post-quantum constructions.
- Dynamic per-device policies rather than a single fixed kernel strategy.

## Why FFT/NTT Libraries on GPUs Gravitate Toward Stockham-Style Kernels

The short version is: Stockham-style FFTs are often friendlier to GPU memory systems than classic decimation schemes that require a separate global permutation.

The core reason appears explicitly in older Microsoft GPU FFT work. In [Fast Computation of General Fourier Transforms on GPUs](https://www.microsoft.com/en-us/research/publication/fast-computation-of-general-fourier-transforms-on-gpus/), Microsoft states that its "basic building block" is a radix-2 Stockham formulation because it avoids expensive bit reversals and exploits GPU memory bandwidth efficiently. In [High Performance Discrete Fourier Transforms on Graphics Processors](https://www.microsoft.com/en-us/research/publication/high-performance-discrete-fourier-transforms-on-graphics-processors/), the authors again emphasize hierarchical FFT algorithms that exploit shared memory using a Stockham formulation.

That choice matches the usual GPU engineering tradeoff:

- Classic DIT/DIF Cooley-Tukey forms are mathematically fine, but practical implementations often need an input or output permutation such as bit-reversal.
- A separate global permutation is expensive on a GPU because it is mostly memory traffic with poor arithmetic intensity.
- Stockham "autosort" kernels fold the permutation into the staged passes, which produces regular per-stage access patterns and avoids a dedicated global reorder step.
- Those regular staged patterns are easier to tile into shared/threadgroup memory and easier to batch over many transforms.

This does not mean Stockham is always universally best. High-performance libraries are usually hybrid:

- mixed-radix for favorable sizes,
- Bluestein or Rader for awkward sizes,
- four-step or tiled decompositions for large transforms,
- device-specific specializations for small kernels.

Still, the broad GPU trend is real: the closer a library is to bandwidth-bound FFT execution, the more valuable it becomes to avoid standalone global permutations.

### Representative Libraries

| Library | GPU vendors / APIs | Public documentation / implementation notes |
|---|---|---|
| [VkFFT](https://github.com/DTolm/VkFFT) | NVIDIA / AMD / Intel / Apple via Vulkan, CUDA, HIP, OpenCL, Level Zero, Metal | The [VkFFT API guide](https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/documentation/VkFFT_API_guide.tex#L458) states that VkFFT reduces its algorithms to a mixed-radix Cooley-Tukey FFT in Stockham autosort form. |
| [SMFFT](https://github.com/KAdamek/SMFFT) | NVIDIA CUDA | The project includes both Cooley-Tukey and Stockham implementations; the Stockham path is presented as an autosort FFT that avoids a separate reorder step. |
| [Microsoft GPU FFT research](https://www.microsoft.com/en-us/research/publication/fast-computation-of-general-fourier-transforms-on-gpus/) | Early GPU work, NVIDIA-focused experiments | The Microsoft papers explicitly use radix-2 Stockham as a basic building block and combine it with hierarchical use of shared memory. |
| [rocFFT](https://rocm.docs.amd.com/projects/rocFFT/en/develop/design/runtime_compilation.html) | AMD ROCm / HIP | AMD's runtime-compilation and codegen docs describe rocFFT as being dominated by Stockham kernels, including specialized tiled, strided, and batched variants. |
| [cuFFT](https://docs.nvidia.com/cuda/cufft/) | NVIDIA CUDA | NVIDIA's public docs describe Cooley-Tukey decompositions for favorable sizes and Bluestein for arbitrary sizes; the public interface does not present one single universal kernel family. |

### Why This Matters for ZK

For ZK proving, FFT/NTT passes often dominate polynomial evaluation/interpolation work. If every pass also forces extra global permutation traffic, mobile GPUs suffer twice:

- limited bandwidth relative to server GPUs,
- stricter thermal envelopes that punish sustained memory-heavy kernels.

That is why the Stockham-style lesson transfers directly into phone-class ZK implementations: avoiding standalone global permutations is often worth more than shaving a few arithmetic instructions.

## Representative GPU FFT/NTT Papers and Engineering Lessons

Several papers and implementation lines matter here, not only one.

### Bailey and Hierarchical-Memory FFTs

David Bailey's [FFTs in external or hierarchical memory](https://ntrs.nasa.gov/citations/19900047338) is one of the clearest statements of the `4-step` idea. The large transform is reorganized into shorter transforms and transposes so that memory access stays regular and unit-stride as much as possible. This is still one of the most relevant design ideas for phone GPUs because threadgroup/shared memory is too small to hold an entire large transform.

### Microsoft GPU FFT Papers

The Microsoft papers on [Fast Computation of General Fourier Transforms on GPUs](https://www.microsoft.com/en-us/research/publication/fast-computation-of-general-fourier-transforms-on-gpus/), [High Performance Discrete Fourier Transforms on Graphics Processors](https://www.microsoft.com/en-us/research/publication/high-performance-discrete-fourier-transforms-on-graphics-processors/), and [Auto-tuning of Fast Fourier Transform on Graphics Processors](https://www.microsoft.com/en-us/research/publication/auto-tuning-of-fast-fourier-transform-on-graphics-processors/) are important because they connect three ideas that still dominate current practice:

- Stockham-style autosort kernels,
- hierarchical use of shared memory,
- and hardware-specific tuning of launch parameters.

### GPU NTT Papers from the HE Literature

The Sabanci work around [GPU Acceleration of Approximate Homomorphic Encryption](https://eprint.iacr.org/2021/124), [Homomorphic Encryption on GPU](https://eprint.iacr.org/2022/1222), [Two Algorithms for Fast GPU Implementation of NTT](https://eprint.iacr.org/2023/1410), and the newer [High-Performance NTT on GPU Through radix2-CT and 4-Step Algorithms](https://eprint.iacr.org/2025/388) is useful because it treats NTT as a memory-system problem. Across these papers, the recurring implementation ideas are:

- merge several stages into one kernel when locality allows,
- use `4-step` decompositions for large transforms,
- tune block shape / launch geometry rather than treating them as constants,
- and select algorithm families by problem size instead of forcing one universal kernel.

### Transparent-Prover Case Studies

[Air-FRI: Acceleration of the FRI Protocol on the GPU for zkSNARK Applications](https://uwspace.uwaterloo.ca/items/6cee69e5-c9cf-4ef2-ba59-ca3541b58f71) matters because it moves the discussion from isolated NTT kernels to the broader transparent-prover pipeline. The important shift is that the performance question becomes "how does the full prover move data through evaluation, commitment, and query phases?" rather than "how fast is one butterfly?".

The common lesson across these papers is straightforward: phone GPUs need the same algorithmic ingredients as server GPUs, but the constraints are tighter, so memory layout, kernel fusion, and transform decomposition matter even more.

## Representative Projects and Public Implementations

### Server-Oriented Acceleration Stacks

- [ICICLE](https://github.com/ingonyama-zk/icicle): hardware acceleration library for ZK primitives across multiple backends, but public open source remains strongest on the CPU/frontend side while CUDA acceleration has historically been the main proving story.
- [cuZK](https://github.com/speakspeak/cuZK): GPU implementation of zkSNARK proving with emphasis on MSM and NTT.
- [PipeZK](https://www.microsoft.com/en-us/research/publication/pipezk-accelerating-zero-knowledge-proof-with-a-pipelined-architecture/): hardware-architecture paper, important because it frames ZK acceleration as a pipeline and memory-traffic problem, not just a raw-compute problem.
- [ZKPoG](https://eprint.iacr.org/2025/765): end-to-end GPU acceleration including witness generation for Plonkish pipelines.
- [ICICLE-Stwo](https://www.ingonyama.com/post/introducing-icicle-stwo-a-gpu-accelerated-stwo-prover): GPU acceleration of StarkWare's Stwo prover, reporting 3.25x to 7x over the SIMD backend.

### Mobile and Browser-Facing Efforts

- [IMP1](https://github.com/ingonyama-zk/imp1): explicitly a mobile-first proving framework for iOS and Android, described as a mobile-optimized Groth16 prover. The README claims up to 3x faster than Rapidsnark on supported devices.
- [PocketProof](https://www.ingonyama.com/pocketproof): packaging and distribution layer for the same mobile proving direction.
- [ICICLE Metal backend docs](https://dev.ingonyama.com/start/architecture/install_gpu_backend) and [architecture overview](https://dev.ingonyama.com/3.6.0/icicle/arch_overview): shows serious movement beyond CUDA, at least for Apple devices.
- [zkSecurity's WebGPU/Stwo article](https://blog.zksecurity.xyz/posts/webgpu/): one of the clearest public practitioner write-ups on mobile-compatible GPU proving via WebGPU, including real engineering constraints.
- [QED's plonky2-wasm](https://github.com/QEDProtocol/plonky2-wasm): browser/client-side experimentation with WebGPU-oriented proving.
- [msm-webgpu](https://lib.rs/crates/msm-webgpu): WebGPU MSM work that points toward browser/mobile-compatible ECC acceleration, though still far from a mature proving stack.

The main gap is not a complete absence of work. The gap is that the public ecosystem is still much richer for server CUDA than for reusable phone-grade Vulkan / Metal / WebGPU proving primitives.

## Why Phone GPUs Differ from "Normal" CUDA GPUs

It is a mistake to think of phone GPUs as merely "smaller CUDA GPUs". The programming model, memory hierarchy, tooling, and performance envelope are different in ways that matter directly for ZK kernels.

### Quick Comparison: Server CUDA GPU vs Phone GPU

| Dimension | Typical server CUDA environment | Typical phone GPU environment | Why it matters for ZK |
|---|---|---|---|
| Programming stack | CUDA, cuFFT, cuBLAS, Nsight, mature vendor libraries | Vulkan compute, Metal, WebGPU, less mature math/profiling stack | More kernel engineering must be done manually on phones |
| Vendor variability | Mostly NVIDIA-specific if using CUDA | Qualcomm, ARM, Apple, Samsung/Xclipse, browser stacks | Kernel tuning is less portable |
| Warp/subgroup behavior | Warp-centric assumptions are common and stable enough to optimize around | Vulkan/Metal subgroup behavior is more implementation-dependent | Shuffle/subgroup-heavy kernels are harder to port reliably |
| Shared/on-chip memory | Larger, better documented, and backed by mature occupancy tooling | Smaller, tighter threadgroup/shared memory budgets | Large NTT stages must be tiled more aggressively |
| Peak vs sustained throughput | Peak numbers are often meaningful for long jobs | Thermal throttling can dominate after short bursts | Benchmarks must emphasize sustained proving throughput |
| Host-device boundary | PCIe or server interconnect, but powerful discrete GPU execution | Often unified memory, but lower bandwidth and more CPU/GPU contention | Round-trip reductions help, but memory bandwidth is still scarce |
| Library ecosystem | Mature FFT/MSM/linear-algebra acceleration culture | Fragmented, younger, often application-specific | Proof-system design must compensate for missing libraries |
| Deployment target | Homogeneous proving servers or clusters | End-user devices with widely varying capabilities | Runtime autotuning matters more |

### 1. API and Tooling Fragmentation

CUDA gives developers a mature stack: language extensions, profiling tools, vendor FFT/BLAS libraries, and predictable deployment on one vendor family. Mobile proving usually means one of:

- Vulkan compute on Android,
- Metal compute on Apple devices,
- WebGPU in browser or cross-platform client settings.

Those environments are improving, but they are not equivalent to CUDA in library maturity. The [zkSecurity WebGPU article](https://blog.zksecurity.xyz/posts/webgpu/) calls out the lack of a comparable optimized library ecosystem and highlights WGSL language limitations such as no native 64-bit integers in the environment they target.

### 2. Subgroup / Warp Behavior Is Less Stable

CUDA programmers often mentally optimize around NVIDIA warps. In Vulkan, subgroup size is explicitly implementation-dependent. The Vulkan docs say the `SubgroupSize` builtin contains the "implementation-dependent number of invocations in a subgroup", and in some modes that size can vary within supported limits: [Vulkan SubgroupSize](https://docs.vulkan.org/refpages/latest/refpages/source/SubgroupSize.html).

That matters because kernels tuned for a fixed subgroup assumption can lose portability or performance on mobile.

### 3. On-Chip Memory Is Smaller and More Constrained

Apple's Metal feature tables show that mobile devices commonly expose 16 KB or 32 KB of total threadgroup memory, with maximum threads per threadgroup often 512 or 1024 depending on family: [Apple Metal Feature Set Tables](https://developer.apple.com/metal/limits/).

This is one of the core reasons full-column FFT/NTT ownership is hard for large transforms on phones. A 16K-point column simply does not fit in on-chip shared/threadgroup memory. Any phone-friendly prover must therefore use staging, tiling, or a hierarchical decomposition.

### 4. Thermal and Sustained Performance Matter More Than Peak Throughput

Android's own performance guidance emphasizes that "Mobile SoCs and Android have more dynamic performance behaviors than desktops and consoles", including thermal state management and varying clocks: [Android ADPF](https://developer.android.com/games/optimize/adpf). The same page recommends fixed-performance mode for benchmarking because otherwise measurements are perturbed by clock changes.

For ZK, this means:

- a kernel that looks excellent for one second may be poor over one minute,
- memory-heavy workloads can become thermally unsustainable faster than arithmetic-heavy ones,
- benchmark methodology matters as much as micro-optimization.

### 5. Unified Memory Helps, but Does Not Solve Everything

The [zkSecurity WebGPU article](https://blog.zksecurity.xyz/posts/webgpu/) notes that many modern mobile devices use unified memory for CPU and GPU in native builds, reducing transfer cost. That is real and useful. But unified memory does not magically remove all bottlenecks:

- total available memory is still limited,
- CPU and GPU contend for the same DRAM pool,
- synchronization overhead still exists,
- browser or sandboxed environments may not expose the same low-copy behavior.

## Vendor-Specific Mobile GPU Notes

It is dangerous to talk about "the phone GPU" as if Android and iPhone devices formed one performance class. The APIs and hardware limits are too different.

### Apple / Metal

Apple is the cleanest mobile target from a programming-model perspective because Metal is stable and the SoC family is more controlled. But the [Metal limits tables](https://developer.apple.com/metal/limits/) still enforce the same core reality: threadgroup memory is limited, maximum threads per threadgroup is finite, and those limits differ by GPU family. For large NTTs, Apple devices still need tiling and hierarchical decomposition.

### Qualcomm / Adreno

Qualcomm's OpenCL programming guide for Snapdragon/Adreno emphasizes device querying, memory behavior, and launch-geometry sensitivity rather than one-size-fits-all constants: [Qualcomm Snapdragon Mobile Platform OpenCL General Programming and Optimization Guide](https://docs.qualcomm.com/bundle/publicresource/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf). That is consistent with the practical reality on Vulkan as well: query limits, test several workgroup shapes, and do not assume the same tuning point as on desktop CUDA.

### Arm / Mali

Arm's public guidance consistently emphasizes bandwidth sensitivity and best-practice validation for Vulkan on Mali devices: [Arm best-practice warnings in Vulkan SDK](https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-best-practice-warnings-in-vulkan-sdk), [Arm Mali best practices release note](https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-mali-best-practices). Public Mali documentation also highlights the importance of understanding cache locality and the memory system, not only ALU throughput.

### Samsung / Galaxy ecosystem

Samsung's [Vulkan Usage Recommendations](https://developer.samsung.com/galaxy-gamedev/resources/articles/usage.html) are useful because they read like a checklist of issues that also matter for compute-heavy provers:

- create pipelines early,
- cache descriptor/pipeline state,
- avoid heavy synchronization primitives,
- prefer careful runtime capability checks,
- and benchmark across multiple devices because the underlying GPU architecture varies across Galaxy phones.

The key engineering conclusion is simple: mobile ZK needs runtime policy selection. Static kernel geometry hard-coded for one phone family is not a credible long-term strategy.

## What This Means for Scheme Design

The mobile GPU question is not just "how to port kernels?". It is also "which proof systems are structurally compatible with phone-class hardware?".

## Scheme-by-Scheme Optimization Pressure

### Pairing-Based SNARKs: Groth16 / PLONK-ish Systems

These are useful as a contrast case, even though they are not post-quantum.

Dominant kernels often include:

- MSM on elliptic-curve points,
- NTT/FFT over scalar fields,
- witness generation that may or may not map well to GPU.

Strengths on GPUs:

- MSM and NTT are known acceleration targets.
- Server-side CUDA work is already mature enough to show strong wins.

Weaknesses on phones:

- Large-integer and elliptic-curve arithmetic are awkward on mobile GPU APIs, especially in WebGPU/WGSL-like environments with limited integer support.
- Mature mobile ECC/MSM libraries are much less available than server CUDA libraries.
- These systems are not post-quantum, so even strong acceleration does not solve the long-term trust model.

Takeaway: pairing-based systems may still be the easiest way to get "mobile proving that works today", but they are not the main answer if post-quantum transparency is a hard requirement.

### FRI / STARK / Circle-STARK Style Systems

This is the most important mobile class.

Why they are promising:

- transparent setup,
- hash/code-based security story,
- arithmetic can often be engineered around 31-bit or 32-bit-friendly fields,
- workloads like FFT/NTT, constraint evaluation, and Merkle hashing are massively parallel.

Why they are still hard:

- trace and codeword sizes become very large,
- Merkle commitment traffic is bandwidth-heavy,
- the prover performs many passes over large arrays,
- late-stage low-degree tests and query openings are latency-sensitive and do not fully utilize the GPU.

Stwo is especially interesting here because StarkWare explicitly designed it around standard 32-bit computer words for CPU and GPU efficiency: [StarkWare blog](https://starkware.co/blog/starkware-new-proving-record/). That design direction is aligned with phones. But being 32-bit-friendly is not enough by itself; the end-to-end prover still has to manage large memory footprints.

Takeaway: FRI-family provers look like the strongest long-term mobile-GPU candidates, but only if they aggressively control memory movement and do not assume datacenter-class bandwidth.

### STIR and WHIR

These are particularly relevant if the goal is post-quantum, transparent, and possibly more verifier-friendly than classic FRI.

Public cryptographic positioning:

- [STIR](https://eprint.iacr.org/2024/390) improves Reed-Solomon proximity testing query complexity and can reduce argument size compared to optimized FRI.
- [WHIR](https://eprint.iacr.org/2024/1586) is framed as a direct replacement for FRI, STIR, BaseFold, and related protocols, with exceptionally fast verification and strong transparent/post-quantum appeal.

Why they are attractive for mobile:

- smaller verifier cost is good if proofs must also be checked locally,
- smaller proofs help bandwidth and on-chain verification settings,
- transparent setup matches post-quantum goals.

Why they may be harder to optimize on phone GPUs:

- the implementation ecosystem is much younger than for classic FRI,
- the prover-side kernel mix is less standardized,
- richer multilinear/sumcheck-style query structures can create less regular memory access than a plain batched FFT/NTT pipeline,
- protocol improvements that help verifier time do not automatically reduce prover memory pressure,
- many engineering teams still lack stable benchmark baselines for WHIR-like prover kernels on mobile hardware.

The honest current status is that WHIR looks cryptographically compelling, but the mobile GPU engineering playbook for it is much less mature than for classic FFT-heavy STARK pipelines.

The practical difference from classic FRI is worth stating explicitly. FRI has a relatively familiar prover shape for systems engineers:

- evaluate over a larger domain,
- perform regular folding / mixing steps,
- commit with Merkle trees,
- answer sampled queries.

WHIR and STIR improve parts of that story cryptographically, especially proof size and verifier work, but they do not automatically turn prover execution into a more GPU-friendly workload. In fact, newer transparent systems can increase engineering complexity because the prover is no longer "just a big pile of regular NTTs plus hashing". That is good cryptography, but it can be harder hardware.

## What Actually Makes a Post-Quantum Phone Prover Hard

Focusing now specifically on transparent/post-quantum systems:

### Memory, Not Arithmetic, Is Often the Real Enemy

People often look at the raw arithmetic cost of field ops or hash ops. On phones, the larger issue is frequently memory residency:

- expanded execution trace,
- blowup-domain codewords,
- multiple Merkle layers,
- scratch buffers for folding, composition, and query answering.

A protocol can be "arithmetic-light" and still fail on mobile simply because it requires too many large passes over memory.

This is one of the main reasons transparent/post-quantum provers are both promising and difficult on phones:

- promising, because they avoid pairings and can often live in 32-bit arithmetic,
- difficult, because they replace curve-heavy work with very large trace/codeword/hash pipelines.

### Hashes Are Parallel, but Trees Are Bandwidth-Hungry

Merkle hashing is a natural GPU target, but full-tree pipelines have poor arithmetic intensity compared to dense matrix-like kernels. This becomes painful on phones because:

- sustained memory bandwidth is much lower than on server GPUs,
- the kernel mix alternates between wide parallel layers and progressively smaller layers,
- small upper layers underutilize the GPU.

### FFT/NTT Kernels Are Good Fits Only If the Field and Data Layout Behave

Transparent/post-quantum systems benefit from FFT/NTT acceleration when:

- moduli fit well in 32-bit or carefully packed 64-bit arithmetic,
- the data layout avoids global permutations,
- work can be tiled into limited threadgroup/shared memory,
- the host avoids dispatch and synchronization overhead between tiny stages.

This is exactly why Circle-STARK/M31-style approaches are so interesting on commodity hardware.

### Query Phases and Small Kernels Are Latency-Bound

Even if the heavy commitment phase maps well to GPU, the later stages of many protocols do not.

Examples:

- small batched openings,
- verifier-style query checks,
- late reductions,
- control-heavy witness-generation fragments.

These often become hybrid CPU/GPU tasks rather than clean full-GPU pipelines.

## Design Rules for Phone-Friendly ZK Provers

If the target is "runs on a phone and remains post-quantum-transparent", the engineering rules are already becoming clear:

### Prefer 32-Bit-Friendly Arithmetic

This is one of the strongest practical lessons. The more the prover can stay in 32-bit lanes, the easier it is to target Vulkan, Metal, and WebGPU portably and efficiently.

### Keep Data on the GPU for as Long as Possible

The [zkSecurity WebGPU article](https://blog.zksecurity.xyz/posts/webgpu/) is correct to emphasize transfer cost. If multiple proving subcomponents are individually offloaded but each round-trips through the CPU, the gains shrink quickly.

### Use Tiled / Hierarchical FFT and Hash Pipelines

For large transforms, "everything in shared memory" is impossible on phones. The practical design is hierarchical:

- fuse what fits on-chip,
- tile the rest,
- minimize global reshuffles,
- reduce host-driven micro-dispatches where possible.

This is also the lesson that keeps recurring across the GPU FFT/NTT literature. The implementation progression is usually:

- naive per-stage kernel launches,
- stage fusion while locality still holds,
- then a structured decomposition such as `4-step` once the transform becomes too large.

### Build Device-Adaptive Policies

Phones vary too much for one static kernel policy. Workgroup sizes, fusion thresholds, and whether a component should run on GPU at all should depend on:

- memory limits,
- supported subgroup sizes,
- thermal headroom,
- measured sustained performance.

### Benchmark for Sustained Performance, Not Just Peak

Use thermal-aware methodology. Android explicitly recommends fixed-performance mode for benchmarking where available: [ADPF](https://developer.android.com/games/optimize/adpf). Without that, comparing one optimization to another can be misleading.

## Practical Research Directions and Reusable Primitives

The most useful next step for the ecosystem is not one more monolithic prover. It is a reusable set of mobile-friendly GPU primitives plus a benchmarking and autotuning layer that multiple projects can share.

### 1. Start with a Benchmark Matrix

Before choosing one universal algorithm, it makes sense to benchmark several algorithm families across:

- transform sizes from small to prover-scale,
- different widths / batch counts,
- row-major, column-major, and strided layouts,
- 32-bit-friendly fields such as M31 / BabyBear and larger-word fields where relevant,
- three initial target environments: Apple/Metal devices, Android Vulkan devices, and WebGPU-capable browsers.

The goal is to learn where each family wins:

- Stockham / autosort kernels,
- merged-stage kernels,
- radix2-CT kernels,
- `4-step` decompositions for large transforms.

This is the data needed to make dynamic per-device policies real rather than aspirational.

### 2. Provide a Reusable DFT / NTT Primitive Layer

A reusable GPU DFT/NTT primitive layer for different layouts would be one of the most useful ecosystem contributions. It would help any project whose prover reduces large parts of the workload to batched polynomial evaluation/interpolation, including `whir-p3` and FRI-family provers more generally.

To make that practical, the primitive layer should accept at least:

- field descriptor: modulus, reduction method, twiddle-generation rules,
- transform descriptor: size, direction, batch count, in-place vs out-of-place,
- layout descriptor: row-major, column-major, strided, transposed, tiled,
- execution descriptor: preferred residency, temporary-buffer budget, synchronization strategy.

The implementation should not hard-code one kernel family. It should expose multiple backends and choose among them at runtime.

### 3. Use Dynamic Per-Device Policies at the Primitive Level

In practice, "dynamic per-device policies" means that the primitive layer queries device capabilities and then picks an execution plan. The relevant inputs include:

- local/threadgroup memory limits,
- subgroup size behavior,
- maximum workgroup size,
- vendor / architecture family,
- measured bandwidth and sustained throughput from a calibration pass.

Then the runtime can choose:

- Stockham-style kernels for small or medium transforms where autosort behavior is a clear win,
- merged-stage kernels while tiles still fit local memory,
- `4-step` decomposition for larger transforms where locality dominates,
- hybrid CPU/GPU execution for latency-dominated tails.

This is more useful than building one prover-specific kernel path because the same policy layer could be reused across multiple proof systems.

### 4. Add Layout and Transpose Primitives

If the ecosystem wants `4-step` NTTs and better memory locality, then transpose and layout-conversion kernels are first-class primitives, not implementation details.

Useful reusable primitives include:

- tiled matrix transpose,
- row-major <-> column-major conversion,
- strided gather/scatter with bounds-aware tiling,
- batched twiddle multiplication kernels,
- permutation kernels for the cases where autosort is not used.

Without these, every project reimplements the same expensive memory-movement layer.

### 5. Add Hash and Merkle Primitives

Transparent systems need more than DFT/NTT. The other high-value reusable layer is commitment-oriented primitives:

- field-friendly hash kernels,
- leaf hashing over trace/codeword layouts,
- batched parent-layer compression,
- optional layer fusion where locality permits,
- query-path extraction helpers.

A phone-friendly ZK stack needs these to live next to the NTT layer, not in a separate silo.

### 6. Add Folding / Mixing Primitives for Transparent Systems

FRI, WHIR, and STIR-like systems all need some form of folding, mixing, or low-degree-combination step. A reusable library can help by exposing:

- batched linear-combination kernels,
- domain-mixing kernels,
- column-wise and row-wise reduction primitives,
- small-kernel fallbacks for late query/opening phases.

This would help bridge the gap between "FFT library" and "actual transparent prover runtime".

### 7. Add Residency and Execution-Orchestration Primitives

End-to-end GPU residency is important enough that it should be treated as a reusable subsystem, not just an optimization note. A mobile-friendly stack would benefit from:

- GPU buffer residency management for traces, codewords, hashes, and temporary workspaces,
- command-graph or pipeline-executor support for chaining DFT/NTT, folding, and hashing phases without unnecessary host round-trips,
- synchronization-minimizing schedulers that batch dependent stages together,
- explicit CPU/GPU handoff policies for small latency-dominated tails,
- temporary-buffer planners that can trade memory footprint against extra passes.

This is the part that turns a collection of fast kernels into an actual prover backend.

### 8. Add Memory-Aware Primitives for Transparent Provers

Transparent systems need more than fast kernels; they need memory-aware execution under explicit device limits. Useful reusable components here would include:

- trace/codeword layout planners that choose chunking and tiling based on memory budget,
- streaming commitment builders for traces too large to process in one resident pass,
- memory-budget-aware query batching and opening schedulers,
- chunked folding and composition kernels with explicit scratch-space bounds,
- protocol-parameter planners that can estimate whether a chosen blowup, width, or batching strategy fits on a target device.

This is the point where "memory-aware protocol choices" becomes something implementable rather than just a design principle.

### 9. Help the Ecosystem with Open Benchmarking and Capability Data

Another concrete contribution would be a public benchmark suite for phone-class GPUs. That suite should report:

- transform throughput and latency by size,
- hash / Merkle throughput,
- transpose bandwidth,
- thermal decay curves,
- device capability snapshots,
- chosen runtime policy and why it was chosen.

Right now, the public data is too thin. Better shared measurements would help every project.

### 10. Likely Algorithmic Direction

There is no reason to bet everything on one algorithm family. A realistic reusable stack would likely look like this:

- Stockham-style kernels for permutation-avoiding small and medium transforms,
- merged-stage fusion where local memory still holds the tile,
- `4-step` decomposition for large transforms,
- explicit transpose/layout primitives to support that decomposition,
- autotuned runtime selection rather than a compile-time constant plan.

## Bottom Line

Phone GPU proving is real, but the winning designs will not be direct ports of server-CUDA provers.

The best mobile candidates, especially under post-quantum / transparent constraints, are likely to be systems whose heavy phases are dominated by:

- 32-bit-friendly field arithmetic,
- regular batched FFT/NTT-like passes,
- hash pipelines that can be staged hierarchically,
- small enough proof systems or memory layouts to fit within mobile DRAM and thermal limits.

That points strongly toward carefully engineered FRI-family and related transparent systems, including Circle-STARK-style provers and possibly WHIR-family descendants once their implementation ecosystem matures.

## Sources

- Microsoft Research, *Fast Computation of General Fourier Transforms on GPUs*: https://www.microsoft.com/en-us/research/publication/fast-computation-of-general-fourier-transforms-on-gpus/
- Microsoft Research, *High Performance Discrete Fourier Transforms on Graphics Processors*: https://www.microsoft.com/en-us/research/publication/high-performance-discrete-fourier-transforms-on-graphics-processors/
- Microsoft Research, *Auto-tuning of Fast Fourier Transform on Graphics Processors*: https://www.microsoft.com/en-us/research/publication/auto-tuning-of-fast-fourier-transform-on-graphics-processors/
- Ozcan and Savas, *Two Algorithms for Fast GPU Implementation of NTT*: https://eprint.iacr.org/2023/1410
- Ozcan et al., *High-Performance NTT on GPU Through radix2-CT and 4-Step Algorithms*: https://eprint.iacr.org/2025/388
- GPU-NTT repository: https://github.com/Alisah-Ozcan/GPU-NTT
- Ozcan et al., *GPU Acceleration of Approximate Homomorphic Encryption*: https://eprint.iacr.org/2021/124
- Ozcan et al., *Homomorphic Encryption on GPU*: https://eprint.iacr.org/2022/1222
- David H. Bailey, *FFTs in external or hierarchical memory*: https://ntrs.nasa.gov/citations/19900047338
- SMFFT repository: https://github.com/KAdamek/SMFFT
- VkFFT repository: https://github.com/DTolm/VkFFT
- VkFFT API guide: https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/documentation/VkFFT_API_guide.tex#L458
- VkFFT paper landing page: https://doi.org/10.1109/access.2023.3242240
- rocFFT runtime compilation design doc: https://rocm.docs.amd.com/projects/rocFFT/en/develop/design/runtime_compilation.html
- rocFFT codegen design doc: https://rocm.docs.amd.com/projects/rocFFT/en/docs-5.1.0/design/codegen.html
- cuFFT documentation: https://docs.nvidia.com/cuda/cufft/
- ICICLE repository: https://github.com/ingonyama-zk/icicle
- ICICLE GPU backends docs: https://dev.ingonyama.com/start/architecture/install_gpu_backend
- ICICLE architecture overview: https://dev.ingonyama.com/3.6.0/icicle/arch_overview
- ICICLE-Stwo announcement: https://www.ingonyama.com/post/introducing-icicle-stwo-a-gpu-accelerated-stwo-prover
- IMP1 mobile prover repository: https://github.com/ingonyama-zk/imp1
- IMP1 mobile post: https://www.ingonyama.com/post/imp1-bringing-zero-knowledge-proofs-to-mobile
- PocketProof: https://www.ingonyama.com/pocketproof
- Air-FRI thesis entry: https://uwspace.uwaterloo.ca/items/6cee69e5-c9cf-4ef2-ba59-ca3541b58f71
- WHIR paper: https://eprint.iacr.org/2024/1586
- STIR paper: https://eprint.iacr.org/2024/390
- BaseFold paper: https://eprint.iacr.org/2023/1705
- DeepFold paper: https://eprint.iacr.org/2024/1595
- PipeZK: https://www.microsoft.com/en-us/research/publication/pipezk-accelerating-zero-knowledge-proof-with-a-pipelined-architecture/
- cuZK repository / paper entry: https://github.com/speakspeak/cuZK
- cuZK ePrint entry: https://ia.cr/2022/1321
- ZKPoG paper: https://eprint.iacr.org/2025/765
- StarkWare Stwo record post: https://starkware.co/blog/starkware-new-proving-record/
- Vulkan subgroup docs: https://docs.vulkan.org/refpages/latest/refpages/source/SubgroupSize.html
- Apple Metal feature limits: https://developer.apple.com/metal/limits/
- Qualcomm Snapdragon Mobile Platform OpenCL General Programming and Optimization Guide: https://docs.qualcomm.com/bundle/publicresource/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf
- Arm best-practice warnings in Vulkan SDK: https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-best-practice-warnings-in-vulkan-sdk
- Arm Mali best practices release note: https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/arm-mali-best-practices
- Samsung Vulkan Usage Recommendations: https://developer.samsung.com/galaxy-gamedev/resources/articles/usage.html
- Android Dynamic Performance Framework: https://developer.android.com/games/optimize/adpf
- zkSecurity WebGPU/Stwo post: https://blog.zksecurity.xyz/posts/webgpu/
