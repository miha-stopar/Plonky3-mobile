# Mobile GPU Optimizations for SNARKs and STARKs

Status: living document  
Last updated: 2026-03-13

This document is a research-and-engineering survey focused on zero-knowledge proving on phone-class GPUs: Android Vulkan/WebGPU devices, Apple Metal devices, and browser-exposed mobile GPUs.

It is written from a prover-engineering perspective rather than a purely cryptographic one. The main question is not only "what is asymptotically good?", but also "what can actually run on a phone without melting the device or exhausting memory?".

## Executive Summary

Phone GPU proving is still much less explored than datacenter GPU proving. The public ZK acceleration ecosystem is still dominated by CUDA-first work such as [ICICLE](https://github.com/ingonyama-zk/icicle), [cuZK](https://github.com/speakspeak/cuZK), [PipeZK](https://www.microsoft.com/en-us/research/publication/pipezk-accelerating-zero-knowledge-proof-with-a-pipelined-architecture/), and newer end-to-end platforms like [ZKPoG](https://eprint.iacr.org/2025/765). By contrast, public mobile-oriented proving efforts are relatively recent and comparatively sparse: [IMP1](https://github.com/ingonyama-zk/imp1), [PocketProof](https://www.ingonyama.com/pocketproof), [Mopro](https://github.com/zkmopro/mopro), [ICICLE Metal backend docs](https://dev.ingonyama.com/start/architecture/install_gpu_backend), and practitioner reports such as [zkSecurity's WebGPU/Stwo write-up](https://blog.zksecurity.xyz/posts/webgpu/).

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

### Public Evidence by Library

The table below is deliberately careful about what is directly stated by public sources versus what is engineering inference.

| Library | GPU vendors / APIs | What public sources indicate | Source quality |
|---|---|---|---|
| [VkFFT](https://github.com/DTolm/VkFFT) | NVIDIA / AMD / Intel / Apple via Vulkan, CUDA, HIP, OpenCL, Level Zero, Metal | Public docs emphasize in-place operation, minimized transpositions, and four-step reshuffling only for large sequences. In practice VkFFT is commonly grouped with Stockham-oriented GPU FFT designs, but I did not find a single explicit official sentence in the checked sources saying "VkFFT uses Stockham". | Medium; partly inference |
| [SMFFT](https://github.com/KAdamek/SMFFT) | NVIDIA CUDA | Explicitly includes both Cooley-Tukey and Stockham implementations; the Stockham implementation is described as autosort and producing correctly ordered output without an explicit reorder step. | High |
| [Microsoft GPU FFT research](https://www.microsoft.com/en-us/research/publication/fast-computation-of-general-fourier-transforms-on-gpus/) | Early GPU work, NVIDIA-focused experiments | Explicitly states radix-2 Stockham as the basic building block; later Microsoft work also highlights hierarchical FFTs using Stockham formulations. | High |
| [rocFFT](https://rocm.docs.amd.com/projects/rocFFT/en/develop/design/runtime_compilation.html) | AMD ROCm / HIP | AMD's runtime compilation design docs say Stockham FFT kernels make up the vast majority of the library; codegen docs also discuss tiled/strided/batched Stockham kernels. | High |
| [cuFFT](https://docs.nvidia.com/cuda/cufft/) | NVIDIA CUDA | Official docs specify Cooley-Tukey building blocks for favorable sizes and Bluestein for difficult sizes, but do not fully document internal kernel families. It is safest to describe cuFFT publicly as mixed-radix / mixed-strategy rather than purely Stockham. | High for "mixed"; low for deeper internals |

### Why This Matters for ZK

For ZK proving, FFT/NTT passes often dominate polynomial evaluation/interpolation work. If every pass also forces extra global permutation traffic, mobile GPUs suffer twice:

- limited bandwidth relative to server GPUs,
- stricter thermal envelopes that punish sustained memory-heavy kernels.

That is why the Stockham-style lesson transfers directly into phone-class ZK implementations: avoiding standalone global permutations is often worth more than shaving a few arithmetic instructions.

## Ozcan-Savas NTT Paper: What It Actually Contributes

The paper [Two Algorithms for Fast GPU Implementation of NTT](https://eprint.iacr.org/2023/1410) is worth reading carefully because it is much closer to the engineering problems in a phone prover than many protocol papers are. Its abstract is unusually clear about the real bottleneck: NTT is often memory-bound, so the decisive question is not only "how many butterfly operations?" but "how much slow global memory traffic and synchronization do we force?".

The paper proposes two GPU-oriented NTT families:

- `Merge-NTT`: reduce the number of GPU kernel launches and global-memory synchronization points by merging multiple sequential NTT stages into one kernel when the memory hierarchy allows it.
- `4-Step NTT`: restructure the transform so that the memory access pattern becomes more local and spatially coherent, following the classical 4-step FFT idea used on hierarchical-memory machines.

### 1. Merge-NTT

The paper's first algorithm starts from the familiar iterative radix-2 NTT structure: there are `log2(n)` outer loop iterations, and each outer loop corresponds to one stage width. A naive GPU mapping launches one kernel per outer loop iteration. That is simple, but it means:

- a fresh kernel launch per stage,
- a global-memory round-trip between stages,
- and a full device-level synchronization point between stages.

Ozcan and Savas attack exactly that cost. Their `Merge-NTT` idea is to fuse several consecutive outer-loop iterations into one GPU kernel whenever the working set still fits the fast levels of the memory hierarchy. The paper's abstract says the implementation is highly sensitive to CUDA parameters such as kernel count, block size, and block shape, and that one of its contributions is a recipe for selecting them for a given polynomial degree.

Conceptually, this is the same systems idea as the fused stage-window kernels we have already been experimenting with in this repo:

- keep data on chip for as long as possible,
- do several dependent butterfly stages before writing back,
- and pay the host/device synchronization cost less often.

This is not a cryptographic trick. It is a memory-traffic trick.

### 2. 4-Step NTT

The second algorithm is the more structurally important one.

The paper says it adopts the `4-Step` method to improve spatial locality of global memory access. The historical idea comes from David Bailey's [FFTs in external or hierarchical memory](https://ntrs.nasa.gov/citations/19900047338), which treats a long transform as a matrix, performs smaller transforms on one dimension, multiplies by twiddle factors, and then transforms along the other dimension. The point of the method is not to change the math. The point is to turn long-stride memory access into more regular, contiguous access that better fits hierarchical memory systems.

The 4-step pattern is:

1. Reshape the length-`n` vector into a 2D matrix.
2. Compute many shorter transforms on one matrix dimension.
3. Multiply by the corresponding twiddle factors.
4. Transpose / reorder so the next short transforms are again memory-friendly.
5. Compute the transforms on the other dimension.
6. Transpose back if needed for the expected output layout.

Why this matters on GPU:

- global memory likes regular, coalesced access,
- shared memory / threadgroup memory is too small for the full transform,
- and short local transforms are much easier to schedule well than one monolithic large transform with terrible strides.

This is exactly the kind of idea that becomes more relevant on phones, not less, because mobile GPUs are even more sensitive to bandwidth and locality than server GPUs.

### 3. What Ideas It Inherits From the References It Cites

The most important referenced ideas behind the paper are these:

- `Bailey 4-step FFT`: use matrix decomposition and transposes to match a hierarchical memory system instead of forcing a huge transform to fight the memory hierarchy. Bailey's paper explicitly emphasizes unit-stride transfers and minimizing passes over external memory.
- `Earlier GPU NTT for HE`: the Sabanci line of work around [GPU Acceleration of Approximate Homomorphic Encryption](https://eprint.iacr.org/2021/124) and [Homomorphic Encryption on GPU](https://eprint.iacr.org/2022/1222) focuses on minimizing kernel calls, using the GPU memory hierarchy efficiently, and tuning launch geometry for NTT-heavy workloads.
- `GPU FFT literature`: the same memory-locality story appears in older Microsoft FFT work and in modern GPU FFT libraries. Even when the algorithm family differs, the same implementation rule keeps recurring: reduce scattered global memory traffic and structure the transform around what fast local memory can hold.

So the paper is not presenting two isolated tricks. It is combining three longstanding HPC ideas:

- fuse dependent stages when locality allows,
- decompose large transforms into memory-friendly subtransforms,
- and tune launch geometry as a first-class part of the algorithm.

### 4. What This Means for Phone GPUs

For a phone prover, the paper suggests three practical directions.

First, stage fusion is worth doing, but only while the tile still fits the local memory budget. That is the same limit we keep hitting with full-column ownership for large `h`.

Second, once the transform is too large for simple fused tiling, a `4-step` style decomposition is the natural next move. It is a better fit than just "keep launching one more stage kernel" because it actively repairs locality.

Third, the launch parameters are part of the algorithm. The paper is explicit that the optimum depends on degree and hardware. That is especially true on mobile, where subgroup size, local-memory limits, and thermal behavior vary significantly by vendor.

## The Current Ecosystem: Why Phone GPU ZK Still Looks Underexplored

The public ecosystem today is still heavily skewed toward server-side NVIDIA GPUs.

### GPU ZK Work That Is Mostly Server / CUDA First

- [ICICLE](https://github.com/ingonyama-zk/icicle): hardware acceleration library for ZK primitives across multiple backends, but public open source remains strongest on the CPU/frontend side while CUDA acceleration has historically been the main proving story.
- [cuZK](https://github.com/speakspeak/cuZK): GPU implementation of zkSNARK proving with emphasis on MSM and NTT.
- [PipeZK](https://www.microsoft.com/en-us/research/publication/pipezk-accelerating-zero-knowledge-proof-with-a-pipelined-architecture/): hardware-architecture paper, important because it frames ZK acceleration as a pipeline and memory-traffic problem, not just a raw-compute problem.
- [ZKPoG](https://eprint.iacr.org/2025/765): end-to-end GPU acceleration including witness generation for Plonkish pipelines.
- [ICICLE-Stwo](https://www.ingonyama.com/post/introducing-icicle-stwo-a-gpu-accelerated-stwo-prover): GPU acceleration of StarkWare's Stwo prover, reporting 3.25x to 7x over the SIMD backend.

### Mobile / Client-Side Efforts That Are Public

- [IMP1](https://github.com/ingonyama-zk/imp1): explicitly a mobile-first proving framework for iOS and Android, described as a mobile-optimized Groth16 prover. The README claims up to 3x faster than Rapidsnark on supported devices.
- [PocketProof](https://www.ingonyama.com/pocketproof): packaging and distribution layer for the same mobile proving direction.
- [Mopro](https://github.com/zkmopro/mopro): mobile tooling and bindings rather than a single GPU backend, but strategically important because mobile ZK adoption depends on integration tooling, not just kernels.
- [ICICLE Metal backend docs](https://dev.ingonyama.com/start/architecture/install_gpu_backend) and [architecture overview](https://dev.ingonyama.com/3.6.0/icicle/arch_overview): shows serious movement beyond CUDA, at least for Apple devices.
- [zkSecurity's WebGPU/Stwo article](https://blog.zksecurity.xyz/posts/webgpu/): one of the clearest public practitioner write-ups on mobile-compatible GPU proving via WebGPU, including real engineering constraints.
- [QED's plonky2-wasm](https://github.com/QEDProtocol/plonky2-wasm): browser/client-side experimentation with WebGPU-oriented proving.
- [msm-webgpu](https://lib.rs/crates/msm-webgpu): WebGPU MSM work that points toward browser/mobile-compatible ECC acceleration, though still far from a mature proving stack.

### Academic and Practitioner Work Worth Tracking

- [Air-FRI: Acceleration of the FRI Protocol on the GPU for zkSNARK Applications](https://uwspace.uwaterloo.ca/items/6cee69e5-c9cf-4ef2-ba59-ca3541b58f71): useful because it treats FRI acceleration as a concrete GPU systems problem rather than a vague "GPU helps hashing" statement. Even though the thesis is not phone-specific, it is directly relevant to mobile because the same phases dominate: polynomial evaluation, FFT-like work, and commitment/query pipelines.
- [zkSecurity's WebGPU/Stwo article](https://blog.zksecurity.xyz/posts/webgpu/): one of the most concrete public descriptions of getting a transparent prover onto a browser/mobile-compatible GPU stack.

Air-FRI is particularly important in this survey because it shows the right way to think about transparent-proof acceleration. The point is not merely that a GPU can evaluate some arithmetic faster than a CPU. The point is that FRI-like provers are a pipeline of domain evaluation, mixing/folding, and commitment work whose performance depends heavily on memory traffic and kernel orchestration. That framing transfers directly to phone GPUs, where memory movement is even more decisive.

### Why It Still Feels Underexplored

There are several structural reasons:

- CUDA gave the ecosystem a fast path to impressive speedups on commodity datacenter GPUs, so most teams optimized there first.
- Mobile hardware is fragmented across Vulkan, Metal, browser WebGPU, and vendor-specific quirks.
- Phones are thermally constrained; sustained throughput matters more than short benchmark bursts.
- Public benchmarking and regression infrastructure across many phone models is still weak.
- Memory budgets are much tighter, especially for transparent/post-quantum protocols with large traces or codewords.
- The market incentive has historically been stronger for rollup sequencers and proving services than for client-side proving.

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

### STIR, WHIR, and Whirlaway

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

[Whirlaway](https://github.com/TomWambsgans/Whirlaway/blob/master/Whirlaway.pdf) is relevant here as a concrete WHIR-adjacent system rather than a vague placeholder name. Based on the PDF abstract, it is a transparent, post-quantum zero-knowledge argument system that combines Fractal-style folding with DEEP-ALI instead of standard FRI, while aiming to preserve Plonkish flexibility and improve proof-size / verifier-cost tradeoffs. For this survey, the practical importance of Whirlaway is not that it changes the mobile GPU answer completely, but that it reinforces the broader trend: newer transparent systems are trying to reduce proof and verification cost, while prover engineering still has to deal with large codewords, memory movement, and irregular folding/query phases.

The practical difference from classic FRI is worth stating explicitly. FRI has a relatively familiar prover shape for systems engineers:

- evaluate over a larger domain,
- perform regular folding / mixing steps,
- commit with Merkle trees,
- answer sampled queries.

WHIR and STIR improve parts of that story cryptographically, especially proof size and verifier work, but they do not automatically turn prover execution into a more GPU-friendly workload. In fact, newer transparent systems can increase engineering complexity because the prover is no longer "just a big pile of regular NTTs plus hashing". That is good cryptography, but it can be harder hardware.

### BaseFold, DeepFold, and Other Fold/Code-Based Successors

Representative references:

- [BaseFold](https://eprint.iacr.org/2023/1705)
- [DeepFold](https://eprint.iacr.org/2024/1595)

These schemes are relevant because they aim to reduce proof size, improve field flexibility, or improve prover/verifier tradeoffs relative to earlier FRI-style systems.

Potential mobile upside:

- moving away from strict FFT-friendly-field assumptions can be valuable in some protocol stacks,
- multilinear commitments can sometimes align better with applications built around Boolean structure.

Potential mobile downside:

- the prover kernel mix may shift away from the most GPU-familiar batched NTT pattern,
- code encodings, foldable-code operations, and list-decoding-adjacent analyses may come with more complex data movement,
- implementation maturity is still lower than "standard FFT plus Merkle" prover pipelines.

Takeaway: these schemes are important research directions, but mobile GPU engineering for them is still less settled than for classic STARK code paths.

### A Note on Whirlaway

Earlier drafts of this survey used the misspelling `Whiraway`. The intended reference is [Whirlaway](https://github.com/TomWambsgans/Whirlaway/blob/master/Whirlaway.pdf), which has a public repository and PDF write-up. In this document it is treated as part of the same practical discussion space as WHIR and other newer transparent/post-quantum systems: attractive verifier-side goals, but still an open question how to make prover kernels mobile-GPU-friendly end to end.

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

This is also the most direct lesson from the Ozcan-Savas paper. The implementation progression is usually:

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

## Practical Research Directions

The highest-value directions for mobile/post-quantum ZK optimization appear to be:

1. End-to-end GPU residency for FRI-family provers, not just isolated FFT kernels.
2. Stockham-style or similarly permutation-avoiding NTT/FFT kernels tuned for 32-bit fields.
3. Merkle/hash pipelines that reduce global memory traffic, possibly via layer fusion or alternative commitment layouts.
4. Hybrid schedulers that assign large regular kernels to GPU and latency-sensitive tails to CPU.
5. Runtime per-device tuning for workgroup shape, stage fusion, and memory strategy.
6. Better open benchmarking suites across many phones, because the field is currently data-poor.
7. More public implementations of WHIR/STIR/Whirlaway-like provers, so that mobile engineering can catch up to protocol progress.

## Implications for This Repo

The current `Plonky3-android` Vulkan path already contains the beginning of the right design direction, but not the full hierarchical step yet.

Relevant files:

- host fused-stage policy: [../native/src/backend_vulkan.rs](../native/src/backend_vulkan.rs)
- current fused shader: [../native/shaders/fft_stage_fused.wgsl](../native/shaders/fft_stage_fused.wgsl)

### What It Already Does

The current host code computes a dynamic fused stage span in [`fused_stage_span`](../native/src/backend_vulkan.rs), capped by a `256`-row tile and a maximum fused stage. The current fused shader loads one `TILE_ROWS` chunk of a column into workgroup memory, performs several stages there, and writes the tile back once.

That is already close in spirit to `Merge-NTT`:

- fewer host-driven stage dispatches,
- fewer global-memory round-trips for early stages,
- and explicit dependence on tile size and workgroup geometry.

### What It Does Not Yet Do

It does not yet implement the more structural `4-step` style decomposition.

Right now, the overall large transform still looks like:

- host drives stage windows,
- each workgroup owns a row-tile slice of several columns,
- later stages fall back to broader global-memory behavior.

What a genuine `4-step` move would change is the layout of the large transform itself. Instead of only asking "how many consecutive stages can this tile execute?", the code would ask:

- how should the big transform be factored into a matrix,
- which short transforms should be computed first,
- where should twiddle multiplication happen,
- and when should we transpose so the next pass is again locality-friendly?

That is a larger rewrite than the current fusion work, but it is the most technically justified next step if the goal is to move beyond stage-window fusion and toward the hierarchical-memory approach the literature recommends.

## Bottom Line

Phone GPU proving is real, but the winning designs will not be direct ports of server-CUDA provers.

The best mobile candidates, especially under post-quantum / transparent constraints, are likely to be systems whose heavy phases are dominated by:

- 32-bit-friendly field arithmetic,
- regular batched FFT/NTT-like passes,
- hash pipelines that can be staged hierarchically,
- small enough proof systems or memory layouts to fit within mobile DRAM and thermal limits.

That points strongly toward carefully engineered FRI-family and related transparent systems, including Circle-STARK-style provers and possibly WHIR-family / Whirlaway-like descendants once their implementation ecosystem matures.

## Sources

- Microsoft Research, *Fast Computation of General Fourier Transforms on GPUs*: https://www.microsoft.com/en-us/research/publication/fast-computation-of-general-fourier-transforms-on-gpus/
- Microsoft Research, *High Performance Discrete Fourier Transforms on Graphics Processors*: https://www.microsoft.com/en-us/research/publication/high-performance-discrete-fourier-transforms-on-graphics-processors/
- Microsoft Research, *Auto-tuning of Fast Fourier Transform on Graphics Processors*: https://www.microsoft.com/en-us/research/publication/auto-tuning-of-fast-fourier-transform-on-graphics-processors/
- Ozcan and Savas, *Two Algorithms for Fast GPU Implementation of NTT*: https://eprint.iacr.org/2023/1410
- GPU-NTT repository: https://github.com/Alisah-Ozcan/GPU-NTT
- Ozcan et al., *GPU Acceleration of Approximate Homomorphic Encryption*: https://eprint.iacr.org/2021/124
- Ozcan et al., *Homomorphic Encryption on GPU*: https://eprint.iacr.org/2022/1222
- David H. Bailey, *FFTs in external or hierarchical memory*: https://ntrs.nasa.gov/citations/19900047338
- SMFFT repository: https://github.com/KAdamek/SMFFT
- VkFFT repository: https://github.com/DTolm/VkFFT
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
- Mopro repository: https://github.com/zkmopro/mopro
- Mopro docs: https://zkmopro.org/docs/intro
- PSE client-side proving page: https://pse.dev/projects/client-side-proving
- Air-FRI thesis entry: https://uwspace.uwaterloo.ca/items/6cee69e5-c9cf-4ef2-ba59-ca3541b58f71
- WHIR paper: https://eprint.iacr.org/2024/1586
- Whirlaway PDF: https://github.com/TomWambsgans/Whirlaway/blob/master/Whirlaway.pdf
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
