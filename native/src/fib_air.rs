use core::borrow::Borrow;
use std::time::Instant;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::BabyBear;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use crate::gpu_dft::{BackendKind, GpuDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_fri::{HidingFriPcs, create_test_fri_params};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeHidingMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};
use p3_dft::TwoAdicSubgroupDft;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

const NUM_FIBONACCI_COLS: usize = 2;

pub fn run_fib_air_zk() -> Result<String, String> {
    type ByteHash = Keccak256Hash;
    let byte_hash = ByteHash {};

    type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    let u64_hash = U64Hash::new(KeccakF {});

    type FieldHash = SerializingHasher<U64Hash>;
    let field_hash = FieldHash::new(u64_hash);

    type MyCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;
    let compress = MyCompress::new(u64_hash);

    type ValHidingMmcs = MerkleTreeHidingMmcs<
        [Val; p3_keccak::VECTOR_LEN],
        [u64; p3_keccak::VECTOR_LEN],
        FieldHash,
        MyCompress,
        SmallRng,
        4,
        4,
    >;

    let rng = SmallRng::seed_from_u64(1);
    let val_mmcs = ValHidingMmcs::new(field_hash, compress, rng);

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    type ChallengeHidingMmcs = ExtensionMmcs<Val, Challenge, ValHidingMmcs>;

    let n = 1 << 3;
    let x = 21;

    let challenge_mmcs = ChallengeHidingMmcs::new(val_mmcs.clone());
    let dft = GpuDft::<Val>::with_backend(BackendKind::Vulkan);
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let fri_params = create_test_fri_params(challenge_mmcs, 2);
    type HidingPcs = HidingFriPcs<Val, GpuDft<Val>, ValHidingMmcs, ChallengeHidingMmcs, SmallRng>;
    type MyHidingConfig = StarkConfig<HidingPcs, Challenge, Challenger>;
    let pcs = HidingPcs::new(dft, val_mmcs, fri_params, 4, SmallRng::seed_from_u64(1));
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    let config = MyHidingConfig::new(pcs, challenger);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    let proof = prove(&config, &FibonacciAir {}, trace, &pis);
    verify(&config, &FibonacciAir {}, &proof, &pis)
        .map_err(|err| format!("{err:?}"))?;

    Ok(format!("fib_air zk ok (n={n}, x={x})"))
}

fn benchmark_input(height: usize, width: usize) -> RowMajorMatrix<Val> {
    let values = (0..(height * width))
        .map(|i| {
            // Deterministic, non-trivial values in field range.
            let v = ((i as u64).wrapping_mul(17).wrapping_add(3)) % 0x7800_0001;
            Val::from_u64(v)
        })
        .collect();
    RowMajorMatrix::new(values, width)
}

fn percentile_ms(mut samples: Vec<f64>, q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = samples.len();
    let idx = ((q * n as f64).ceil() as usize).saturating_sub(1).min(n - 1);
    samples[idx]
}

pub fn run_dft_benchmark() -> Result<String, String> {
    crate::backend_vulkan::is_vulkan_available()?;

    let cpu = GpuDft::<Val>::with_backend(BackendKind::Cpu);
    let vulkan = GpuDft::<Val>::with_backend(BackendKind::Vulkan);
    let cases = [
        (256usize, 8usize),
        (1024, 8),
        (4096, 8),
        (16384, 8),
        (4096, 32),
        (16384, 32),
    ];
    let warmup = 1usize;
    let repeats = 10usize;

    let mut lines = vec![format!(
        "dft benchmark (repeats={repeats}, warmup={warmup}, stats=avg/median/p95)"
    )];

    for &(height, width) in &cases {
        let input = benchmark_input(height, width);
        let _ = crate::gpu_dft::take_last_vulkan_error();
        for _ in 0..warmup {
            let _ = cpu.dft_batch(input.clone());
            let _ = vulkan.dft_batch(input.clone());
        }

        let mut cpu_samples_ms = Vec::with_capacity(repeats);
        let mut cpu_out = None;
        for _ in 0..repeats {
            let start = Instant::now();
            cpu_out = Some(cpu.dft_batch(input.clone()));
            cpu_samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let cpu_avg_ms = cpu_samples_ms.iter().copied().sum::<f64>() / repeats as f64;
        let cpu_med_ms = percentile_ms(cpu_samples_ms.clone(), 0.50);
        let cpu_p95_ms = percentile_ms(cpu_samples_ms, 0.95);

        let mut vk_samples_ms = Vec::with_capacity(repeats);
        let mut vk_out = None;
        for _ in 0..repeats {
            let start = Instant::now();
            vk_out = Some(vulkan.dft_batch(input.clone()));
            vk_samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let vk_avg_ms = vk_samples_ms.iter().copied().sum::<f64>() / repeats as f64;
        let vk_med_ms = percentile_ms(vk_samples_ms.clone(), 0.50);
        let vk_p95_ms = percentile_ms(vk_samples_ms, 0.95);

        if let Some(err) = crate::gpu_dft::take_last_vulkan_error() {
            return Err(format!(
                "vulkan benchmark fallback at h={height}, w={width}: {err}"
            ));
        }

        let cpu_out = cpu_out.ok_or_else(|| "cpu benchmark output missing".to_string())?;
        let vk_out = vk_out.ok_or_else(|| "vulkan benchmark output missing".to_string())?;
        if cpu_out.values != vk_out.values {
            return Err(format!("dft benchmark mismatch at h={height}, w={width}"));
        }

        let speedup_avg = if vk_avg_ms > 0.0 {
            cpu_avg_ms / vk_avg_ms
        } else {
            0.0
        };
        lines.push(format!(
            "h={height}, w={width}: cpu(avg={cpu_avg_ms:.3} med={cpu_med_ms:.3} p95={cpu_p95_ms:.3})ms \
vk(avg={vk_avg_ms:.3} med={vk_med_ms:.3} p95={vk_p95_ms:.3})ms speedup(avg)={speedup_avg:.2}x"
        ));
    }

    Ok(lines.join("\n"))
}

pub struct FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (
            main.row_slice(0).expect("Matrix is empty?"),
            main.row_slice(1).expect("Matrix only has 1 row?"),
        );
        let local: &FibonacciRow<AB::Var> = (*local).borrow();
        let next: &FibonacciRow<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.left.clone(), a);
        when_first_row.assert_eq(local.right.clone(), b);

        let mut when_transition = builder.when_transition();

        // a' <- b
        when_transition.assert_eq(local.right.clone(), next.left.clone());

        // b' <- a + b
        when_transition.assert_eq(local.left.clone() + local.right.clone(), next.right.clone());

        builder.when_last_row().assert_eq(local.right.clone(), x);
    }
}

pub fn generate_trace_rows<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace = RowMajorMatrix::new(F::zero_vec(n * NUM_FIBONACCI_COLS), NUM_FIBONACCI_COLS);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<FibonacciRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = FibonacciRow::new(F::from_u64(a), F::from_u64(b));

    for i in 1..n {
        rows[i].left = rows[i - 1].right;
        rows[i].right = rows[i - 1].left + rows[i - 1].right;
    }

    trace
}

pub struct FibonacciRow<F> {
    pub left: F,
    pub right: F,
}

impl<F> FibonacciRow<F> {
    const fn new(left: F, right: F) -> Self {
        Self { left, right }
    }
}

impl<F> Borrow<FibonacciRow<F>> for [F] {
    fn borrow(&self) -> &FibonacciRow<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibonacciRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}
