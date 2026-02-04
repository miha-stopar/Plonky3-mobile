#[cfg(feature = "plonky3")]
use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear};
#[cfg(feature = "plonky3")]
use p3_symmetric::Permutation;

const WIDTH: usize = 16;

pub fn poseidon2_or_fallback(input: &[i32]) -> Vec<i32> {
    #[cfg(feature = "plonky3")]
    {
        if input.len() == WIDTH {
            let mut state = [BabyBear::ZERO; WIDTH];
            for (idx, value) in input.iter().enumerate() {
                state[idx] = BabyBear::from_u32(*value as u32);
            }

            let perm = default_babybear_poseidon2_16();
            perm.permute_mut(&mut state);

            return state
                .map(|value| value.as_canonical_u64() as i32)
                .to_vec();
        }
    }

    input.iter().map(|value| value.wrapping_add(1)).collect()
}
