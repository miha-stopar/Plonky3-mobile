pub fn fallback(input: &[i32]) -> Vec<i32> {
    // Placeholder until GPU Poseidon2 permutation is wired in.
    input.iter().map(|value| value.wrapping_add(1)).collect()
}
