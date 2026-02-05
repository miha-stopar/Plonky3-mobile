use p3_dft::Radix2DitParallel;
use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;

pub fn dft_batch<F: TwoAdicField>(
    _cpu: &Radix2DitParallel<F>,
    _mat: RowMajorMatrix<F>,
) -> Result<RowMajorMatrix<F>, String> {
    Err("vulkan backend not implemented".to_string())
}
