use core::sync::atomic::{AtomicU8, Ordering};

use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_dft::Radix2DitParallel;

use crate::backend_metal;
use crate::backend_vulkan;
use crate::backend_webgpu;

#[derive(Clone, Copy, Debug)]
pub enum BackendKind {
    Cpu,
    Vulkan,
    Metal,
    WebGpu,
}

impl BackendKind {
    fn to_u8(self) -> u8 {
        match self {
            BackendKind::Cpu => 0,
            BackendKind::Vulkan => 1,
            BackendKind::Metal => 2,
            BackendKind::WebGpu => 3,
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            1 => BackendKind::Vulkan,
            2 => BackendKind::Metal,
            3 => BackendKind::WebGpu,
            _ => BackendKind::Cpu,
        }
    }
}

static BACKEND_KIND: AtomicU8 = AtomicU8::new(0);

pub fn set_backend_kind(kind: BackendKind) {
    BACKEND_KIND.store(kind.to_u8(), Ordering::Relaxed);
}

pub fn get_backend_kind() -> BackendKind {
    BackendKind::from_u8(BACKEND_KIND.load(Ordering::Relaxed))
}

pub fn set_backend_kind_from_str(value: &str) -> Result<(), String> {
    let kind = match value.to_ascii_lowercase().as_str() {
        "cpu" => BackendKind::Cpu,
        "vulkan" => BackendKind::Vulkan,
        "metal" => BackendKind::Metal,
        "webgpu" => BackendKind::WebGpu,
        other => return Err(format!("unknown backend '{other}'")),
    };
    set_backend_kind(kind);
    Ok(())
}

#[derive(Clone, Debug)]
pub struct GpuDft<F: TwoAdicField> {
    backend: BackendKind,
    cpu: Radix2DitParallel<F>,
}

impl<F: TwoAdicField> Default for GpuDft<F> {
    fn default() -> Self {
        Self {
            backend: get_backend_kind(),
            cpu: Radix2DitParallel::default(),
        }
    }
}

impl<F: TwoAdicField> GpuDft<F> {
    pub fn with_backend(backend: BackendKind) -> Self {
        Self {
            backend,
            cpu: Radix2DitParallel::default(),
        }
    }
}

impl<F: TwoAdicField + Ord> TwoAdicSubgroupDft<F> for GpuDft<F> {
    type Evaluations = RowMajorMatrix<F>;

    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> Self::Evaluations {
        match self.backend {
            BackendKind::Cpu => self.cpu.dft_batch(mat).to_row_major_matrix(),
            BackendKind::Vulkan => backend_vulkan::dft_batch(&self.cpu, mat.clone())
                .unwrap_or_else(|_| self.cpu.dft_batch(mat).to_row_major_matrix()),
            BackendKind::Metal => backend_metal::dft_batch(&self.cpu, mat.clone())
                .unwrap_or_else(|_| self.cpu.dft_batch(mat).to_row_major_matrix()),
            BackendKind::WebGpu => backend_webgpu::dft_batch(&self.cpu, mat.clone())
                .unwrap_or_else(|_| self.cpu.dft_batch(mat).to_row_major_matrix()),
        }
    }
}
