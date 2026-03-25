/// Kernel functions for kernel density estimation.
///
/// Shapes and constants follow ELKI's `KernelDensityFunction` API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDensityFunction {
    Uniform,
    Triangular,
    Epanechnikov,
    Biweight,
    Triweight,
    Cosine,
    Gaussian,
}

impl KernelDensityFunction {
    pub fn density(self, delta: f64) -> f64 {
        let x = delta.abs();
        match self {
            KernelDensityFunction::Uniform => {
                if x <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            KernelDensityFunction::Triangular => {
                if x <= 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            KernelDensityFunction::Epanechnikov => {
                if x < 1.0 {
                    0.75 * (1.0 - x * x)
                } else {
                    0.0
                }
            }
            KernelDensityFunction::Biweight => {
                if x < 1.0 {
                    0.9375 * (1.0 - x * x).powi(2)
                } else {
                    0.0
                }
            }
            KernelDensityFunction::Triweight => {
                if x < 1.0 {
                    1.09375 * (1.0 - x * x).powi(3)
                } else {
                    0.0
                }
            }
            KernelDensityFunction::Cosine => {
                if x < 1.0 {
                    std::f64::consts::FRAC_PI_4 * (std::f64::consts::PI * x / 2.0).cos()
                } else {
                    0.0
                }
            }
            KernelDensityFunction::Gaussian => {
                (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * x * x).exp()
            }
        }
    }

    pub fn canonical_bandwidth(self) -> f64 {
        match self {
            KernelDensityFunction::Uniform => 1.0,
            KernelDensityFunction::Triangular => 1.0,
            KernelDensityFunction::Epanechnikov => 15f64.powf(0.2),
            KernelDensityFunction::Biweight => 15f64.powf(0.2),
            KernelDensityFunction::Triweight => (9450.0_f64 / 143.0_f64).powf(0.2_f64),
            KernelDensityFunction::Cosine => (std::f64::consts::PI * std::f64::consts::PI
                / (16.0_f64 * (1.0_f64 - 8.0_f64 / (std::f64::consts::PI * std::f64::consts::PI))))
                .powf(0.2_f64),
            KernelDensityFunction::Gaussian => (0.25_f64 / std::f64::consts::PI).powf(0.1_f64),
        }
    }

    pub fn standard_deviation(self) -> f64 {
        match self {
            KernelDensityFunction::Uniform => (1.0_f64 / 3.0_f64).sqrt(),
            KernelDensityFunction::Triangular => (1.0_f64 / 6.0_f64).sqrt(),
            KernelDensityFunction::Epanechnikov => (1.0_f64 / 5.0_f64).sqrt(),
            KernelDensityFunction::Biweight => (1.0_f64 / 7.0_f64).sqrt(),
            KernelDensityFunction::Triweight => (1.0_f64 / 9.0_f64).sqrt(),
            KernelDensityFunction::Cosine => {
                ((1.0 - 8.0 / (std::f64::consts::PI * std::f64::consts::PI)).max(0.0)).sqrt()
            }
            KernelDensityFunction::Gaussian => 1.0,
        }
    }

    pub fn r_value(self) -> f64 {
        match self {
            KernelDensityFunction::Uniform => 0.5,
            KernelDensityFunction::Triangular => 2.0 / 3.0,
            KernelDensityFunction::Epanechnikov => 3.0 / 5.0,
            KernelDensityFunction::Biweight => 3.0 / 7.0,
            KernelDensityFunction::Triweight => 350.0 / 429.0,
            KernelDensityFunction::Cosine => std::f64::consts::PI * std::f64::consts::PI / 16.0,
            KernelDensityFunction::Gaussian => 0.5 / std::f64::consts::PI.sqrt(),
        }
    }
}
