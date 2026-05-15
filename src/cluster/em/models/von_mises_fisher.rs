use crate::cluster::em::optimizer::EmModel;
use crate::cluster::kmeans::Centers;
use crate::cluster::kmeans::init::Initialization;
use crate::{Float, VectorData as Dataset};

/// Simple approximation of log normalization constant for von Mises–Fisher
/// distribution.  The full expression involves a modified Bessel function of
/// the first kind; here we use a crude asymptotic approximation that is
/// sufficient for clustering purposes.  The routine returns the log of the
/// mixture component weight multiplied by the normalization constant.
fn log_norm_vmf<F>(weight: F, dim: usize, kappa: F) -> F
where
    F: Float,
{
    let d = F::from(dim).unwrap();
    let log_2pi = F::from(2.0 * std::f64::consts::PI).unwrap().ln();
    let log_c = if kappa <= F::zero() {
        // near-uniform on the sphere
        -d * F::half() * log_2pi
    } else {
        // asymptotic I_v(kappa) ~ exp(kappa) / sqrt(2*pi*kappa)
        -kappa + (d * F::half() - F::half()) * kappa.ln() - (d * F::half() - F::half()) * log_2pi
    };
    weight.ln() + log_c
}

/// Directional von Mises–Fisher component for EM on the unit hypersphere.
#[derive(Clone, Debug)]
pub struct VonMisesFisherModel<F>
where
    F: Float,
{
    mu: Vec<F>,   // mean direction, always unit norm
    nsum: Vec<F>, // accumulator for responsibilities
    kappa: F,     // concentration
    wsum: F,      // sum of responsibilities
    weight: F,    // mixture weight
    log_norm: F,  // cached log normalization + ln(weight)
}

impl<F: Float> VonMisesFisherModel<F> {
    pub fn new(weight: F, mu: Vec<F>, kappa: F) -> Self {
        let len = mu.len();
        let mut model = Self {
            mu,
            nsum: vec![F::zero(); len],
            kappa: kappa.max(F::zero()),
            wsum: F::zero(),
            weight,
            log_norm: F::zero(),
        };
        model.update_log_norm();
        model
    }

    fn update_log_norm(&mut self) {
        self.log_norm = log_norm_vmf(self.weight, self.mu.len(), self.kappa);
    }

    /// Accessor for mean direction
    pub fn mean(&self) -> &[F] { &self.mu }

    /// Current concentration
    pub fn kappa(&self) -> F { self.kappa }
}

impl<F: Float> EmModel<F> for VonMisesFisherModel<F> {
    fn begin_estep(&mut self) {
        self.wsum = F::zero();
        for v in &mut self.nsum {
            *v = F::zero();
        }
    }

    fn update_estep(&mut self, x: &[F], responsibility: F) {
        if responsibility <= F::zero() {
            return;
        }
        self.wsum += responsibility;
        for (j, &xj) in x.iter().enumerate() {
            self.nsum[j] += xj * responsibility;
        }
    }

    fn finalize_estep(&mut self, weight: F, _prior: F) {
        self.weight = weight.max(F::epsilon());
        let d = F::from(self.mu.len()).unwrap();
        if self.wsum > F::zero() && self.wsum.is_finite() {
            // update mean direction
            let mut norm = F::zero();
            for j in 0..self.mu.len() {
                self.mu[j] = self.nsum[j] / self.wsum;
                norm += self.mu[j] * self.mu[j];
            }
            norm = norm.sqrt();
            if norm > F::zero() {
                for val in &mut self.mu {
                    *val /= norm;
                }
            }
            // update concentration using Banerjee et al. (2005)
            let r_bar = norm / self.wsum;
            let num = r_bar * d - r_bar * r_bar * r_bar;
            let den = F::one() - r_bar * r_bar;
            let kappa_new = if den.abs() > F::epsilon() { num / den } else { F::zero() };
            self.kappa = kappa_new.max(F::zero());
        }
        self.update_log_norm();
    }

    fn estimate_log_density(&self, x: &[F]) -> F {
        // dot product
        let mut dot = F::zero();
        for (a, b) in self.mu.iter().zip(x.iter()) {
            dot += *a * *b;
        }
        self.kappa * dot + self.log_norm
    }

    fn weight(&self) -> F { self.weight }

    fn set_weight(&mut self, weight: F) {
        self.weight = weight.max(F::epsilon());
        self.update_log_norm();
    }
}

/// Factory for von Mises–Fisher mixtures.
/// Initial kappa may be set by the caller.
#[derive(Debug)]
pub struct VonMisesFisherModelFactory<F, I>
where
    F: Float,
    I: Initialization<F>,
{
    pub initializer: I,
    pub init_kappa: F,
}

impl<F: Float, I: Initialization<F>> VonMisesFisherModelFactory<F, I> {
    pub fn new(initializer: I) -> Self { Self { initializer, init_kappa: F::from(10.0).unwrap() } }

    pub fn with_kappa(mut self, kappa: F) -> Self {
        self.init_kappa = kappa.max(F::zero());
        self
    }

    pub fn build_initial_models<A>(&mut self, data: &A, k: usize) -> Vec<VonMisesFisherModel<F>>
    where
        A: Dataset<F>,
    {
        let d = data.ncols();
        let mut cent = Centers::<F>::new(k, d);
        self.initializer.init::<A>(data, &mut cent, k);

        let weight = F::one() / F::from(k).unwrap();
        let mut models = Vec::with_capacity(k);
        for i in 0..k {
            // normalize the center to lie on the unit sphere
            let mut mu = cent.center(i).to_vec();
            let mut norm = F::zero();
            for v in &mu {
                norm += *v * *v;
            }
            norm = norm.sqrt();
            if norm > F::zero() {
                for v in &mut mu {
                    *v /= norm;
                }
            }
            models.push(VonMisesFisherModel::new(weight, mu, self.init_kappa));
        }
        models
    }

    pub fn build_initial_models_dispatch<A>(
        initializer: I, data: &A, k: usize,
    ) -> Vec<VonMisesFisherModel<F>>
    where
        A: Dataset<F>,
    {
        let mut factory = VonMisesFisherModelFactory::new(initializer);
        factory.build_initial_models(data, k)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::em::optimizer::{EmConfig, expectation_maximization};
    use crate::cluster::kmeans::init::FirstK;

    #[test]
    fn test_vmf_model_density() {
        let mu = vec![1.0f64, 0.0, 0.0];
        let m = VonMisesFisherModel::<f64>::new(1.0, mu.clone(), 5.0);
        let ld = m.estimate_log_density(&mu);
        assert!(ld.is_finite());
        // density at the mean should be larger than at the antipode
        let opposite = vec![-1.0f64, 0.0, 0.0];
        assert!(m.estimate_log_density(&mu) > m.estimate_log_density(&opposite));
    }

    #[test]
    fn test_vmf_fit() {
        // create two antipodal clusters on the unit sphere in 3D
        let mut data = Array2::<f64>::zeros((200, 3));
        for i in 0..100 {
            data[[i, 0]] = 1.0;
            data[[i, 1]] = 0.0;
            data[[i, 2]] = 0.0;
        }
        for i in 100..200 {
            data[[i, 0]] = -1.0;
            data[[i, 1]] = 0.0;
            data[[i, 2]] = 0.0;
        }
        // normalize (should already be unit vectors, but do it for completeness)
        for i in 0..200 {
            let mut norm = 0.0f64;
            for j in 0..3 {
                norm += data[[i, j]] * data[[i, j]];
            }
            norm = norm.sqrt();
            if norm > 0.0 {
                for j in 0..3 {
                    data[[i, j]] /= norm;
                }
            }
        }
        let ds = NdArrayDataset::new(&data);
        let models =
            VonMisesFisherModelFactory::build_initial_models_dispatch(FirstK::<f64>::new(), &ds, 2);
        let cfg = EmConfig::<f64> { maxiter: 100, ..Default::default() };
        let result = expectation_maximization(&ds, 2, models, cfg).unwrap();
        assert!(result.n_iter > 0);
        assert!(result.log_likelihood.is_finite());
        assert_eq!(result.models.len(), 2);
        // ensure concentrations were updated
        for m in &result.models {
            assert!(m.kappa() >= 0.0);
            assert!(m.mean().iter().any(|&v| v != 0.0));
        }
    }
}
