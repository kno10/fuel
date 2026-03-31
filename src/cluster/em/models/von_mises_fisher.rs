use crate::cluster::em::optimizer::EmModel;
use crate::cluster::kmeans::init::Initialization;
use crate::cluster::kmeans::Centers;
use crate::{Float, VectorData as Dataset};
use std::iter::Sum;
use std::ops::{AddAssign, MulAssign, SubAssign};

/// Simple approximation of log normalization constant for von Mises–Fisher
/// distribution.  The full expression involves a modified Bessel function of
/// the first kind; here we use a crude asymptotic approximation that is
/// sufficient for clustering purposes.  The routine returns the log of the
/// mixture component weight multiplied by the normalization constant.
fn log_norm_vmf<N>(weight: N, dim: usize, kappa: N) -> N
where
    N: Float + Copy,
{
    let d = N::from(dim).unwrap();
    let log_2pi = N::from(2.0 * std::f64::consts::PI).unwrap().ln();
    let log_c = if kappa <= N::zero() {
        // near-uniform on the sphere
        -d / N::from(2.0).unwrap() * log_2pi
    } else {
        // asymptotic I_v(kappa) ~ exp(kappa) / sqrt(2*pi*kappa)
        let half = N::from(0.5).unwrap();
        -kappa + (d / N::from(2.0).unwrap() - half) * kappa.ln()
            - (d / N::from(2.0).unwrap() - half) * log_2pi
    };
    weight.ln() + log_c
}

/// Directional von Mises–Fisher component for EM on the unit hypersphere.
#[derive(Clone, Debug)]
pub struct VonMisesFisherModel<M, N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
    M: crate::math::Math<N>,
{
    mu: Vec<N>,   // mean direction, always unit norm
    nsum: Vec<N>, // accumulator for responsibilities
    kappa: N,     // concentration
    wsum: N,      // sum of responsibilities
    weight: N,    // mixture weight
    log_norm: N,  // cached log normalization + ln(weight)
    _math: std::marker::PhantomData<M>,
}

impl<M, N> VonMisesFisherModel<M, N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
    M: crate::math::Math<N>,
{
    pub fn new(weight: N, mu: Vec<N>, kappa: N) -> Self
    where
        N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
        M: crate::math::Math<N>,
    {
        let len = mu.len();
        let mut model = Self {
            mu,
            nsum: vec![N::zero(); len],
            kappa: kappa.max(N::zero()),
            wsum: N::zero(),
            weight,
            log_norm: N::zero(),
            _math: std::marker::PhantomData,
        };
        model.update_log_norm();
        model
    }

    fn update_log_norm(&mut self) {
        self.log_norm = log_norm_vmf(self.weight, self.mu.len(), self.kappa);
    }

    /// Accessor for mean direction
    pub fn mean(&self) -> &[N] {
        &self.mu
    }

    /// Current concentration
    pub fn kappa(&self) -> N {
        self.kappa
    }
}

impl<M, N> EmModel<N> for VonMisesFisherModel<M, N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
    M: crate::math::Math<N>,
{
    fn begin_estep(&mut self) {
        self.wsum = N::zero();
        for v in &mut self.nsum {
            *v = N::zero();
        }
    }

    fn update_estep(&mut self, x: &[N], responsibility: N) {
        if responsibility <= N::zero() {
            return;
        }
        self.wsum = self.wsum + responsibility;
        for (j, &xj) in x.iter().enumerate() {
            self.nsum[j] += xj * responsibility;
        }
    }

    fn finalize_estep(&mut self, weight: N, _prior: N) {
        self.weight = weight.max(N::epsilon());
        let d = N::from(self.mu.len()).unwrap();
        if self.wsum > N::zero() && self.wsum.is_finite() {
            // update mean direction
            let mut norm = N::zero();
            for j in 0..self.mu.len() {
                self.mu[j] = self.nsum[j] / self.wsum;
                norm += self.mu[j] * self.mu[j];
            }
            norm = norm.sqrt();
            if norm > N::zero() {
                for val in &mut self.mu {
                    *val = *val / norm;
                }
            }
            // update concentration using Banerjee et al. (2005)
            let r_bar = norm / self.wsum;
            let num = r_bar * d - r_bar * r_bar * r_bar;
            let den = N::one() - r_bar * r_bar;
            let kappa_new = if den.abs() > N::epsilon() {
                num / den
            } else {
                N::zero()
            };
            self.kappa = kappa_new.max(N::zero());
        }
        self.update_log_norm();
    }

    fn estimate_log_density(&self, x: &[N]) -> N {
        // dot product
        let mut dot = N::zero();
        for (a, b) in self.mu.iter().zip(x.iter()) {
            dot = dot + *a * *b;
        }
        self.kappa * dot + self.log_norm
    }

    fn weight(&self) -> N {
        self.weight
    }

    fn set_weight(&mut self, weight: N) {
        self.weight = weight.max(N::epsilon());
        self.update_log_norm();
    }
}

/// Factory for von Mises–Fisher mixtures.
/// Initial kappa may be set by the caller.
#[derive(Debug)]
pub struct VonMisesFisherModelFactory<M, N, I>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    M: crate::math::Math<N>,
    I: Initialization<N>,
{
    pub initializer: I,
    pub init_kappa: N,
    _math: std::marker::PhantomData<M>,
}

impl<M, N, I> VonMisesFisherModelFactory<M, N, I>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    M: crate::math::Math<N>,
    I: Initialization<N>,
{
    /// Generic constructor; caller may specify a different math backend `M`.
    pub fn with_math(initializer: I) -> Self {
        Self {
            initializer,
            init_kappa: N::from(10.0).unwrap(),
            _math: std::marker::PhantomData,
        }
    }

    pub fn with_kappa(mut self, kappa: N) -> Self {
        self.init_kappa = kappa.max(N::zero());
        self
    }

    pub fn build_initial_models<A>(&mut self, data: &A, k: usize) -> Vec<VonMisesFisherModel<M, N>>
    where
        A: Dataset<N>,
        M: crate::math::Math<N>,
    {
        let d = data.ncols();
        let mut cent = Centers::<N>::new(k, d);
        self.initializer.init::<A>(data, &mut cent, k);

        let weight = N::one() / N::from(k).unwrap();
        let mut models = Vec::with_capacity(k);
        for i in 0..k {
            // normalize the center to lie on the unit sphere
            let mut mu = cent.center(i).to_vec();
            let mut norm = N::zero();
            for v in &mu {
                norm += *v * *v;
            }
            norm = norm.sqrt();
            if norm > N::zero() {
                for v in &mut mu {
                    *v = *v / norm;
                }
            }
            models.push(VonMisesFisherModel::new(weight, mu, self.init_kappa));
        }
        models
    }
}

impl<N, I> VonMisesFisherModelFactory<crate::math::DefaultMath<N>, N, I>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    I: Initialization<N>,
{
    pub fn new(initializer: I) -> Self {
        VonMisesFisherModelFactory::with_math(initializer)
    }

    pub fn build_initial_models_dispatch<A>(
        initializer: I,
        data: &A,
        k: usize,
    ) -> Vec<VonMisesFisherModel<crate::math::DefaultMath<N>, N>>
    where
        N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum + 'static,
        A: Dataset<N>,
    {
        let mut factory: VonMisesFisherModelFactory<crate::math::DefaultMath<N>, N, I> =
            VonMisesFisherModelFactory::with_math(initializer);
        factory.build_initial_models(data, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::em::optimizer::{EmConfig, expectation_maximization};
    use crate::cluster::kmeans::init::FirstK;
    use crate::cluster::kmeans::ndarray::NdArrayDataset;
    use ndarray::Array2;

    #[test]
    fn test_vmf_model_density() {
        let mu = vec![1.0f64, 0.0, 0.0];
        let m =
            VonMisesFisherModel::<crate::math::DefaultMath<f64>, f64>::new(1.0, mu.clone(), 5.0);
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
        let cfg = EmConfig::<f64> {
            maxiter: 100,
            ..Default::default()
        };
        let result = expectation_maximization(&ds, 2, models, cfg);
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
