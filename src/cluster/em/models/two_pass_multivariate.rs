use ndarray_linalg::Scalar;

use crate::cluster::em::models::common::{
    idx, mahalanobis_distance_from_cholesky, refresh_cholesky_log_norm_det,
    scale_component_covariance, symmetrize,
};
use crate::cluster::em::optimizer::EmModel;
use crate::cluster::kmeans::Centers;
use crate::cluster::kmeans::init::Initialization;
use crate::{Float, VectorData as Dataset, math};

/// "Textbook" two‑pass multivariate Gaussian component; slightly more
/// numerically stable than the single-pass textbook version, but still not
/// recommended for production use. Added for reference.
#[derive(Clone, Debug)]
pub struct TwoPassMultivariateGaussianModel<N>
where
    N: Float,
{
    mean: Vec<N>,
    covariance: Vec<N>,
    wsum: N,
    weight: N,
    log_norm: N,
    log_norm_det: N,
    chol: Vec<N>,
    prior_covariance: Option<Vec<N>>,
    min_variance: N,
}

impl<N> TwoPassMultivariateGaussianModel<N>
where
    N: Float + Scalar + ndarray_linalg::Lapack,
{
    pub fn new(weight: N, mean: Vec<N>, covariance: Vec<N>, min_variance: N) -> Self {
        let dim = mean.len();
        assert_eq!(covariance.len(), dim * dim, "covariance size mismatch");
        let log_2pi = num_traits::Float::ln(N::from(2.0 * std::f64::consts::PI).unwrap());
        let log_norm = N::from(dim).unwrap() * log_2pi;
        let mut model = Self {
            mean,
            covariance,
            wsum: N::zero(),
            weight,
            log_norm,
            log_norm_det: N::zero(),
            chol: vec![N::zero(); dim * dim],
            prior_covariance: None,
            min_variance,
        };
        model.refresh_cholesky();
        model
    }

    pub fn mean(&self) -> &[N] { &self.mean }

    pub fn covariance(&self) -> &[N] { &self.covariance }

    pub fn min_variance(&self) -> N { self.min_variance }

    fn refresh_cholesky(&mut self) {
        let dim = self.mean.len();
        self.log_norm_det = refresh_cholesky_log_norm_det(
            &mut self.covariance,
            dim,
            self.min_variance,
            self.weight,
            self.log_norm,
            &mut self.chol,
        );
    }
}

impl<N> EmModel<N> for TwoPassMultivariateGaussianModel<N>
where
    N: Float + Scalar + ndarray_linalg::Lapack,
{
    fn begin_estep(&mut self) {
        self.wsum = N::zero();
        self.mean.fill(N::zero());
        self.covariance.fill(N::zero());
    }

    fn needs_two_pass(&self) -> bool { true }

    fn first_pass_estep(&mut self, x: &[N], responsibility: N) {
        if responsibility <= N::zero() {
            return;
        }
        for (i, &xi) in x.iter().enumerate() {
            self.mean[i] += xi * responsibility;
        }
        self.wsum += responsibility;
    }

    fn finalize_first_pass_estep(&mut self) {
        if self.wsum > N::zero() {
            let inv = self.wsum.recip();
            for m in &mut self.mean {
                *m *= inv;
            }
        }
    }

    fn update_estep(&mut self, x: &[N], responsibility: N) {
        if responsibility <= N::zero() {
            return;
        }
        let dim = self.mean.len();
        // compute residual vector once
        let mut diff = vec![N::zero(); dim];
        for (i, &xi) in x.iter().enumerate().take(dim) {
            diff[i] = xi - self.mean[i];
        }
        for u in 0..dim {
            let scale = diff[u] * responsibility;
            let row = &mut self.covariance[u * dim..u * dim + (u + 1)];
            math::axpy(row, scale, &diff[..(u + 1)], u + 1);
        }
    }

    fn finalize_estep(&mut self, weight: N, prior: N) {
        self.weight = weight.max(N::epsilon());
        let dim = self.mean.len();
        if self.wsum > N::zero() && self.wsum.is_finite() {
            if prior > N::zero() {
                if let Some(prior_cov) = self.prior_covariance.as_ref() {
                    let nu = N::from(dim + 2).unwrap();
                    let denom =
                        self.wsum + prior * (nu + N::from(dim).unwrap() + N::from(2.0).unwrap());
                    for i in 0..dim {
                        for j in 0..=i {
                            let idx = idx(i, j, dim);
                            let scaled = (self.covariance[idx] + prior * prior_cov[idx]) / denom;
                            self.covariance[idx] = scaled;
                        }
                    }
                }
            } else {
                let inv = self.wsum.recip();
                for i in 0..dim {
                    for j in 0..=i {
                        let idx = idx(i, j, dim);
                        self.covariance[idx] *= inv;
                    }
                }
            }
        }
        symmetrize(&mut self.covariance, dim);
        self.refresh_cholesky();
        if prior > N::zero() && self.prior_covariance.is_none() {
            self.prior_covariance = Some(self.covariance.clone());
        }
    }

    fn estimate_log_density(&self, x: &[N]) -> N {
        mahalanobis_distance_from_cholesky(&self.chol, &self.mean, x)
            - N::from(0.5).unwrap() * mahalanobis_distance_from_cholesky(&self.chol, &self.mean, x)
            + self.log_norm_det
    }

    fn weight(&self) -> N { self.weight }

    fn set_weight(&mut self, weight: N) {
        self.weight = weight.max(N::epsilon());
        self.refresh_cholesky();
    }
}

/// Factory for two‑pass textbook multivariate model.
#[derive(Debug)]
pub struct TwoPassMultivariateGaussianModelFactory<N, I>
where
    N: Float,
    I: Initialization<N>,
{
    pub initializer: I,
    pub min_variance: N,
}

impl<N, I> TwoPassMultivariateGaussianModelFactory<N, I>
where
    N: Float + Scalar + ndarray_linalg::Lapack,
    I: Initialization<N>,
{
    pub fn new(initializer: I) -> Self {
        Self { initializer, min_variance: N::from(1e-10).unwrap() }
    }

    fn global_covariance<A>(&self, data: &A) -> Vec<N>
    where
        A: Dataset<N>,
    {
        let (n, d) = (data.nrows(), data.ncols());
        let nf = N::from(n).unwrap();
        let mut scratch = vec![N::zero(); d];
        let mut cov = vec![N::zero(); d * d];

        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            // for each row u add scratch[u] * scratch vector
            for u in 0..d {
                // maintain same half-matrix semantics
                let row = &mut cov[u * d..u * d + (u + 1)];
                math::axpy(row, scratch[u], &scratch[..(u + 1)], u + 1);
            }
        }

        for i in 0..d {
            for j in 0..=i {
                let pos = idx(i, j, d);
                cov[pos] = (cov[pos] / nf).max(self.min_variance);
                cov[idx(j, i, d)] = cov[pos];
            }
        }
        cov
    }

    pub fn build_initial_models<A>(
        &mut self, data: &A, k: usize,
    ) -> Vec<TwoPassMultivariateGaussianModel<N>>
    where
        A: Dataset<N>,
    {
        let d = data.ncols();
        let mut cent = Centers::<N>::new(k, d);
        self.initializer.init::<A>(data, &mut cent, k);

        let mut cov = self.global_covariance(data);
        scale_component_covariance(&mut cov, k, d, self.min_variance);

        let weight = N::one() / N::from(k).unwrap();
        let mut models = Vec::with_capacity(k);
        for i in 0..k {
            models.push(TwoPassMultivariateGaussianModel::new(
                weight,
                cent.center(i).to_vec(),
                cov.clone(),
                self.min_variance,
            ));
        }
        models
    }

    pub fn build_initial_models_dispatch<A>(
        initializer: I, data: &A, k: usize,
    ) -> Vec<TwoPassMultivariateGaussianModel<N>>
    where
        A: Dataset<N>,
    {
        let mut factory = TwoPassMultivariateGaussianModelFactory::new(initializer);
        factory.build_initial_models(data, k)
    }
}

// tests
#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::em::optimizer::{EmConfig, expectation_maximization};
    use crate::cluster::kmeans::init::FirstK;

    fn two_blob_data() -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((200, 2));
        for i in 0..100 {
            data[[i, 0]] = -4.0 + (i as f64) * 0.01;
            data[[i, 1]] = -4.0 + (i as f64) * 0.01;
        }
        for i in 100..200 {
            data[[i, 0]] = 4.0 + ((i - 100) as f64) * 0.01;
            data[[i, 1]] = 4.0 + ((i - 100) as f64) * 0.01;
        }
        data
    }

    #[test]
    fn test_two_pass_density() {
        let mean = vec![0.0f64, 0.0];
        let cov = vec![1.0, 0.0, 0.0, 1.0];
        let mut m = TwoPassMultivariateGaussianModel::<f64>::new(1.0, mean.clone(), cov, 1e-10);
        let ld = m.estimate_log_density(&mean);
        let dim = 2;
        let expected = -0.5 * (dim as f64) * (2.0 * std::f64::consts::PI).ln();
        assert!((ld - expected).abs() < 1e-12);
        m.set_weight(0.2);
        let ld2 = m.estimate_log_density(&mean);
        assert!(ld2 < ld);
    }

    #[test]
    fn test_two_pass_fit() {
        let data = two_blob_data();
        let ds = NdArrayDataset::new(&data);
        let models = TwoPassMultivariateGaussianModelFactory::build_initial_models_dispatch(
            FirstK::<f64>::new(),
            &ds,
            2,
        );
        let cfg = EmConfig::<f64> { maxiter: 100, return_soft: true, ..Default::default() };
        let result = expectation_maximization(&ds, 2, models, cfg).unwrap();
        assert!(result.n_iter > 0);
        assert!(result.log_likelihood.is_finite());
        let means = result.models.iter().map(|m| m.mean()[0]).collect::<Vec<_>>();
        assert_eq!(means.len(), 2);
        assert!((means[0] - means[1]).abs() > 1e-6);
    }
}
