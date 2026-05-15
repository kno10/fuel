use crate::cluster::em::models::common::{log_norm_det_spherical, scale_component_variance};
use crate::cluster::em::optimizer::EmModel;
use crate::cluster::kmeans::Centers;
use crate::cluster::kmeans::init::Initialization;
use crate::{Float, VectorData as Dataset, math};

/// Numerically stable spherical Gaussian component for EM.
#[derive(Clone, Debug)]
pub struct SphericalGaussianModel<N>
where
    N: Float,
{
    mean: Vec<N>,
    nmean: Vec<N>,
    variance: N,
    wsum: N,
    weight: N,
    log_norm_det: N,
    prior_variance: N,
    min_variance: N,
}

impl<N: Float> SphericalGaussianModel<N> {
    pub fn new(weight: N, mean: Vec<N>, variance: N, min_variance: N) -> Self {
        let mut model = Self {
            nmean: mean.clone(),
            mean,
            variance: variance.max(min_variance),
            wsum: N::zero(),
            weight,
            log_norm_det: N::zero(),
            prior_variance: variance.max(min_variance),
            min_variance,
        };
        model.update_log_norm_det();
        model
    }

    pub fn mean(&self) -> &[N] { &self.mean }

    pub fn variance(&self) -> N { self.variance }

    /// Minimum variance used by the model.
    pub fn min_variance(&self) -> N { self.min_variance }

    fn update_log_norm_det(&mut self) {
        self.log_norm_det =
            log_norm_det_spherical(self.weight, self.mean.len(), self.variance, self.min_variance);
    }
}

impl<N: Float> EmModel<N> for SphericalGaussianModel<N> {
    fn begin_estep(&mut self) {
        self.wsum = N::zero();
        self.variance = N::zero();
    }

    fn update_estep(&mut self, x: &[N], responsibility: N) {
        if responsibility <= N::zero() {
            return;
        }
        let nwsum = self.wsum + responsibility;
        if !nwsum.is_finite() || nwsum <= N::zero() {
            return;
        }
        let f = responsibility / nwsum;
        for (j, &xj) in x.iter().enumerate() {
            self.nmean[j] = self.mean[j] + (xj - self.mean[j]) * f;
        }
        for (j, &xj) in x.iter().enumerate() {
            self.variance += (xj - self.nmean[j]) * (xj - self.mean[j]) * responsibility;
        }
        self.wsum = nwsum;
        self.mean.copy_from_slice(&self.nmean);
    }

    fn finalize_estep(&mut self, weight: N, prior: N) {
        self.weight = weight.max(N::epsilon());
        let d = N::from(self.mean.len()).unwrap();
        if prior > N::zero() {
            let nu = d + N::from(2.0).unwrap();
            self.variance /= d;
            self.variance += prior * self.prior_variance;
            self.variance /= self.wsum + prior * (nu + d + N::from(2.0).unwrap());
        } else if self.wsum > N::zero() {
            self.variance /= d * self.wsum;
        }
        self.variance = self.variance.max(self.min_variance);

        self.update_log_norm_det();
    }

    fn estimate_log_density(&self, x: &[N]) -> N {
        let var = self.variance.max(self.min_variance);
        let mut mahal = N::zero();
        for (j, &xj) in x.iter().enumerate() {
            let diff = xj - self.mean[j];
            mahal += diff * diff / var;
        }
        -N::from(0.5).unwrap() * mahal + self.log_norm_det
    }

    fn weight(&self) -> N { self.weight }

    fn set_weight(&mut self, weight: N) {
        self.weight = weight.max(N::epsilon());
        self.update_log_norm_det();
    }
}

/// Factory for numerically stable spherical Gaussian mixture models.
#[derive(Debug)]
pub struct SphericalGaussianModelFactory<N, I>
where
    N: Float,
    I: Initialization<N>,
{
    pub initializer: I,
    pub min_variance: N,
}

impl<N: Float, I: Initialization<N>> SphericalGaussianModelFactory<N, I> {
    pub fn new(initializer: I) -> Self {
        Self { initializer, min_variance: N::from(1e-10).unwrap() }
    }

    fn global_spherical_variance<A>(&self, data: &A) -> N
    where
        A: Dataset<N>,
    {
        let (n, d) = (data.nrows(), data.ncols());
        let nf = N::from(n).unwrap();
        let df = N::from(d).unwrap();
        let mut scratch = vec![N::zero(); d];

        // running mean and next-mean for stability
        let mut mean = vec![N::zero(); d];
        let mut acc = N::zero();

        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            let nwsum = N::from(i + 1).unwrap();
            if !nwsum.is_finite() || nwsum <= N::zero() {
                continue;
            }
            let f = N::one() / nwsum;

            // nmean = mean + (scratch - mean) * f
            let mut delta = scratch.clone();
            math::sub_assign(&mut delta, &mean, d);
            math::mul_assign(&mut delta, f, d);
            let mut nmean_vec = mean.clone();
            math::add_assign(&mut nmean_vec, &delta, d);

            // acc += dot(scratch - nmean, scratch - mean)
            let mut delta2 = scratch.clone();
            math::sub_assign(&mut delta2, &nmean_vec, d);
            acc += math::dot(&delta2, &delta, d);

            mean.copy_from_slice(&nmean_vec);
        }

        (acc / (nf * df)).max(self.min_variance)
    }

    pub fn build_initial_models<A>(&mut self, data: &A, k: usize) -> Vec<SphericalGaussianModel<N>>
    where
        A: Dataset<N>,
    {
        let d = data.ncols();
        let mut cent = Centers::<N>::new(k, d);
        self.initializer.init::<A>(data, &mut cent, k);

        let var =
            scale_component_variance(self.global_spherical_variance(data), k, d, self.min_variance);
        let weight = N::one() / N::from(k).unwrap();
        let mut models = Vec::with_capacity(k);
        for i in 0..k {
            models.push(SphericalGaussianModel::new(
                weight,
                cent.center(i).to_vec(),
                var,
                self.min_variance,
            ));
        }
        models
    }

    pub fn build_initial_models_dispatch<A>(
        initializer: I, data: &A, k: usize,
    ) -> Vec<SphericalGaussianModel<N>>
    where
        A: Dataset<N>,
    {
        let mut factory = SphericalGaussianModelFactory::new(initializer);
        factory.build_initial_models(data, k)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::em::models::spherical::SphericalGaussianModelFactory;
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
    fn test_spherical_model_density() {
        let dim = 2;
        let mean = vec![0.0f64; dim];
        let var = 1.0f64;
        let mut m = SphericalGaussianModel::<f64>::new(1.0, mean.clone(), var, 1e-10);
        let ld = m.estimate_log_density(&mean);
        let expected = -0.5 * (dim as f64) * (2.0 * std::f64::consts::PI).ln();
        assert!((ld - expected).abs() < 1e-12);
        m.set_weight(0.3);
        let ld2 = m.estimate_log_density(&mean);
        assert!(ld2 < ld);
    }

    #[test]
    fn test_spherical_gmm_fit() {
        let data = two_blob_data();
        let ds = NdArrayDataset::new(&data);
        let models = SphericalGaussianModelFactory::build_initial_models_dispatch(
            FirstK::<f64>::new(),
            &ds,
            2,
        );
        let cfg = EmConfig::<f64> { maxiter: 100, ..Default::default() };
        let result = expectation_maximization(&ds, 2, models, cfg).unwrap();
        assert!(result.n_iter > 0);
        assert!(result.log_likelihood.is_finite());
        assert!(result.models.iter().all(|m| m.variance() > 0.0));
        let means = result.models.iter().map(|m| m.mean()[0]).collect::<Vec<_>>();
        assert_eq!(means.len(), 2);
        assert!((means[0] - means[1]).abs() > 1e-6);
    }
}
