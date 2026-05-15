use crate::cluster::em::models::common::{log_norm_det_spherical, scale_component_variance};
use crate::cluster::em::optimizer::EmModel;
use crate::cluster::kmeans::Centers;
use crate::cluster::kmeans::init::Initialization;
use crate::{Float, VectorData as Dataset};

/// Textbook (numerically weaker) spherical Gaussian component for EM.
#[derive(Clone, Debug)]
pub struct TextbookSphericalGaussianModel<N>
where
    N: Float,
{
    mean: Vec<N>,
    variance: N,
    sumsq: N,
    wsum: N,
    weight: N,
    log_norm_det: N,
    prior_variance: N,
    min_variance: N,
}

impl<N: Float> TextbookSphericalGaussianModel<N> {
    pub fn new(weight: N, mean: Vec<N>, variance: N, min_variance: N) -> Self {
        let mut model = Self {
            mean,
            variance: variance.max(min_variance),
            sumsq: N::zero(),
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

    pub fn min_variance(&self) -> N { self.min_variance }

    fn update_log_norm_det(&mut self) {
        self.log_norm_det =
            log_norm_det_spherical(self.weight, self.mean.len(), self.variance, self.min_variance);
    }
}

impl<N: Float> EmModel<N> for TextbookSphericalGaussianModel<N> {
    fn begin_estep(&mut self) {
        self.wsum = N::zero();
        self.mean.fill(N::zero());
        self.sumsq = N::zero();
    }

    fn update_estep(&mut self, x: &[N], responsibility: N) {
        if responsibility <= N::zero() {
            return;
        }
        for (j, &xj) in x.iter().enumerate() {
            let wx = responsibility * xj;
            self.mean[j] += wx;
            self.sumsq += wx * xj;
        }
        self.wsum += responsibility;
    }

    fn finalize_estep(&mut self, weight: N, prior: N) {
        self.weight = weight.max(N::epsilon());
        let d = self.mean.len();
        let df = N::from(d).unwrap();
        if self.wsum > N::zero() && self.wsum.is_finite() {
            let inv_w = self.wsum.recip();
            for m in &mut self.mean {
                *m *= inv_w;
            }
            if prior > N::zero() {
                let nu = N::from(d + 2).unwrap();
                let denom = self.wsum + prior * (nu + N::from(d + 2).unwrap());
                let mean_sq_sum =
                    self.mean.iter().fold(N::zero(), |acc, &m| acc + m * m * self.wsum);
                let sse = (self.sumsq - mean_sq_sum).max(N::zero());
                self.variance = ((sse / df) + prior * self.prior_variance) / denom;
            } else {
                let mean_sq = self.mean.iter().fold(N::zero(), |acc, &m| acc + m * m) / df;
                self.variance = (self.sumsq / (self.wsum * df) - mean_sq).max(self.min_variance);
            }
        } else {
            self.variance = self.min_variance;
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

/// Factory for textbook (numerically weaker) spherical Gaussian mixture models.
#[derive(Debug)]
pub struct TextbookSphericalGaussianModelFactory<N, I>
where
    N: Float,
    I: Initialization<N>,
{
    pub initializer: I,
    pub min_variance: N,
}

impl<N: Float, I: Initialization<N>> TextbookSphericalGaussianModelFactory<N, I> {
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
        let mut sum = vec![N::zero(); d];
        let mut sumsq = vec![N::zero(); d];

        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            for j in 0..d {
                let x = scratch[j];
                sum[j] += x;
                sumsq[j] += x * x;
            }
        }

        // compute sum of squared errors across dimensions
        let mut sse = N::zero();
        for j in 0..d {
            let mean_j = sum[j] / nf;
            sse += sumsq[j] - nf * mean_j * mean_j;
        }

        (sse / (nf * df)).max(self.min_variance)
    }

    pub fn build_initial_models<A>(
        &mut self, data: &A, k: usize,
    ) -> Vec<TextbookSphericalGaussianModel<N>>
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
            models.push(TextbookSphericalGaussianModel::new(
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
    ) -> Vec<TextbookSphericalGaussianModel<N>>
    where
        A: Dataset<N>,
    {
        let mut factory = TextbookSphericalGaussianModelFactory::new(initializer);
        factory.build_initial_models(data, k)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use crate::NdArrayDataset;
    use crate::cluster::em::models::textbook_spherical::TextbookSphericalGaussianModelFactory;
    use crate::cluster::em::optimizer::{EmConfig, EmResult, expectation_maximization};
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
    fn test_textbook_spherical_gmm_fit() {
        let data = two_blob_data();
        let ds = NdArrayDataset::new(&data);
        let models = TextbookSphericalGaussianModelFactory::build_initial_models_dispatch(
            FirstK::<f64>::new(),
            &ds,
            2,
        );
        let cfg = EmConfig::<f64> { maxiter: 100, ..Default::default() };
        let result: EmResult<_, _> = expectation_maximization(&ds, 2, models, cfg).unwrap();
        assert!(result.n_iter > 0);
        assert!(result.log_likelihood.is_finite());
        let means = result.models.iter().map(|m| m.mean()[0]).collect::<Vec<_>>();
        assert_eq!(means.len(), 2);
        assert!((means[0] - means[1]).abs() > 1e-6);
    }
}
