use crate::cluster::em::models::common::{log_norm_det_diagonal, scale_component_covariance};
use crate::cluster::em::optimizer::EmModel;
use crate::cluster::kmeans::Centers;
use crate::cluster::kmeans::init::Initialization;
use crate::{Float, VectorData as Dataset};

/// Textbook (numerically weaker) diagonal-covariance Gaussian component for EM.
#[derive(Clone, Debug)]
pub struct TextbookDiagonalGaussianModel<N>
where
    N: Float,
{
    mean: Vec<N>,
    variance: Vec<N>,
    sumsq: Vec<N>,
    wsum: N,
    weight: N,
    log_norm_det: N,
    prior_variance: Option<Vec<N>>,
    min_variance: N,
}

impl<N: Float> TextbookDiagonalGaussianModel<N> {
    pub fn new(weight: N, mean: Vec<N>, variance: Vec<N>, min_variance: N) -> Self {
        assert_eq!(mean.len(), variance.len(), "mean/variance size mismatch");
        let dim = variance.len();
        let mut model = Self {
            mean,
            variance,
            sumsq: vec![N::zero(); dim],
            wsum: N::zero(),
            weight,
            log_norm_det: N::zero(),
            prior_variance: None,
            min_variance,
        };
        model.prior_variance = Some(model.variance.clone());
        model.update_log_norm_det();
        model
    }

    pub fn mean(&self) -> &[N] { &self.mean }

    pub fn variance(&self) -> &[N] { &self.variance }

    pub fn min_variance(&self) -> N { self.min_variance }

    fn update_log_norm_det(&mut self) {
        self.log_norm_det = log_norm_det_diagonal(self.weight, &self.variance, self.min_variance);
    }
}

impl<N: Float> EmModel<N> for TextbookDiagonalGaussianModel<N> {
    fn begin_estep(&mut self) {
        self.wsum = N::zero();
        self.mean.fill(N::zero());
        self.sumsq.fill(N::zero());
    }

    fn update_estep(&mut self, x: &[N], responsibility: N) {
        if responsibility <= N::zero() {
            return;
        }
        for (j, &xj) in x.iter().enumerate() {
            let wx = responsibility * xj;
            self.mean[j] = self.mean[j] + wx;
            self.sumsq[j] = self.sumsq[j] + wx * xj;
        }
        self.wsum = self.wsum + responsibility;
    }

    fn finalize_estep(&mut self, weight: N, prior: N) {
        self.weight = weight.max(N::epsilon());
        let d = self.mean.len();
        if self.wsum > N::zero() && self.wsum.is_finite() {
            let inv_w = self.wsum.recip();
            for m in &mut self.mean {
                *m = *m * inv_w;
            }
            if prior > N::zero() && self.prior_variance.is_some() {
                let prior_diag = self.prior_variance.as_ref().unwrap();
                let nu = N::from(d + 2).unwrap();
                let denom = self.wsum + prior * (nu + N::from(d + 2).unwrap());
                for (i, &pval) in prior_diag.iter().enumerate().take(d) {
                    let sse =
                        (self.sumsq[i] - self.mean[i] * self.mean[i] * self.wsum).max(N::zero());
                    let v = (sse + prior * pval) / denom;
                    self.variance[i] = v.max(self.min_variance);
                }
            } else {
                for (i, &s) in self.sumsq.iter().enumerate().take(d) {
                    let v = s * inv_w - self.mean[i] * self.mean[i];
                    self.variance[i] = v.max(self.min_variance);
                }
            }
        } else {
            self.variance.fill(self.min_variance);
        }
        if prior > N::zero() && self.prior_variance.is_none() {
            self.prior_variance = Some(self.variance.clone());
        }
        self.update_log_norm_det();
    }

    fn estimate_log_density(&self, x: &[N]) -> N {
        let mut mahal = N::zero();
        for (j, &xj) in x.iter().enumerate() {
            let var = self.variance[j].max(self.min_variance);
            let diff = xj - self.mean[j];
            mahal = mahal + diff * diff / var;
        }
        -N::from(0.5).unwrap() * mahal + self.log_norm_det
    }

    fn weight(&self) -> N { self.weight }

    fn set_weight(&mut self, weight: N) {
        self.weight = weight.max(N::epsilon());
        self.update_log_norm_det();
    }
}

/// Factory for textbook (numerically weaker) diagonal Gaussian mixture models.
#[derive(Debug)]
pub struct TextbookDiagonalGaussianModelFactory<N, I>
where
    N: Float,
    I: Initialization<N>,
{
    pub initializer: I,
    pub min_variance: N,
}

impl<N: Float, I: Initialization<N>> TextbookDiagonalGaussianModelFactory<N, I> {
    pub fn new(initializer: I) -> Self {
        Self { initializer, min_variance: N::from(1e-10).unwrap() }
    }

    fn global_variance<A>(&self, data: &A) -> Vec<N>
    where
        A: Dataset<N>,
    {
        Self::global_diagonal_variance(data, self.min_variance)
    }

    fn global_diagonal_variance<A>(data: &A, min_variance: N) -> Vec<N>
    where
        A: Dataset<N>,
    {
        let (n, d) = (data.nrows(), data.ncols());
        let nf = N::from(n).unwrap();
        let mut scratch = vec![N::zero(); d];
        let mut sum = vec![N::zero(); d];
        let mut sumsq = vec![N::zero(); d];
        let mut var = vec![N::zero(); d];

        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            for j in 0..d {
                let x = scratch[j];
                sum[j] += x;
                sumsq[j] += x * x;
            }
        }

        for j in 0..d {
            let mean = sum[j] / nf;
            let second_moment = sumsq[j] / nf;
            var[j] = (second_moment - mean * mean).max(min_variance);
        }
        var
    }

    pub fn build_initial_models<A>(
        &mut self, data: &A, k: usize,
    ) -> Vec<TextbookDiagonalGaussianModel<N>>
    where
        A: Dataset<N>,
    {
        let d = data.ncols();
        let mut cent = Centers::<N>::new(k, d);
        self.initializer.init::<A>(data, &mut cent, k);

        let mut base = self.global_variance(data);
        scale_component_covariance(&mut base, k, d, self.min_variance);

        let weight = N::one() / N::from(k).unwrap();
        let mut models = Vec::with_capacity(k);
        for i in 0..k {
            models.push(TextbookDiagonalGaussianModel::new(
                weight,
                cent.center(i).to_vec(),
                base.clone(),
                self.min_variance,
            ));
        }
        models
    }

    pub fn build_initial_models_dispatch<A>(
        initializer: I, data: &A, k: usize,
    ) -> Vec<TextbookDiagonalGaussianModel<N>>
    where
        A: Dataset<N>,
    {
        let mut factory = TextbookDiagonalGaussianModelFactory::new(initializer);
        factory.build_initial_models(data, k)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use crate::cluster::em::models::textbook_diagonal::TextbookDiagonalGaussianModelFactory;
    use crate::cluster::em::optimizer::{EmConfig, expectation_maximization};
    use crate::cluster::kmeans::init::FirstK;
    use crate::cluster::kmeans::ndarray::NdArrayDataset;

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
    fn test_textbook_diagonal_gmm_fit() {
        let data = two_blob_data();
        let ds = NdArrayDataset::new(&data);
        let models = TextbookDiagonalGaussianModelFactory::build_initial_models_dispatch(
            FirstK::<f64>::new(),
            &ds,
            2,
        );
        let cfg = EmConfig::<f64> { maxiter: 100, return_soft: true, ..Default::default() };
        let result = expectation_maximization(&ds, 2, models, cfg);
        assert!(result.n_iter > 0);
        assert!(result.log_likelihood.is_finite());
        let means = result.models.iter().map(|m| m.mean()[0]).collect::<Vec<_>>();
        assert_eq!(means.len(), 2);
        assert!((means[0] - means[1]).abs() > 1e-6);
    }
}
