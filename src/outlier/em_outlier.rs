/// EM clustering outlier detection.
///
/// Scores each point by its log-likelihood under the fitted mixture model.
/// Low log-likelihood indicates an outlier (the point is unlikely under the
/// model).
///
/// Reference: McLachlan, Basford - *Mixture Models*, 1988.
use crate::Float;
use crate::VectorData;
use crate::cluster::em::{EmModel, EmResult, log_sum_exp};
use crate::outlier::common::{OutlierResult, make_outlier_result};

/// EM-based outlier detection using per-point log-likelihood.
///
/// For each point `x`, computes the mixture log-likelihood
///   `log sum_j exp(models[j].estimate_log_density(x))`
/// where `estimate_log_density` already includes the mixture weight.
///
/// Low scores correspond to outliers (`ascending=true`).
pub fn em_outlier<N, A, Mo>(data: &A, result: &EmResult<N, Mo>) -> OutlierResult<N>
where
    N: Float,
    A: VectorData<N>,
    Mo: EmModel<N>,
{
    let n = data.nrows();
    let d = data.ncols();
    let k = result.models.len();

    let mut scratch = vec![N::zero(); d];
    let mut log_densities = vec![N::zero(); k];
    let mut scores = vec![N::zero(); n];

    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        for (j, model) in result.models.iter().enumerate() {
            log_densities[j] = model.estimate_log_density(&scratch);
        }
        scores[i] = log_sum_exp(&log_densities);
    }

    // Use the observed max as baseline (typical inlier log-likelihood)
    let baseline = scores.iter().copied().fold(N::neg_infinity(), |a, b| if b > a { b } else { a });

    make_outlier_result(
        scores,
        "EMOutlier",
        true, // ascending: low log-likelihood = outlier
        baseline,
        N::neg_infinity(),
        N::infinity(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::em::{EmResult, MultivariateGaussianModel};
    use ndarray::Array2;

    #[test]
    fn outlier_has_lower_log_likelihood() {
        // Manually construct tight Gaussian models to test em_outlier without
        // depending on EM convergence. Two clusters: A at (0,0), B at (10,0),
        // covariance σ²=0.1 in each dimension.
        let cov = vec![0.1_f64, 0.0, 0.0, 0.1];
        let model_a =
            MultivariateGaussianModel::<f64>::new(0.5, vec![0.0, 0.0], cov.clone(), 1e-10);
        let model_b =
            MultivariateGaussianModel::<f64>::new(0.5, vec![10.0, 0.0], cov, 1e-10);
        let result: EmResult<f64, MultivariateGaussianModel<f64>> = EmResult {
            models: vec![model_a, model_b],
            assignments: vec![0; 9],
            responsibilities: None,
            n_iter: 0,
            log_likelihood: 0.0,
        };

        // 4 points near cluster A, 4 near cluster B, 1 far outlier at index 8
        #[rustfmt::skip]
        let data: &[f64] = &[
             0.0, 0.0,   0.1, 0.0,  -0.1, 0.0,   0.0, 0.1,
            10.0, 0.0,  10.1, 0.0,   9.9, 0.0,  10.0, 0.1,
           500.0, 0.0,
        ];
        let mat = Array2::from_shape_vec((9, 2), data.to_vec()).unwrap();
        let ds = NdArrayDataset::new(&mat);

        let scores = em_outlier(&ds, &result);
        let outlier_score = scores.scores[8];
        let inlier_max = scores.scores[..8].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            outlier_score < inlier_max,
            "outlier log-likelihood {} should be lower than inlier max {}",
            outlier_score,
            inlier_max
        );
    }
}
