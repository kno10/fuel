/// Outlier detection based on k-means-- (truncated k-means / tkmeans).
///
/// Reference: Chawla, Gionis - *k-means--: A Unified Approach to Clustering and
/// Outlier Detection*, SDM 2013.
///
/// Reference (star variant): Vinces, Schubert, Zimek, Cordeiro -
/// *A comparative evaluation of clustering-based outlier detection*, DMKD 2025.
///
/// Both functions require a [`KMeansResult`] produced by [`tkmeans`] (which
/// populates `noise_mask`).  Calling them with a result that has no `noise_mask`
/// returns an error.
///
/// [`tkmeans`]: crate::cluster::kmeans::tkmeans
use crate::Float;
use crate::cluster::kmeans::util::KMeansResult;
use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{VectorData, math};

/// Binary outlier labeling from k-means-- (KMeansMinusMinusOutlierDetection).
///
/// Trimmed (noise) points receive score 1.0; cluster members receive 0.0.
///
/// Returns an error if `result.noise_mask` is `None`.
pub fn tkmeans_outlier<F: Float>(result: &KMeansResult<F>) -> Result<OutlierResult<F>, String> {
    let mask = result.noise_mask.as_ref().ok_or_else(|| {
        "tkmeans_outlier requires a KMeansResult with a noise_mask; use tkmeans()".to_string()
    })?;

    let scores: Vec<F> =
        mask.iter().map(|&noise| if noise { F::one() } else { F::zero() }).collect();

    Ok(make_outlier_result(scores, "tkmeans", false, F::zero(), F::zero(), F::one()))
}

/// Continuous outlier scoring from k-means--* (KMeansMinusMinusStar).
///
/// - Trimmed (noise) points: squared distance to the nearest cluster center.
/// - Cluster members: squared distance to their assigned center.
///
/// Returns an error if `result.noise_mask` is `None`.
pub fn tkmeans_star_outlier<D, F>(
    data: &D, result: &KMeansResult<F>,
) -> Result<OutlierResult<F>, String>
where
    F: Float,
    D: VectorData<F>,
{
    let mask = result.noise_mask.as_ref().ok_or_else(|| {
        "tkmeans_star_outlier requires a KMeansResult with a noise_mask; use tkmeans()".to_string()
    })?;

    let n = data.nrows();
    let d = data.ncols();
    let k = result.centers.nrows();

    // Copy centers into owned vecs (row() returns a temporary view)
    let centers: Vec<Vec<F>> = (0..k).map(|j| result.centers.row(j).to_vec()).collect();

    let mut scratch = vec![F::zero(); d];
    let mut scores = vec![F::zero(); n];

    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        if mask[i] {
            // Noise point: minimum squared distance to any cluster center
            scores[i] = centers
                .iter()
                .map(|c| math::sqdist(c, &scratch, d))
                .fold(F::infinity(), |acc, v| if v < acc { v } else { acc });
        } else {
            let a = result.assignments[i];
            scores[i] = math::sqdist(&centers[a], &scratch, d);
        }
    }

    Ok(make_outlier_result(scores, "tkmeans*", false, F::zero(), F::zero(), F::infinity()))
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::init::RandomSample;
    use crate::cluster::kmeans::{lloyd, tkmeans};

    fn make_data() -> Array2<f64> {
        // 8 points near 0, 8 near 10, 1 outlier at 5
        let mut rows: Vec<f64> = Vec::new();
        for i in 0..8 {
            rows.push(i as f64 * 0.05);
            rows.push(0.0);
        }
        for i in 0..8 {
            rows.push(10.0 + i as f64 * 0.05);
            rows.push(0.0);
        }
        rows.push(5.0);
        rows.push(0.0); // index 16
        Array2::from_shape_vec((17, 2), rows).unwrap()
    }

    #[test]
    fn binary_labels_noise_point_as_outlier() {
        let mat = make_data();
        let ds = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        // alpha=1.0 (absolute) trims exactly 1 point
        let result = tkmeans(&ds, 2, &mut init, 200, 1e-10, 1.0_f32).unwrap();
        let scores = tkmeans_outlier(&result).unwrap();
        // Exactly one point should have score 1 (the trimmed point)
        let n_outliers = scores.scores.iter().filter(|&&s| s == 1.0).count();
        assert_eq!(n_outliers, 1);
    }

    #[test]
    fn star_noise_point_score_higher_than_cluster_members() {
        let mat = make_data();
        let ds = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        let result = tkmeans(&ds, 2, &mut init, 200, 1e-10, 1.0_f32).unwrap();
        let scores = tkmeans_star_outlier(&ds, &result).unwrap();
        let mask = result.noise_mask.as_ref().unwrap();
        let noise_idx = mask.iter().position(|&m| m).expect("exactly one noise point");
        let max_cluster_score = scores
            .scores
            .iter()
            .enumerate()
            .filter(|&(i, _)| !mask[i])
            .map(|(_, &s)| s)
            .fold(0.0f64, f64::max);
        assert!(
            scores.scores[noise_idx] > max_cluster_score,
            "noise score {} should exceed cluster max {}",
            scores.scores[noise_idx],
            max_cluster_score
        );
    }

    #[test]
    fn error_on_missing_noise_mask() {
        let mat = make_data();
        let ds = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        let result = lloyd::<f64, _, _>(&ds, 2, &mut init, 100, 1e-10).unwrap();
        assert!(result.noise_mask.is_none());
        assert!(tkmeans_outlier(&result).is_err());
        assert!(tkmeans_star_outlier(&ds, &result).is_err());
    }
}
