/// K-means distance-based outlier detection.
///
/// Reference: He, Xu, Deng - *Discovering cluster-based local outliers*,
/// Pattern Recognition Letters 24(9-10), 2003.
///
/// Scores each point by its (squared Euclidean) distance to its assigned cluster
/// center, optionally adjusted for cluster size.  Three scoring rules are
/// provided to handle singleton clusters differently.
use crate::Float;
use crate::VectorData;
use crate::cluster::kmeans::util::KMeansResult;
use crate::math;
use crate::outlier::common::{OutlierResult, make_outlier_result};

/// Scoring rule for [`kmeans_outlier`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KMeansOutlierRule {
    /// Score = squared distance to own cluster center.
    /// Singleton clusters (size 1) score 0 because their center equals the
    /// point.
    Distance,
    /// Score = squared distance to own cluster center.
    /// Singleton clusters are treated as suspicious and score the minimum
    /// squared distance to any cluster center (including non-singletons).
    DistanceSingletons,
    /// Score = squared distance * size / (size - 1).
    /// This inflates scores for smaller clusters, penalising their members
    /// more heavily.  Singletons use min_dist * 2 (analogous to size=2
    /// in the size/(size-1) formula applied to the nearest other center).
    Variance,
}

/// K-means distance-based outlier detection.
///
/// Takes a precomputed `result` from any k-means variant and scores each point
/// according to `rule`.  Distances are squared Euclidean, consistent with the
/// k-means objective.
///
/// Returns scores with `ascending=false` (higher score = more outlier-like).
pub fn kmeans_outlier<D, F>(
    data: &D, result: &KMeansResult<F>, rule: KMeansOutlierRule,
) -> OutlierResult<F>
where
    F: Float,
    D: VectorData<F>,
{
    let n = data.nrows();
    let d = data.ncols();
    let k = result.centers.nrows();

    // Compute cluster sizes
    let mut sizes = vec![0usize; k];
    for &a in &result.assignments {
        sizes[a] += 1;
    }

    let mut scratch = vec![F::zero(); d];
    let mut scores = vec![F::zero(); n];

    // Pre-compute centers as owned vecs (row() returns a temporary view)
    let centers: Vec<Vec<F>> = (0..k).map(|j| result.centers.row(j).to_vec()).collect();

    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        let a = result.assignments[i];
        let center_a = &centers[a];

        scores[i] = match rule {
            KMeansOutlierRule::Distance => {
                math::sqdist(center_a, &scratch, d)
            }
            KMeansOutlierRule::DistanceSingletons => {
                if sizes[a] == 1 {
                    // min distance to any center
                    centers
                        .iter()
                        .map(|c| math::sqdist(c, &scratch, d))
                        .fold(F::infinity(), |acc, v| if v < acc { v } else { acc })
                } else {
                    math::sqdist(center_a, &scratch, d)
                }
            }
            KMeansOutlierRule::Variance => {
                let sz = sizes[a];
                if sz == 1 {
                    // Find nearest other center; use factor 1/2 ~ size/(size-1) at size=2
                    let min_dist = centers
                        .iter()
                        .map(|c| math::sqdist(c, &scratch, d))
                        .fold(F::infinity(), |acc, v| if v < acc { v } else { acc });
                    // size=1 treated as size=2 for the ratio: factor = 2/1 = 2
                    min_dist * F::from_usize(2).unwrap()
                } else {
                    let dist = math::sqdist(center_a, &scratch, d);
                    let sz_f = F::from_usize(sz).unwrap();
                    dist * sz_f / (sz_f - F::one())
                }
            }
        };
    }

    make_outlier_result(scores, "KMeansOutlier", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::lloyd;
    use crate::cluster::kmeans::init::RandomSample;
    use rand::SeedableRng;
    use rand_pcg::Pcg32;
    use ndarray::Array2;

    fn make_data() -> Array2<f64> {
        // Two tight clusters at 0 and 10, plus an outlier at 5
        let mut rows: Vec<f64> = Vec::new();
        for i in 0..8 { rows.push(i as f64 * 0.05); rows.push(0.0); }
        for i in 0..8 { rows.push(10.0 + i as f64 * 0.05); rows.push(0.0); }
        rows.push(5.0); rows.push(0.0); // outlier at index 16
        Array2::from_shape_vec((17, 2), rows).unwrap()
    }

    #[test]
    fn distance_rule_outlier_scores_higher() {
        let mat = make_data();
        let ds = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        let result = lloyd::<f64, _, _>(&ds, 2, &mut init, 100, 1e-10).unwrap();
        let scores = kmeans_outlier(&ds, &result, KMeansOutlierRule::Distance);
        let outlier_score = scores.scores[16];
        let cluster_a_max = (0..8).map(|i| scores.scores[i]).fold(0.0f64, f64::max);
        let cluster_b_max = (8..16).map(|i| scores.scores[i]).fold(0.0f64, f64::max);
        assert!(
            outlier_score > cluster_a_max && outlier_score > cluster_b_max,
            "outlier={outlier_score} a_max={cluster_a_max} b_max={cluster_b_max}"
        );
    }

    #[test]
    fn variance_rule_penalises_small_clusters_more() {
        let mat = make_data();
        let ds = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        let result = lloyd::<f64, _, _>(&ds, 2, &mut init, 100, 1e-10).unwrap();
        let dist_scores = kmeans_outlier(&ds, &result, KMeansOutlierRule::Distance);
        let var_scores = kmeans_outlier(&ds, &result, KMeansOutlierRule::Variance);
        // Variance scores should be >= Distance scores for the same point
        // (multiplier >= 1 for any cluster of size >= 2)
        for i in 0..16 {
            assert!(var_scores.scores[i] >= dist_scores.scores[i] - 1e-12);
        }
    }
}
