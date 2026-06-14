/// Cluster-based local outlier factor (CBLOF).
///
/// Reference: He, Xu, Deng - *Discovering cluster-based local outliers*,
/// Pattern Recognition Letters 24(9-10), 2003.
///
/// Clusters are split into "large" and "small" groups using the cumulative-size
/// threshold `alpha` and the consecutive-size-ratio threshold `beta`.  Members
/// of large clusters are scored by their distance to their own center; members
/// of small clusters are scored by their distance to the nearest large cluster
/// center.  All distances are squared Euclidean, consistent with the k-means
/// objective; the score is further multiplied by the point's cluster size.
use crate::Float;
use crate::cluster::kmeans::util::KMeansResult;
use crate::outlier::common::OutlierResult;
use crate::{ParMap, VectorData, math};

/// Cluster-based local outlier factor.
///
/// `alpha` - fraction of the data that must be covered by the "large" clusters
///   (e.g. 0.9).  Range (0, 1).
/// `beta`  - minimum size ratio between two consecutive clusters at the
///   large/small boundary (e.g. 3.0).
///
/// The function determines the boundary between large and small clusters by
/// iterating through the clusters sorted by size (descending) and stopping as
/// soon as either condition is satisfied:
///   - cumulative size of seen clusters >= n * alpha
///   - size of current cluster >= beta * size of next cluster
///
/// Returns scores with `ascending=false` (higher score = more outlier-like).
pub fn cblof<D, F>(data: &D, result: &KMeansResult<F>, alpha: F, beta: F) -> OutlierResult<F>
where
    F: Float,
    D: VectorData<F> + Sync,
{
    let n = data.nrows();
    let d = data.ncols();
    let k = result.centers.nrows();

    // Compute cluster sizes
    let mut sizes = vec![0usize; k];
    for &a in &result.assignments {
        sizes[a] += 1;
    }

    // Sort cluster indices by size descending
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]));

    // Determine large/small boundary
    let threshold = F::from_usize(n).unwrap() * alpha;
    let mut cum_size = 0usize;
    let mut boundary = order.len() - 1; // default: all large
    for i in 0..order.len() {
        cum_size += sizes[order[i]];
        let ratio_exceeded = i + 1 < order.len()
            && F::from_usize(sizes[order[i]]).unwrap()
                >= beta * F::from_usize(sizes[order[i + 1]]).unwrap();
        if F::from_usize(cum_size).unwrap() >= threshold || ratio_exceeded {
            boundary = i;
            break;
        }
    }

    // Mark which original cluster ids are "large"
    let mut is_large = vec![false; k];
    for &idx in &order[..=boundary] {
        is_large[idx] = true;
    }

    // Collect large cluster centers as owned vecs (row() returns a temporary view)
    let large_centers: Vec<Vec<F>> =
        order[..=boundary].iter().map(|&j| result.centers.row(j).to_vec()).collect();
    // All centers for large-cluster scoring
    let all_centers: Vec<Vec<F>> = (0..k).map(|j| result.centers.row(j).to_vec()).collect();

    let scores: Vec<F> = (0..n).par_map(|i| {
        let mut scratch = vec![F::zero(); d];
        data.load_into(i, &mut scratch, d);
        let a = result.assignments[i];
        let sz = F::from_usize(sizes[a]).unwrap();

        if is_large[a] {
            sz * math::sqdist(&all_centers[a], &scratch, d)
        } else {
            // minimum squared distance to any large cluster center
            let min_dist = large_centers
                .iter()
                .map(|c| math::sqdist(c, &scratch, d))
                .fold(F::infinity(), |acc, v| if v < acc { v } else { acc });
            sz * min_dist
        }
    });

    OutlierResult::builder(scores, "CBLOF")
        .baseline(F::one())
        .theoretical_min(F::zero())
        .theoretical_max(F::infinity())
        .build()
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::init::RandomSample;
    use crate::cluster::kmeans::lloyd;

    fn make_data() -> Array2<f64> {
        // 10 points in cluster A (near 0), 10 in cluster B (near 10),
        // 1 isolated outlier at 5
        let mut rows: Vec<f64> = Vec::new();
        for i in 0..10 {
            rows.push(i as f64 * 0.05);
            rows.push(0.0);
        }
        for i in 0..10 {
            rows.push(10.0 + i as f64 * 0.05);
            rows.push(0.0);
        }
        rows.push(5.0);
        rows.push(0.0); // index 20
        Array2::from_shape_vec((21, 2), rows).unwrap()
    }

    #[test]
    fn outlier_scores_higher_than_inliers() {
        let mat = make_data();
        let ds = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        let result = lloyd::<f64, _, _>(&ds, 2, &mut init, 200, 1e-12).unwrap();
        let scores = cblof(&ds, &result, 0.9, 3.0);
        let outlier_score = scores.scores[20];
        let inlier_max = (0..20).map(|i| scores.scores[i]).fold(0.0f64, f64::max);
        assert!(
            outlier_score > inlier_max,
            "outlier score {outlier_score} should exceed inlier max {inlier_max}"
        );
    }

    #[test]
    fn all_scores_finite_and_nonnegative() {
        // k=3: two large clusters, one tiny cluster of 1. The lonely point
        // should be scored against the nearest large cluster.
        let mat = make_data();
        let ds = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(7));
        let result = lloyd::<f64, _, _>(&ds, 3, &mut init, 200, 1e-12).unwrap();
        let scores = cblof(&ds, &result, 0.9, 3.0);
        for &s in &scores.scores {
            assert!(s.is_finite() && s >= 0.0_f64);
        }
    }
}
