use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::api::VectorData; // needed for MatrixDataAccess.point/dims in this file
use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float};

/// A very simple outlier score used for trivial baseline detectors.
///
/// Outlierness is the Euclidean distance of each point from the origin.
///
/// This only makes sense for vector data where the feature values are
/// floating point numbers; the implementation therefore requires that the
/// underlying data type implements `AsRef<[f64]>`.  For other formats the
/// user should convert to a numeric matrix first.
///
/// # Panics
///
/// Panics if the data set is empty (division by zero when computing the mean).
pub fn distance_from_origin<D: VectorData<F> + Sync, F: Float + Send + Sync + std::iter::Sum>(
    data: D,
) -> OutlierResult<F> {
    let scores: Vec<F> = if cfg!(feature = "parallel") {
        use rayon::prelude::*;
        (0..data.len())
            .into_par_iter()
            .map(|idx| {
                let coords = data.point(idx);
                let sq = coords.iter().map(|&x| x * x).sum::<F>();
                sq.sqrt()
            })
            .collect()
    } else {
        data.iter()
            .map(|idx| {
                let coords = data.point(idx);
                let sq = coords.iter().map(|&x| x * x).sum::<F>();
                sq.sqrt()
            })
            .collect()
    };

    make_outlier_result(scores, "Distance from origin", false, F::zero(), F::zero(), F::infinity())
}

/// Outlierness is the distance of each point from the mean of the
/// dataset.
///
/// The centre is computed by averaging each coordinate.  For an empty
/// dataset this function will panic when trying to divide by zero.
pub fn distance_from_center<
    D: VectorData<F> + Sync,
    F: Float + Send + Sync + std::ops::AddAssign + std::ops::DivAssign + std::iter::Sum,
>(
    data: D,
) -> OutlierResult<F> {
    let size = data.len();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "Distance from center",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let dims = data.dims();
    let mut centre = vec![F::zero(); dims];

    for idx in data.iter() {
        let coords = data.point(idx);
        for i in 0..dims {
            centre[i] += coords[i];
        }
    }
    for x in &mut centre {
        *x /= F::from_usize(size).unwrap_or(F::one());
    }

    let scores: Vec<F> = data
        .iter()
        .map(|idx| {
            let coords = data.point(idx);
            let sq: F = coords
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let d = x - centre[i];
                    d * d
                })
                .sum();
            sq.sqrt()
        })
        .collect();

    make_outlier_result(scores, "Distance from center", false, F::zero(), F::zero(), F::infinity())
}

/// Produce random outlier scores in the interval `[0,1)`.  A seed is required
/// to make the results reproducible for benchmarking and testing.
pub fn random<D: DistanceData<F>, F: Float>(data: D, seed: u64) -> OutlierResult<F> {
    let mut rng = StdRng::seed_from_u64(seed);

    let scores: Vec<F> =
        data.iter().map(|_idx| F::from_f64(rng.r#gen::<f64>()).unwrap_or(F::one())).collect();

    make_outlier_result(scores, "Random outlier score", false, F::zero(), F::zero(), F::infinity())
}

/// A completely non-informative detector that assigns a score of zero to every
/// point.  It still returns a sorted list so that the caller observes a
/// deterministic order (indices in ascending order).
pub fn zero<D: DistanceData<F> + Sync, F: Float + Send + Sync>(data: D) -> OutlierResult<F> {
    let scores: Vec<F> = if cfg!(feature = "parallel") {
        use rayon::prelude::*;
        (0..data.len()).into_par_iter().map(|_| F::zero()).collect()
    } else {
        data.iter().map(|_| F::zero()).collect()
    };

    make_outlier_result(scores, "Random outlier score", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::api::Data;
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;

    fn make_simple_data() -> (TableWithDistance<'static, f64, Vec<f64>, Euclidean, f64>, VPTree<f64>)
    {
        // allocate the backing vector on the heap and leak it so that the
        // returned `MatrixDataAccess` can safely hold a `'static` reference.
        let leaked: &'static mut Vec<Vec<f64>> = Box::leak(Box::new(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![6.0, 6.0],
        ]));
        let data = TableWithDistance::with_distance(leaked, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree = VPTree::new(&data, 2, &mut rng);
        (data, tree)
    }

    #[test]
    fn origin_ranks_remote_point_highest() {
        let (data, _tree) = make_simple_data();
        let results = distance_from_origin(&data);

        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        assert_eq!(best_index, 4);
    }

    #[test]
    fn center_ranks_remote_point_highest() {
        let (data, _tree) = make_simple_data();
        let results = distance_from_center(&data);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, 4);
    }

    #[test]
    fn random_is_reproducible() {
        let (data, _tree) = make_simple_data();
        let s1 = random(&data, 7);
        let s2 = random(&data, 7);
        assert_eq!(s1, s2);
        // they should not all be equal
        assert!(s1.scores.iter().any(|&e| e != 0.0));
    }

    #[test]
    fn zero_scores_behaviour() {
        let (data, _tree) = make_simple_data();
        let scores = zero(&data);
        assert!(scores.scores.iter().all(|&e| e == 0.0));
        assert_eq!(scores.scores.len(), data.len());
    }
}
