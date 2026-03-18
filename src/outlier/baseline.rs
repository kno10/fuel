use num_traits::Float;
use num_traits::FromPrimitive;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::DistanceData;
use crate::api::VectorData; // needed for MatrixDataAccess.point/dims in this file

use super::common::{OutlierScoreEntry, sort_outlier_scores};

/// A very simple outlier score used for trivial baseline detectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaselineOutlierScore<F: Float> {
    pub index: usize,
    pub score: F,
}

impl<F: Float> OutlierScoreEntry<F> for BaselineOutlierScore<F> {
    fn index(&self) -> usize {
        self.index
    }

    fn score(&self) -> F {
        self.score
    }
}

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
pub fn distance_from_origin_outlier_scores<D: VectorData<F>, F: Float + std::iter::Sum>(
    data: D,
) -> Vec<BaselineOutlierScore<F>> {
    let mut scores = Vec::with_capacity(data.size());

    for idx in data.iter() {
        let coords = data.point(idx);
        // Euclidean norm
        let sq = coords.iter().map(|&x| x * x).sum::<F>();
        let score = sq.sqrt();
        scores.push(BaselineOutlierScore { index: idx, score });
    }

    sort_outlier_scores(&mut scores);
    scores
}

/// Outlierness is the distance of each point from the mean of the
/// dataset.
///
/// The centre is computed by averaging each coordinate.  For an empty
/// dataset this function will panic when trying to divide by zero.
pub fn distance_from_center_outlier_scores<
    D: VectorData<F>,
    F: Float + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + std::iter::Sum,
>(
    data: D,
) -> Vec<BaselineOutlierScore<F>> {
    let size = data.size();
    if size == 0 {
        return Vec::new();
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

    let mut scores = Vec::with_capacity(size);
    for idx in data.iter() {
        let coords = data.point(idx);
        let sq: F = coords
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let d = x - centre[i];
                d * d
            })
            .sum();
        let score = sq.sqrt();
        scores.push(BaselineOutlierScore { index: idx, score });
    }

    sort_outlier_scores(&mut scores);
    scores
}

/// Produce random outlier scores in the interval `[0,1)`.  A seed is required
/// to make the results reproducible for benchmarking and testing.
pub fn random_outlier_scores<D: DistanceData<F>, F: Float + FromPrimitive>(
    data: D,
    seed: u64,
) -> Vec<BaselineOutlierScore<F>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut scores = Vec::with_capacity(data.size());

    for idx in data.iter() {
        scores.push(BaselineOutlierScore {
            index: idx,
            score: F::from_f64(rng.r#gen::<f64>()).unwrap_or(F::one()),
        });
    }

    sort_outlier_scores(&mut scores);
    scores
}

/// A completely non-informative detector that assigns a score of zero to every
/// point.  It still returns a sorted list so that the caller observes a
/// deterministic order (indices in ascending order).
pub fn zero_outlier_scores<D: DistanceData<F>, F: Float>(data: D) -> Vec<BaselineOutlierScore<F>> {
    let mut scores = data
        .iter()
        .map(|idx| BaselineOutlierScore {
            index: idx,
            score: F::zero(),
        })
        .collect::<Vec<_>>();
    sort_outlier_scores(&mut scores);
    scores
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::EuclideanDistance;
    use crate::vptree::VPTree;

    fn make_simple_data() -> (
        TableWithDistance<'static, Vec<f64>, EuclideanDistance, f64>,
        VPTree<f64>,
    ) {
        // allocate the backing vector on the heap and leak it so that the
        // returned `MatrixDataAccess` can safely hold a `'static` reference.
        let leaked: &'static mut Vec<Vec<f64>> = Box::leak(Box::new(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![6.0, 6.0],
        ]));
        let data = TableWithDistance::with_distance(leaked, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(42);
        let tree = VPTree::new(&data, 2, &mut rng);
        (data, tree)
    }

    #[test]
    fn origin_ranks_remote_point_highest() {
        let (data, _tree) = make_simple_data();
        let scores = distance_from_origin_outlier_scores(&data);
        assert_eq!(scores[0].index, 4);
        assert!(scores[0].score > scores[1].score);
    }

    #[test]
    fn center_ranks_remote_point_highest() {
        let (data, _tree) = make_simple_data();
        let scores = distance_from_center_outlier_scores(&data);
        assert_eq!(scores[0].index, 4);
        assert!(scores[0].score > scores[1].score);
    }

    #[test]
    fn random_is_reproducible() {
        let (data, _tree) = make_simple_data();
        let s1 = random_outlier_scores(&data, 7);
        let s2 = random_outlier_scores(&data, 7);
        assert_eq!(s1, s2);
        // they should not all be equal
        assert!(s1.iter().any(|e| e.score != 0.0));
    }

    #[test]
    fn zero_scores_behaviour() {
        let (data, _tree) = make_simple_data();
        let scores = zero_outlier_scores(&data);
        assert!(scores.iter().all(|e| e.score == 0.0));
        // sorted by index ascending because all scores equal
        for (i, e) in scores.iter().enumerate() {
            assert_eq!(e.index, i);
        }
    }
}
