use crate::DataAccess;

use super::hdbscan_common::{
    HdbscanHierarchy, compute_core_distances, mutual_reachability_distance,
};
use super::pointer::{PointerRepresentation, pointer_to_merge_history};

/// Perform HDBSCAN's linear-memory SLINK variant on arbitrary metric data.
///
/// This implements the same update rules as standard SLINK, but replaces the
/// pairwise metric distance with mutual reachability distance.
#[must_use]
pub fn slink_hdbscan_linear_memory_pointer<D: DataAccess>(
    data: &D,
    min_points: usize,
) -> (PointerRepresentation<f64>, Vec<f64>) {
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    assert!(min_points > 0, "min_points must be greater than 0");

    let core_distances = compute_core_distances(data, min_points);

    let mut pi: Vec<usize> = (0..n).collect();
    let mut lambda = vec![f64::INFINITY; n];
    let mut m = vec![f64::INFINITY; n];

    for cur in 1..n {
        m[cur] = f64::INFINITY;

        for (i, slot) in m.iter_mut().enumerate().take(cur) {
            *slot = mutual_reachability_distance(data, &core_distances, cur, i);
        }

        for i in 0..cur {
            let l_i = lambda[i];
            let m_i = m[i];
            let p_i = pi[i];
            let m_p = m[p_i];

            if l_i >= m_i {
                if l_i < m_p {
                    m[p_i] = l_i;
                }
                lambda[i] = m_i;
                pi[i] = cur;
            } else if m_i < m_p {
                m[p_i] = m_i;
            }
        }

        for i in 0..cur {
            let p_i = pi[i];
            if lambda[i] >= lambda[p_i] {
                pi[i] = cur;
            }
        }
    }

    (PointerRepresentation::new(pi, lambda), core_distances)
}

/// Perform HDBSCAN's linear-memory SLINK variant and return merge history and
/// per-point core distances.
#[must_use]
pub fn slink_hdbscan_linear_memory<D: DataAccess>(data: &D, min_points: usize) -> HdbscanHierarchy {
    let (pointer, core_distances) = slink_hdbscan_linear_memory_pointer(data, min_points);
    let merges = pointer_to_merge_history(&pointer);
    HdbscanHierarchy::new(merges, core_distances)
}

#[cfg(test)]
mod tests {
    use crate::{EuclideanDistance, MatrixDataAccess};

    use super::slink_hdbscan_linear_memory;

    #[test]
    fn slink_hdbscan_produces_complete_hierarchy() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![3.0, 3.0],
            vec![3.2, 3.1],
            vec![10.0, 10.0],
        ];

        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let result = slink_hdbscan_linear_memory(&data, 2);

        assert_eq!(result.core_distances.len(), points.len());
        assert_eq!(result.merges.len(), points.len() - 1);
        assert_eq!(result.merges.last().expect("non-empty").size, points.len());
    }
}
