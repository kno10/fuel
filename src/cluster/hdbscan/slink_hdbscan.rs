use super::hdbscan_common::{
    HdbscanHierarchy, compute_core_distances, mutual_reachability_distance,
};
use crate::cluster::hierarchical::pointer::{PointerRepresentation, pointer_to_merge_history};
use crate::{DistanceData, Float};

/// Perform HDBSCAN's linear-memory SLINK variant on arbitrary metric data.
///
/// This implements the same update rules as standard SLINK, but replaces the
/// pairwise metric distance with mutual reachability distance.
#[must_use]
pub fn slink_hdbscan_pointer<F: Float, D: DistanceData<F>>(
    data: &D, min_points: usize,
) -> (PointerRepresentation<F>, Vec<F>) {
    let n = data.len();
    assert!(n > 0, "number of points must be positive");
    assert!(min_points > 0, "min_points must be greater than 0");

    let core_distances: Vec<F> = compute_core_distances(data, min_points);

    let mut pi: Vec<usize> = (0..n).collect();
    let mut lambda = vec![F::infinity(); n];
    let mut m = vec![F::infinity(); n];

    for cur in 1..n {
        m[cur] = F::infinity();

        for (i, slot) in m.iter_mut().enumerate().take(cur) {
            *slot = mutual_reachability_distance(&core_distances, cur, i, data.distance(cur, i));
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

#[must_use]
pub fn slink_hdbscan<F: Float, D: DistanceData<F>>(
    data: &D, min_points: usize,
) -> HdbscanHierarchy<F> {
    let (pointer, core_distances) = slink_hdbscan_pointer::<F, _>(data, min_points);
    HdbscanHierarchy::new(pointer_to_merge_history(&pointer), core_distances)
}

#[cfg(test)]
mod tests {
    use super::slink_hdbscan;
    use crate::TableWithDistance;
    use crate::cluster::hdbscan::HdbscanHierarchy;
    use crate::distance::Euclidean;

    #[test]
    fn slink_hdbscan_produces_complete_hierarchy() {
        let points =
            vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![3.0, 3.0], vec![3.2, 3.1], vec![10.0, 10.0]];

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let result: HdbscanHierarchy<f64> = slink_hdbscan(&data, 2);

        assert_eq!(result.core_distances.len(), points.len());
        assert_eq!(result.merges.len(), points.len() - 1);
        assert_eq!(result.merges.last().expect("non-empty").size, points.len());
    }
}
