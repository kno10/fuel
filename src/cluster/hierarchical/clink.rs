use crate::cluster::hierarchical::MergeHistory;
use crate::cluster::hierarchical::pointer::{PointerRepresentation, pointer_to_merge_history};
use crate::{DistanceData, Float};

// Original CLINK uses the same pointer format as SLINK.
pub fn clink_pointer<F: Float, D: DistanceData<F>>(data: &D) -> PointerRepresentation<F> {
    let n = data.len();
    assert!(n > 0, "number of points must be positive");

    let mut pi: Vec<usize> = (0..n).collect();
    let mut lambda = vec![F::infinity(); n];
    let mut m = vec![F::infinity(); n];

    for cur in 1..n {
        m[cur] = F::infinity();

        for (i, slot) in m.iter_mut().enumerate().take(cur) {
            *slot = data.distance(cur, i);
        }

        for i in 0..cur {
            let m_i = m[i];
            if lambda[i] < m_i {
                let p_i = pi[i];
                if m[p_i] < m_i {
                    m[p_i] = m_i;
                }
                m[i] = F::infinity();
            }
        }

        let mut a = cur - 1;
        for i in (0..cur).rev() {
            let p_i = pi[i];
            let mp_i = m[p_i];
            if lambda[i] >= mp_i {
                if m[i] < m[a] {
                    a = i;
                }
            } else {
                m[i] = F::infinity();
            }
        }

        let mut b = pi[a];
        let mut c = lambda[a];
        pi[a] = cur;
        lambda[a] = m[a];

        if a < cur - 1 {
            let last = cur - 1;
            while b != cur {
                if b == last {
                    pi[b] = cur;
                    lambda[b] = c;
                    break;
                }
                let d = pi[b];
                pi[b] = cur;
                std::mem::swap(&mut lambda[b], &mut c);
                b = d;
            }
        }

        for i in 0..cur {
            let p_i = pi[i];
            let pp_i = pi[p_i];
            if pp_i == cur && lambda[i] >= lambda[p_i] {
                pi[i] = cur;
            }
        }
    }

    PointerRepresentation::new(pi, lambda)
}

/// Perform CLINK and convert to merge history (generic version).
#[must_use]
pub fn clink<F: Float, D: DistanceData<F>>(data: &D) -> MergeHistory<F> {
    pointer_to_merge_history(&clink_pointer(data))
}

#[cfg(test)]
mod tests {
    use super::{clink, clink_pointer};
    use crate::CondensedDistanceMatrix;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::pointer::pointer_to_merge_history;
    use crate::cluster::hierarchical::test::test_clustering_table;
    use crate::cluster::hierarchical::{CompleteLinkage, agnes};

    #[test]
    fn clink_complete_regression() {
        test_clustering_table(
            "CLINK",
            "clink",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let cm = CondensedDistanceMatrix::new_from_data(access);
                let history = clink(&cm);
                cut_dendrogram_by_number_of_clusters(&history, min_clusters)
            },
        );
    }

    #[test]
    fn clink_runs_and_merges_all_points() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let binding = d.clone();
        let cm = CondensedDistanceMatrix::new_from_condensed(binding, 5, false);
        let h = pointer_to_merge_history(&clink_pointer(&cm));
        assert_eq!(h.len(), 4);
        assert_eq!(h.last().expect("non-empty history").size, 5);
    }

    #[test]
    fn clink_and_agnes_complete_agree_on_simple_chain_case() {
        // For this strictly chain-like case with increasing gaps both methods agree.
        let d = vec![1.0, 3.0, 8.0, 2.0, 7.0, 6.0];
        let cm_a = CondensedDistanceMatrix::new_from_condensed(d.clone(), 4, false);
        let a = agnes(&cm_a, CompleteLinkage);
        let binding = d.clone();
        let cm = CondensedDistanceMatrix::new_from_condensed(binding, 4, false);
        let c = pointer_to_merge_history(&clink_pointer(&cm));
        assert_eq!(a, c);
    }

    #[test]
    fn clink_pointer_has_valid_shape() {
        let d = vec![1.0, 3.0, 8.0, 2.0, 7.0, 6.0];
        let binding = d.clone();
        let cm = CondensedDistanceMatrix::new_from_condensed(binding, 4, false);
        let p = clink_pointer(&cm);
        assert_eq!(p.pi.len(), 4);
        assert_eq!(p.lambda.len(), 4);
    }
}
