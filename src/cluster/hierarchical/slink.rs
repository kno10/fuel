use super::common::MergeHistory;
use super::pointer::{PointerRepresentation, pointer_to_merge_history};
use crate::{DistanceData, Float};

// Version using the original "pointer" representation
pub fn slink_pointer<F: Float, D: DistanceData<F>>(data: &D) -> PointerRepresentation<F> {
    let n = data.size();
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

    PointerRepresentation::new(pi, lambda)
}

/// Convenience wrapper returning the common merge history format.
#[must_use]
pub fn slink<F: Float, D: DistanceData<F>>(data: &D) -> MergeHistory<F> {
    pointer_to_merge_history(&slink_pointer(data))
}

#[cfg(test)]
mod tests {
    use super::super::pointer::pointer_to_merge_history;
    use super::slink_pointer;
    use crate::cluster::hierarchical::{SingleLinkage, agnes};
    use crate::data::CondensedDistanceMatrix;

    #[test]
    fn slink_matches_agnes_single_on_unique_distances() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let cm = CondensedDistanceMatrix::new(&d, 5);
        let a = agnes(&d, 5, SingleLinkage, false);
        let b = pointer_to_merge_history(&slink_pointer(&cm));
        assert_eq!(a, b);
    }

    #[test]
    fn slink_pointer_has_valid_shape() {
        let d = vec![1.0, 3.0, 8.0, 2.0, 7.0, 6.0];
        let cm = CondensedDistanceMatrix::new(&d, 4);
        let p = slink_pointer(&cm);
        assert_eq!(p.pi.len(), 4);
        assert_eq!(p.lambda.len(), 4);
    }
}
