use num_traits::Float;

use super::common::MergeHistory;
use super::pointer::{PointerRepresentation, pointer_to_merge_history};

/// Perform SLINK single-link hierarchical clustering in `O(n^2)` time and
/// `O(n)` memory, returning its native pointer representation.
#[must_use]
pub fn slink_pointer<F: Float>(distances: &[F], n: usize) -> PointerRepresentation<F> {
    assert!(n > 0, "number of points must be positive");
    assert_eq!(
        distances.len(),
        n * (n - 1) / 2,
        "bad condensed matrix length"
    );

    let mut pi: Vec<usize> = (0..n).collect();
    let mut lambda = vec![F::infinity(); n];
    let mut m = vec![F::infinity(); n];

    for cur in 1..n {
        m[cur] = F::infinity();

        for i in 0..cur {
            m[i] = dist(distances, cur, i);
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

/// Perform SLINK and immediately convert its pointer representation to merge
/// history.
#[must_use]
pub fn slink<F: Float>(distances: &[F], n: usize) -> MergeHistory<F> {
    pointer_to_merge_history(&slink_pointer(distances, n))
}

#[inline]
fn dist<F: Float>(mat: &[F], i: usize, j: usize) -> F {
    let (a, b) = if i > j { (i, j) } else { (j, i) };
    mat[(a * (a - 1)) / 2 + b]
}

#[cfg(test)]
mod tests {
    use crate::cluster::hierarchical::agnes;
    use crate::cluster::hierarchical::linkage::SingleLinkage;

    use super::{slink, slink_pointer};

    #[test]
    fn slink_matches_agnes_single_on_unique_distances() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let a = agnes(&d, 5, SingleLinkage, false);
        let b = slink(&d, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn slink_pointer_has_valid_shape() {
        let d = vec![1.0, 3.0, 8.0, 2.0, 7.0, 6.0];
        let p = slink_pointer(&d, 4);
        assert_eq!(p.pi.len(), 4);
        assert_eq!(p.lambda.len(), 4);
    }
}
