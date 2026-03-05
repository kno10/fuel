use num_traits::Float;

use super::common::MergeHistory;
use super::pointer::{PointerRepresentation, pointer_to_merge_history};

/// Perform CLINK complete-link hierarchical clustering in `O(n^2)` time and
/// `O(n)` memory, returning native pointer representation.
///
/// This algorithm is order-dependent and can differ from standard AGNES
/// complete linkage, matching Defays CLINK behavior.
#[must_use]
pub fn clink_pointer<F: Float>(distances: &[F], n: usize) -> PointerRepresentation<F> {
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
            if lambda[i] < m[i] {
                let p_i = pi[i];
                if m[p_i] < m[i] {
                    m[p_i] = m[i];
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
                let old = lambda[b];
                lambda[b] = c;
                c = old;
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

/// Perform CLINK and convert to merge history.
#[must_use]
pub fn clink<F: Float>(distances: &[F], n: usize) -> MergeHistory<F> {
    pointer_to_merge_history(&clink_pointer(distances, n))
}

#[inline]
fn dist<F: Float>(mat: &[F], i: usize, j: usize) -> F {
    let (a, b) = if i > j { (i, j) } else { (j, i) };
    mat[(a * (a - 1)) / 2 + b]
}

#[cfg(test)]
mod tests {
    use crate::cluster::hierarchical::agnes;
    use crate::cluster::hierarchical::linkage::CompleteLinkage;

    use super::{clink, clink_pointer};

    #[test]
    fn clink_runs_and_merges_all_points() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let h = clink(&d, 5);
        assert_eq!(h.len(), 4);
        assert_eq!(h.last().expect("non-empty history").size, 5);
    }

    #[test]
    fn clink_and_agnes_complete_agree_on_simple_chain_case() {
        // For this strictly chain-like case with increasing gaps both methods agree.
        let d = vec![1.0, 3.0, 8.0, 2.0, 7.0, 6.0];
        let a = agnes(&d, 4, CompleteLinkage, false);
        let c = clink(&d, 4);
        assert_eq!(a, c);
    }

    #[test]
    fn clink_pointer_has_valid_shape() {
        let d = vec![1.0, 3.0, 8.0, 2.0, 7.0, 6.0];
        let p = clink_pointer(&d, 4);
        assert_eq!(p.pi.len(), 4);
        assert_eq!(p.lambda.len(), 4);
    }
}
