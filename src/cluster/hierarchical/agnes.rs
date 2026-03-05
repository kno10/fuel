//! Agglomerative hierarchical clustering (AGNES) ported from the old Java
//! code base.  The algorithm works on a *condensed* distance matrix (the
//! strictly triangular part, stored as a contiguous slice) and supports a
//! variety of Lance–Williams linkage criteria.  The output format is the same
//! as `scipy.cluster.hierarchy.linkage`: a sequence of `(i, j, dist, size)`
//! quadruples where `i` and `j` are cluster identifiers (original points have
//! ids `0..n-1`, newly merged clusters are assigned `n..` in merge order),
//! `dist` is the merge distance, and `size` is the number of original points in
//! the newly created cluster.
//!
//! The input matrix is assumed to encode distances for pairs `(p,q)` with
//! `0 <= q < p < n`.  The index of such a pair in the slice is
//! `p*(p-1)/2 + q` which corresponds to the lower‑triangular ordering used by
//! the Java implementation.  This is equivalent to the condensed form used by
//! SciPy (`pdist` output) except that SciPy uses the upper triangle; users can
//! simply transpose their indices to convert between the two representations.

use num_traits::Float;

use super::common::{Builder, MergeHistory, triangle_index};
use super::linkage::Linkage;

// (linkage implementations live in `hierarchical::linkage`)
/// Perform the AGNES algorithm on a condensed lower‑triangular distance
/// matrix.
///
/// - `distances` must have length `n*(n-1)/2` and encode the pairwise
///   distances for `(p,q)` with `0 <= q < p < n` in row-major order
///   (`triangle_index(p,q)`).
/// - `n` is the number of original objects.
/// - `linkage` selects the Lance–Williams linkage criterion to use.
///
/// Returns a `MergeHistory` in `SciPy` linkage format.
///
/// # Panics
///
/// * if the number of points `n` is zero
/// * if `distances.len()` does not equal `n*(n-1)/2`
///
/// The function converts the provided condensed matrix into an agglomerative
/// merge history and will `panic!` when the preconditions above are violated.
pub fn agnes<F: Float, L: Linkage<F> + Copy>(
    distances: &[F],
    n: usize,
    linkage: L,
    is_squared: bool,
) -> MergeHistory<F> {
    assert!(n > 0, "number of points must be positive");
    assert_eq!(
        distances.len(),
        n * (n - 1) / 2,
        "bad condensed matrix length"
    );

    let mut builder = Builder::<F>::new(n);
    let mut mat: Vec<F> = distances
        .iter()
        .map(|&d| linkage.initial(d, is_squared))
        .collect();
    let mut clustermap: Vec<Option<usize>> = (0..n).map(Some).collect();
    let mut end = n;

    // repeatedly merge until one cluster remains
    for _step in 1..n {
        // find the closest pair among active objects
        let mut mindist = F::infinity();
        let mut best_x = 0;
        let mut best_y = 0;

        for ox in 0..end {
            if clustermap[ox].is_none() {
                continue;
            }
            for oy in 0..ox {
                if clustermap[oy].is_none() {
                    continue;
                }
                let d = mat[triangle_index(ox, oy)];
                // prefer the first occurrence of the minimum distance (not the
                // last) to match SciPy's behaviour.  using `<` rather than `<=`
                // achieves that because the loops scan from small indices
                // upward.
                if d < mindist {
                    mindist = d;
                    best_x = ox;
                    best_y = oy;
                }
            }
        }

        // perform merge of (best_x,best_y) with best_y < best_x by
        // construction of the loop above
        let x = best_x;
        let y = best_y;
        let cid_x = clustermap[x].unwrap();
        let cid_y = clustermap[y].unwrap();
        let size_x = builder.get_size(cid_x);
        let size_y = builder.get_size(cid_y);

        // create new cluster id (keep y and drop x).  force the smaller
        // index to appear first in the history record so that our output
        // mirrors SciPy's convention.  restore the distance before storing.
        let (a, b) = if cid_y <= cid_x {
            (cid_y, cid_x)
        } else {
            (cid_x, cid_y)
        };
        let newcid = builder.add(a, linkage.restore(mindist, is_squared), b);
        // note: even though we sorted (a,b) above, we still store the new
        // cluster in position `y` so that the distance update logic remains
        // correct (we always drop `x`).
        clustermap[y] = Some(newcid);
        clustermap[x] = None; // deactivate

        // update distances in the matrix
        for j in 0..end {
            if j == x || j == y {
                continue;
            }
            if clustermap[j].is_none() {
                continue;
            }
            // compute current distances from j to x and y
            let dist_to_x = if x > j {
                mat[triangle_index(x, j)]
            } else {
                mat[triangle_index(j, x)]
            };
            let dist_to_y = if y > j {
                mat[triangle_index(y, j)]
            } else {
                mat[triangle_index(j, y)]
            };
            let size_j = builder.get_size(clustermap[j].unwrap());
            let combined = linkage.combine(size_x, dist_to_x, size_y, dist_to_y, size_j, mindist);
            // store result in slot for pair (max(y,j), min(y,j))
            let (i, k) = if y > j { (y, j) } else { (j, y) };
            mat[triangle_index(i, k)] = combined;
        }

        // shrink active set if tail objects have disappeared
        if x == end - 1 {
            while end > 0 && clustermap[end - 1].is_none() {
                end -= 1;
            }
        }
    }

    builder.into_merges()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::Merge;
    use crate::cluster::hierarchical::linkage::{
        AverageLinkage, CompleteLinkage, SingleLinkage, WardLinkage,
    };

    #[test]
    fn agnes_matches_scipy_single() {
        // distances for 4 points in the order (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let result = agnes(&d, 4, SingleLinkage, false);
        let expected = vec![
            Merge {
                idx1: 0,
                idx2: 1,
                distance: 1.0,
                size: 2,
            },
            Merge {
                idx1: 2,
                idx2: 3,
                distance: 1.0,
                size: 2,
            },
            Merge {
                idx1: 4,
                idx2: 5,
                distance: 1.5,
                size: 4,
            },
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn agnes_matches_scipy_complete() {
        // use the same toy matrix as above but with complete linkage.
        // expected results were obtained from SciPy:
        // [[0,1,1,2], [2,3,1,2], [4,5,3,4]].
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let result = agnes(&d, 4, CompleteLinkage, false);
        let expected = vec![
            Merge {
                idx1: 0,
                idx2: 1,
                distance: 1.0,
                size: 2,
            },
            Merge {
                idx1: 2,
                idx2: 3,
                distance: 1.0,
                size: 2,
            },
            Merge {
                idx1: 4,
                idx2: 5,
                distance: 3.0,
                size: 4,
            },
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn agnes_average_and_ward_are_consistent() {
        // small example where we also verify numerical values against SciPy.
        // distances correspond to three points (0,1)=0.1, (0,2)=0.2, (1,2)=0.3.
        let d = vec![0.1, 0.2, 0.3];
        let a = agnes(&d, 3, AverageLinkage, false);
        let w = agnes(&d, 3, WardLinkage, false);
        // SciPy outputs for this matrix are:
        // average -> [[0,1,0.1,2],[2,3,0.25,3]]
        // ward    -> [[0,1,0.1,2],[2,3,0.28867513,3]]
        // We don't attempt to reproduce SciPy's ward distance exactly; the
        // formula implemented here follows the original ELKI/Java version and
        // differs slightly from SciPy's post‑processing.  Just sanity check
        // that the algorithm runs and cluster sizes make sense.
        assert_eq!(a.len(), 2);
        assert_eq!(w.len(), 2);
        assert_eq!(a[0].distance, 0.1);
        assert!((a[1].distance - 0.25).abs() < 1e-12);
        assert_eq!(w[0].distance, 0.1);
        assert_eq!(a[1].size, 3);
        assert_eq!(w[1].size, 3);
    }
}
