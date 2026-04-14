use std::cmp::Ordering;

use crate::Float;
use crate::cluster::hdbscan::Merge;
use crate::cluster::hierarchical::MergeHistory;

/// Native pointer representation used by SLINK/CLINK.
///
/// For each object `i`, `pi[i]` is the parent object index and `lambda[i]` is
/// the linkage distance to that parent. Root objects satisfy `pi[i] == i` and
/// typically have `lambda[i] = +inf`.
#[derive(Debug, Clone, PartialEq)]
pub struct PointerRepresentation<F: Float> {
    pub pi: Vec<usize>,
    pub lambda: Vec<F>,
}

impl<F: Float> PointerRepresentation<F> {
    #[must_use]
    pub fn new(pi: Vec<usize>, lambda: Vec<F>) -> Self {
        assert_eq!(pi.len(), lambda.len(), "pi/lambda length mismatch");
        Self { pi, lambda }
    }

    /// Convert pointer representation to SciPy-style merge history.
    #[must_use]
    pub fn to_merge_history(&self) -> MergeHistory<F> { pointer_to_merge_history(self) }
}

/// Port of ELKI's builder-based conversion from pointer representation to
/// merge history (process nodes sorted by `lambda`, then `id`).
#[must_use]
pub fn pointer_to_merge_history<F: Float>(pointer: &PointerRepresentation<F>) -> MergeHistory<F> {
    let n = pointer.pi.len();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        pointer.lambda[a]
            .partial_cmp(&pointer.lambda[b])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut parent: Vec<usize> = (0..(2 * n - 1)).collect();
    let mut size = vec![1usize; 2 * n - 1];
    let mut merges = MergeHistory::with_capacity(n.saturating_sub(1));

    for &source in &order {
        let target = pointer.pi[source];
        if source == target {
            continue;
        }

        let s = uf_find(&mut parent, source);
        let t = uf_find(&mut parent, target);
        if s == t {
            continue;
        }

        let ss = size[s];
        let st = size[t];
        let (idx1, idx2) = if s <= t { (s, t) } else { (t, s) };
        merges.push(Merge {
            idx1,
            idx2,
            distance: pointer.lambda[source],
            size: ss + st,
            prototype: usize::MAX,
        });

        let new_id = n + merges.len() - 1;
        parent[s] = new_id;
        parent[t] = new_id;
        parent[new_id] = new_id;
        size[new_id] = ss + st;
    }

    assert_eq!(merges.len(), n - 1, "invalid pointer representation");
    merges
}

#[inline]
fn uf_find(parent: &mut [usize], x: usize) -> usize {
    let mut p = x;
    while parent[p] != p {
        p = parent[p];
    }
    let mut i = x;
    while parent[i] != p {
        let next = parent[i];
        parent[i] = p;
        i = next;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::pointer_to_merge_history;
    use crate::CondensedDistanceMatrix;
    use crate::cluster::hierarchical::slink::slink_pointer;

    #[test]
    fn pointer_conversion_builds_full_history() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let d_clone = d.clone();
        let cm = CondensedDistanceMatrix::new_from_condensed(d_clone, 5, false);
        let p = slink_pointer(&cm);
        let h = pointer_to_merge_history(&p);
        assert_eq!(h.len(), 4);
        assert_eq!(h.last().expect("non-empty history").size, 5);
    }
}
