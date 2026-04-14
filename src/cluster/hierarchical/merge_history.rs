use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::cluster::hierarchical::idsize;
use crate::{DistPair, Float};

fn topo_distance_order<F: Float>(distances: &[F], children: &[(usize, usize)]) -> Vec<usize> {
    let m = distances.len();
    let mut remaining = vec![0usize; m];
    let mut parent = vec![usize::MAX; m];

    for (i, &(left, right)) in children.iter().enumerate() {
        if left != usize::MAX {
            remaining[i] += 1;
            parent[left] = i;
        }
        if right != usize::MAX {
            remaining[i] += 1;
            parent[right] = i;
        }
    }

    let mut heap: BinaryHeap<Reverse<DistPair<F>>> = BinaryHeap::with_capacity(m);
    for i in 0..m {
        if remaining[i] == 0 {
            heap.push(Reverse(DistPair::new(distances[i], i)));
        }
    }

    let mut order = Vec::with_capacity(m);
    while let Some(Reverse(entry)) = heap.pop() {
        let i = entry.index;
        order.push(i);
        let p = parent[i];
        if p != usize::MAX {
            remaining[p] -= 1;
            if remaining[p] == 0 {
                heap.push(Reverse(DistPair::new(distances[p], p)));
            }
        }
    }

    assert_eq!(order.len(), m, "merge history contains a cycle");
    order
}

/// A merge event in a hierarchical clustering history.
///
/// Each entry corresponds to the tuple `(i, j, d, s)` used by SciPy's
/// `linkage` output: `i` and `j` are cluster identifiers (initial points
/// have ids `0..n-1`, merged clusters are numbered `n..`). `d` is the merge
/// distance, `s` is the size of the new cluster, and `prototype` is the
/// representative point for prototype-based algorithms.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Merge<F: Float> {
    pub idx1: usize,
    pub idx2: usize,
    pub distance: F,
    pub size: usize,
    pub prototype: usize,
}

impl<F: Float> Merge<F> {
    #[inline]
    pub fn idx1(&self) -> usize { self.idx1 }

    #[inline]
    pub fn idx2(&self) -> usize { self.idx2 }

    #[inline]
    pub fn distance(&self) -> F { self.distance }

    #[inline]
    pub fn size(&self) -> usize { self.size }

    #[inline]
    pub fn prototype(&self) -> Option<usize> {
        if self.prototype != usize::MAX { Some(self.prototype) } else { None }
    }
}

/// Vertical storage for a complete hierarchical merge history.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MergeHistory<F: Float> {
    pub idx1: Vec<usize>,
    pub idx2: Vec<usize>,
    pub distance: Vec<F>,
    pub size: Vec<usize>,
    pub prototype: Option<Vec<usize>>,
}

impl<F: Float> MergeHistory<F> {
    #[must_use]
    pub fn new() -> Self { Self::with_capacity(0) }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            idx1: Vec::with_capacity(capacity),
            idx2: Vec::with_capacity(capacity),
            distance: Vec::with_capacity(capacity),
            size: Vec::with_capacity(capacity),
            prototype: None,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize { self.idx1.len() }

    #[must_use]
    pub fn is_empty(&self) -> bool { self.idx1.is_empty() }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<Merge<F>> {
        if index >= self.len() {
            return None;
        }

        Some(Merge {
            idx1: self.idx1[index],
            idx2: self.idx2[index],
            distance: self.distance[index],
            size: self.size[index],
            prototype: match &self.prototype {
                Some(proto) => proto[index],
                None => usize::MAX,
            },
        })
    }

    #[must_use]
    pub fn last(&self) -> Option<Merge<F>> {
        if self.is_empty() { None } else { self.get(self.len() - 1) }
    }

    pub fn push(&mut self, merge: Merge<F>) {
        let len = self.len();
        self.idx1.push(merge.idx1);
        self.idx2.push(merge.idx2);
        self.distance.push(merge.distance);
        self.size.push(merge.size);

        match (&mut self.prototype, merge.prototype) {
            (Some(prototype), value) => prototype.push(value),
            (None, value) if value != usize::MAX => {
                let mut prototype = Vec::with_capacity(self.idx1.capacity());
                prototype.resize(len, usize::MAX);
                prototype.push(value);
                self.prototype = Some(prototype);
            }
            _ => {}
        }
    }

    #[must_use]
    pub fn iter(&self) -> MergeHistoryIter<'_, F> { MergeHistoryIter { history: self, index: 0 } }
}

impl<F: Float> Default for MergeHistory<F> {
    fn default() -> Self { Self::new() }
}

impl<F: Float> From<Vec<Merge<F>>> for MergeHistory<F> {
    fn from(merges: Vec<Merge<F>>) -> Self { merges.into_iter().collect() }
}

impl<F: Float> FromIterator<Merge<F>> for MergeHistory<F> {
    fn from_iter<T: IntoIterator<Item = Merge<F>>>(iter: T) -> Self {
        let mut history = MergeHistory::new();
        history.extend(iter);
        history
    }
}

impl<F: Float> Extend<Merge<F>> for MergeHistory<F> {
    fn extend<T: IntoIterator<Item = Merge<F>>>(&mut self, iter: T) {
        for merge in iter {
            self.push(merge);
        }
    }
}

pub struct MergeHistoryIter<'a, F: Float> {
    history: &'a MergeHistory<F>,
    index: usize,
}

impl<'a, F: Float> Iterator for MergeHistoryIter<'a, F> {
    type Item = Merge<F>;

    fn next(&mut self) -> Option<Self::Item> {
        let merge = self.history.get(self.index);
        self.index += 1;
        merge
    }
}

impl<'a, F: Float> IntoIterator for &'a MergeHistory<F> {
    type Item = Merge<F>;
    type IntoIter = MergeHistoryIter<'a, F>;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

pub struct MergeHistoryIntoIter<F: Float> {
    idx1: std::vec::IntoIter<usize>,
    idx2: std::vec::IntoIter<usize>,
    distance: std::vec::IntoIter<F>,
    size: std::vec::IntoIter<usize>,
    prototype: Option<std::vec::IntoIter<usize>>,
}

impl<F: Float> Iterator for MergeHistoryIntoIter<F> {
    type Item = Merge<F>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx1 = self.idx1.next()?;
        let idx2 = self.idx2.next().expect("inconsistent merge history lengths");
        let distance = self.distance.next().expect("inconsistent merge history lengths");
        let size = self.size.next().expect("inconsistent merge history lengths");
        let prototype = match &mut self.prototype {
            Some(iter) => iter.next().expect("inconsistent merge history lengths"),
            None => usize::MAX,
        };
        Some(Merge { idx1, idx2, distance, size, prototype })
    }
}

impl<F: Float> IntoIterator for MergeHistory<F> {
    type Item = Merge<F>;
    type IntoIter = MergeHistoryIntoIter<F>;

    fn into_iter(self) -> Self::IntoIter {
        MergeHistoryIntoIter {
            idx1: self.idx1.into_iter(),
            idx2: self.idx2.into_iter(),
            distance: self.distance.into_iter(),
            size: self.size.into_iter(),
            prototype: self.prototype.map(|p| p.into_iter()),
        }
    }
}

/// Builder for merge history and cluster sizes.
pub(crate) struct Builder<F: Float> {
    n: usize,
    merges: MergeHistory<F>,
}

impl<F: Float> Builder<F> {
    pub(crate) fn new(n: usize) -> Self {
        Self { n, merges: MergeHistory::with_capacity(n.saturating_sub(1)) }
    }

    pub(crate) fn get_size(&self, cid: idsize) -> usize {
        if cid as usize >= self.n {
            self.merges.get(cid as usize - self.n).unwrap().size
        } else {
            1
        }
    }

    /// Record a new merge of clusters `a` and `b` at distance `dist`.
    ///
    /// Returns the identifier assigned to the newly-created cluster.
    pub(crate) fn add(&mut self, a: idsize, dist: F, b: idsize) -> idsize {
        self.add_with_prototype(a, dist, b, usize::MAX)
    }

    pub(crate) fn add_with_prototype(
        &mut self, a: idsize, dist: F, b: idsize, prototype: usize,
    ) -> idsize {
        let size = self.get_size(a) + self.get_size(b);
        let new_id = self.n + self.merges.len();
        self.merges.push(Merge {
            idx1: a as usize,
            idx2: b as usize,
            distance: dist,
            size,
            prototype,
        });
        new_id as idsize
    }

    /// Check if the merge distances are non-decreasing.
    pub(crate) fn check_monotone(&self) -> bool {
        if self.merges.is_empty() {
            return true;
        }

        let mut cur = self.merges.get(0).unwrap().distance;
        for merge in self.merges.iter().skip(1) {
            if merge.distance < cur {
                return false;
            }
            cur = merge.distance;
        }
        true
    }

    /// Reorder the merge history to make cluster distances monotone when possible.
    ///
    /// Returns a mapping from old merge index to new merge index if reordering
    /// was applied, or `None` if input was already monotone.
    pub(crate) fn optimize_order_in_place(&mut self) -> Option<Vec<usize>> {
        if self.check_monotone() {
            return None;
        }

        let (n, m) = (self.n, self.merges.len());
        let distances: Vec<F> = self.merges.iter().map(|merge| merge.distance).collect();
        let children: Vec<(usize, usize)> = self
            .merges
            .iter()
            .map(|merge| {
                (
                    if merge.idx1 >= n { merge.idx1 - n } else { usize::MAX },
                    if merge.idx2 >= n { merge.idx2 - n } else { usize::MAX },
                )
            })
            .collect();
        let order = topo_distance_order(&distances, &children);

        let old_merges = std::mem::take(&mut self.merges);

        let mut reverse = vec![usize::MAX; m];
        let mut new_merges = MergeHistory::with_capacity(m);
        for (new_index, &old_index) in order.iter().enumerate() {
            let old = old_merges.get(old_index).unwrap();
            let idx1 = if old.idx1 < n { old.idx1 } else { reverse[old.idx1 - n] + n };
            let idx2 = if old.idx2 < n { old.idx2 } else { reverse[old.idx2 - n] + n };
            new_merges.push(Merge {
                idx1,
                idx2,
                distance: old.distance,
                size: old.size,
                prototype: usize::MAX,
            });
            reverse[old_index] = new_index;
        }

        self.merges = new_merges;
        Some(reverse)
    }

    /// Consume the builder and return the collected merge history.
    pub(crate) fn into_merges(self) -> MergeHistory<F> { self.merges }
}

#[cfg(test)]
mod tests {
    use super::Builder;

    #[test]
    fn builder_optimize_order_reorders_nomonotone() {
        let mut builder = Builder::<f64>::new(4);
        builder.add(0, 2.0, 1);
        builder.add(2, 1.0, 3);

        assert!(!builder.check_monotone());
        let order = builder.optimize_order_in_place();
        assert!(order.is_some());

        let history = builder.into_merges();
        assert_eq!(history.get(0).unwrap().distance, 1.0);
        assert_eq!(history.get(1).unwrap().distance, 2.0);
    }

    #[test]
    fn builder_optimize_order_preserves_topology_on_inversion() {
        let mut builder = Builder::<f64>::new(6);
        let left = builder.add(0, 10.0, 1);
        let right = builder.add(2, 1.0, 3);
        builder.add(left, 0.5, right);

        assert!(!builder.check_monotone());
        let order = builder.optimize_order_in_place();
        assert!(order.is_some());

        let history = builder.into_merges();
        assert_eq!(history.get(0).unwrap().distance, 1.0);
        assert_eq!(history.get(1).unwrap().distance, 10.0);
        assert_eq!(history.get(2).unwrap().distance, 0.5);
    }
}
