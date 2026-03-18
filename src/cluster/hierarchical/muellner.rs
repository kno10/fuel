use std::cmp::Ordering;
use std::collections::BinaryHeap;

use num_traits::Float;

use super::common::{
    Builder,
    MergeHistory,
    // nn-cache helpers
    find_best,
    initialize_nn_cache,
    shrink_active_end,
    update_matrix_and_cache_with_hook,
};
use super::linkage::Linkage;

#[derive(Clone, Copy, Debug)]
struct HeapEntry<F: Float> {
    dist: F,
    x: usize,
    y: usize,
}

impl<F: Float> PartialEq for HeapEntry<F> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.x == other.x && self.y == other.y
    }
}

impl<F: Float> Eq for HeapEntry<F> {}

impl<F: Float> PartialOrd for HeapEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Float> Ord for HeapEntry<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self
            .dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.x.cmp(&other.x).then_with(|| self.y.cmp(&other.y)),
        }
    }
}

/// Perform hierarchical clustering using Müllner's generic-linkage approach
/// with an Anderberg nearest-neighbor cache and a heap for candidate retrieval.
#[must_use]
pub fn muellner<F: Float, L: Linkage<F> + Copy>(
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

    let mut bestd = vec![F::infinity(); n];
    let mut besti = vec![usize::MAX; n];
    initialize_nn_cache(&mat, &clustermap, &mut bestd, &mut besti);

    let mut heap = BinaryHeap::with_capacity(n);
    for x in 1..n {
        push_candidate(&mut heap, &bestd, &besti, x);
    }

    for _ in 1..n {
        let (mindist, x, y) = pop_valid_merge(&mut heap, &bestd, &besti, &clustermap);

        let cid_x = clustermap[x].expect("x must be active");
        let cid_y = clustermap[y].expect("y must be active");
        let size_x = builder.get_size(cid_x);
        let size_y = builder.get_size(cid_y);

        let (h1, h2) = if cid_y <= cid_x {
            (cid_y, cid_x)
        } else {
            (cid_x, cid_y)
        };
        let new_id = builder.add(h1, linkage.restore(mindist, is_squared), h2);
        clustermap[y] = Some(new_id);
        clustermap[x] = None;
        besti[x] = usize::MAX;
        bestd[x] = F::infinity();

        update_matrix_and_cache_with_hook(
            &mut mat,
            &clustermap,
            &mut bestd,
            &mut besti,
            &builder,
            linkage,
            mindist,
            x,
            y,
            size_x,
            size_y,
            end,
            |bestd_slice, besti_slice, j| push_candidate(&mut heap, bestd_slice, besti_slice, j),
        );

        if y > 0 {
            find_best(&mat, &clustermap, &mut bestd, &mut besti, y);
            push_candidate(&mut heap, &bestd, &besti, y);
        }

        if x == end - 1 {
            shrink_active_end(&clustermap, &mut end);
        }
    }

    builder.into_merges()
}

fn push_candidate<F: Float>(
    heap: &mut BinaryHeap<HeapEntry<F>>,
    bestd: &[F],
    besti: &[usize],
    x: usize,
) {
    let y = besti[x];
    if y != usize::MAX {
        heap.push(HeapEntry {
            dist: bestd[x],
            x,
            y,
        });
    }
}

fn pop_valid_merge<F: Float>(
    heap: &mut BinaryHeap<HeapEntry<F>>,
    bestd: &[F],
    besti: &[usize],
    clustermap: &[Option<usize>],
) -> (F, usize, usize) {
    while let Some(entry) = heap.pop() {
        if clustermap[entry.x].is_none() {
            continue;
        }
        if entry.y == usize::MAX || clustermap[entry.y].is_none() {
            continue;
        }
        if besti[entry.x] != entry.y {
            continue;
        }
        if bestd[entry.x] != entry.dist {
            continue;
        }
        let (x, y) = if entry.y < entry.x {
            (entry.x, entry.y)
        } else {
            (entry.y, entry.x)
        };
        return (entry.dist, x, y);
    }

    panic!("no merge candidate found");
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
mod tests {
    use crate::cluster::hierarchical::agnes;
    use crate::cluster::hierarchical::regression_support::{
        DATASETS, cluster_and_cut, evaluate_clustering, load_dataset, optionally_report,
    };
    use crate::cluster::hierarchical::{AverageLinkage, CompleteLinkage};

    use super::muellner;

    #[test]
    fn muellner_matches_agnes_complete_on_unique_distances() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let a = agnes(&d, 5, CompleteLinkage, false);
        let b = muellner(&d, 5, CompleteLinkage, false);
        assert_eq!(a, b);
    }

    #[test]
    fn muellner_matches_agnes_average_on_unique_distances() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let a = agnes(&d, 5, AverageLinkage, false);
        let b = muellner(&d, 5, AverageLinkage, false);
        assert_eq!(a, b);
    }

    #[test]
    fn muellner_regression_on_sample_datasets() {
        for dataset in DATASETS.iter().filter(|d| d.name != "nested_clusters") {
            let (features, truth) = load_dataset(dataset.name);
            let labels = cluster_and_cut(muellner, &features, dataset.min_clusters, AverageLinkage);
            let (ari, nmi) = evaluate_clustering(&labels, &truth);
            optionally_report("Muellner", dataset.name, ari, nmi);
            assert!(
                ari >= dataset.min_ari,
                "{name} ARI too low after Mullner: {ari:.3} < {min_ari:.3}",
                name = dataset.name,
                ari = ari,
                min_ari = dataset.min_ari
            );
            assert!(
                nmi >= dataset.min_nmi,
                "{name} NMI too low after Mullner: {nmi:.3} < {min_nmi:.3}",
                name = dataset.name,
                nmi = nmi,
                min_nmi = dataset.min_nmi
            );
        }
    }
}
