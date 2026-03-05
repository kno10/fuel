use num_traits::Float;

use super::common::{
    Builder, MergeHistory, condensed_get, condensed_set, find_active, shrink_active_end,
};
use super::linkage::Linkage;

/// Perform hierarchical clustering using the NN-Chain algorithm.
///
/// Input and output conventions are the same as [`crate::cluster::hierarchical::agnes`].
/// The input matrix uses lower-triangular condensed indexing.
#[must_use]
pub fn nn_chain<F: Float, L: Linkage<F> + Copy>(
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
    let mut chain: Vec<usize> = Vec::with_capacity((n / 4).max(2));
    let mut merged = 0usize;

    while merged < n - 1 {
        let mut a;
        let mut b;

        if chain.len() < 2 {
            a = find_active(0, end, &clustermap).expect("at least one active cluster");
            b = find_active(a + 1, end, &clustermap).expect("at least two active clusters");
            chain.clear();
            chain.push(a);
        } else {
            a = chain[chain.len() - 2];
            b = chain[chain.len() - 1];
            if clustermap[a].is_none() {
                // Irreducible linkage inversions can invalidate a cached chain.
                chain.truncate(chain.len() - 2);
                continue;
            }
            chain.pop();
        }

        let mut min_dist = condensed_get(&mat, a, b);
        loop {
            let mut c = b;
            for i in 0..end {
                if i == a || i == b || clustermap[i].is_none() {
                    continue;
                }
                let d = condensed_get(&mat, a, i);
                if d < min_dist {
                    min_dist = d;
                    c = i;
                }
            }
            b = a;
            a = c;
            chain.push(a);
            if chain.len() >= 3 && a == chain[chain.len() - 3] {
                break;
            }
        }

        let (x, y) = if a > b { (a, b) } else { (b, a) };
        let cid_x = clustermap[x].expect("x must be active");
        let cid_y = clustermap[y].expect("y must be active");
        let size_x = builder.get_size(cid_x);
        let size_y = builder.get_size(cid_y);

        let (h1, h2) = if cid_y <= cid_x {
            (cid_y, cid_x)
        } else {
            (cid_x, cid_y)
        };
        let new_id = builder.add(h1, linkage.restore(min_dist, is_squared), h2);
        clustermap[y] = Some(new_id);
        clustermap[x] = None;

        for j in 0..end {
            if j == x || j == y || clustermap[j].is_none() {
                continue;
            }
            let d_xj = condensed_get(&mat, x, j);
            let d_yj = condensed_get(&mat, y, j);
            let size_j = builder.get_size(clustermap[j].expect("j must be active"));
            let d = linkage.combine(size_x, d_xj, size_y, d_yj, size_j, min_dist);
            condensed_set(&mut mat, y, j, d);
        }

        if x == end - 1 {
            shrink_active_end(&clustermap, &mut end);
        }

        if chain.len() >= 3 {
            chain.truncate(chain.len() - 3);
        } else {
            chain.clear();
        }
        chain.push(y);

        merged += 1;
    }

    builder.into_merges()
}

#[cfg(test)]
mod tests {
    use crate::cluster::hierarchical::agnes;
    use crate::cluster::hierarchical::linkage::{AverageLinkage, CompleteLinkage};
    use crate::cluster::hierarchical::regression_support::{
        DATASETS, cluster_and_cut, evaluate_clustering, load_dataset, optionally_report,
    };

    use super::nn_chain;

    #[test]
    fn nn_chain_matches_agnes_complete_on_unique_distances() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let a = agnes(&d, 5, CompleteLinkage, false);
        let b = nn_chain(&d, 5, CompleteLinkage, false);
        assert_eq!(a, b);
    }

    #[test]
    fn nn_chain_matches_agnes_average_on_unique_distances() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let a = agnes(&d, 5, AverageLinkage, false);
        let b = nn_chain(&d, 5, AverageLinkage, false);
        assert_eq!(a, b);
    }

    #[test]
    fn nn_chain_regression_on_sample_datasets() {
        for dataset in DATASETS.iter().filter(|d| d.name != "nested_clusters") {
            let (features, truth) = load_dataset(dataset.name);
            let labels = cluster_and_cut(nn_chain, &features, dataset.min_clusters, AverageLinkage);
            let (ari, nmi) = evaluate_clustering(&labels, &truth);
            optionally_report("NNChain", dataset.name, ari, nmi);
            assert!(
                ari >= dataset.min_ari,
                "{name} ARI too low after NN-chain: {ari:.3} < {min_ari:.3}",
                name = dataset.name,
                ari = ari,
                min_ari = dataset.min_ari
            );
            assert!(
                nmi >= dataset.min_nmi,
                "{name} NMI too low after NN-chain: {nmi:.3} < {min_nmi:.3}",
                name = dataset.name,
                nmi = nmi,
                min_nmi = dataset.min_nmi
            );
        }
    }
}
