use super::common::{
    Builder, MergeHistory, find_best, run_anderberg_nn_cache, update_matrix_and_cache_with_hook,
};
use super::linkage::Linkage;
use crate::Float;

/// Perform hierarchical clustering using Anderberg's NN-cache acceleration.
///
/// Input and output conventions are the same as [`crate::cluster::hierarchical::agnes`].
#[must_use]
pub fn anderberg<F: Float, L: Linkage<F> + Copy>(
    distances: &[F], n: usize, linkage: L, is_squared: bool,
) -> MergeHistory<F> {
    assert!(n > 0, "number of points must be positive");
    assert_eq!(distances.len(), n * (n - 1) / 2, "bad condensed matrix length");

    let mat: Vec<F> = distances.iter().map(|&d| linkage.initial(d, is_squared)).collect();
    let mut empty_prototypes = Vec::new();

    run_anderberg_nn_cache::<F, Builder<F>, _, _, _>(
        mat,
        n,
        move |mat,
              clustermap,
              builder,
              bestd,
              besti,
              x,
              y,
              mindist,
              end,
              size_x,
              size_y,
              _offset,
              _prototypes,
              _prototype| {
            update_matrix_and_cache_with_hook(
                mat,
                clustermap,
                bestd,
                besti,
                builder,
                linkage,
                mindist,
                x,
                y,
                size_x,
                size_y,
                end,
                |_, _, _| {},
            );
        },
        move |y, _x, clustermap, mat, bestd, besti| {
            if y > 0 {
                find_best(mat, clustermap, bestd, besti, y);
            }
        },
        move |mindist, _x, _y, _offset, _prototypes| (linkage.restore(mindist, is_squared), None),
        true,
        &mut empty_prototypes,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
mod tests {
    // imported via full path to avoid module/name conflict
    use super::anderberg;
    use crate::cluster::hierarchical::regression_support::{
        DATASETS, cluster_and_cut, evaluate_clustering, load_dataset, optionally_report,
    };
    use crate::cluster::hierarchical::{AverageLinkage, CompleteLinkage};

    #[test]
    fn anderberg_matches_agnes_complete_on_unique_distances() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let a = crate::cluster::hierarchical::agnes(&d, 5, CompleteLinkage, false);
        let b = anderberg(&d, 5, CompleteLinkage, false);
        assert_eq!(a, b);
    }

    #[test]
    fn anderberg_matches_agnes_average_on_unique_distances() {
        let d = vec![1.0, 8.0, 15.0, 22.0, 2.0, 9.0, 16.0, 3.0, 10.0, 4.0];
        let a = crate::cluster::hierarchical::agnes(&d, 5, AverageLinkage, false);
        let b = anderberg(&d, 5, AverageLinkage, false);
        assert_eq!(a, b);
    }

    #[test]
    fn anderberg_regression_on_sample_datasets() {
        for dataset in DATASETS.iter().filter(|d| d.name != "nested_clusters") {
            let (features, truth) = load_dataset(dataset.name);
            let labels =
                cluster_and_cut(anderberg, &features, dataset.min_clusters, AverageLinkage);
            let (ari, nmi) = evaluate_clustering(&labels, &truth);
            optionally_report("Anderberg", dataset.name, ari, nmi);
            assert!(
                ari >= dataset.min_ari,
                "{name} ARI too low after Anderberg: {ari:.3} < {min_ari:.3}",
                name = dataset.name,
                ari = ari,
                min_ari = dataset.min_ari
            );
            assert!(
                nmi >= dataset.min_nmi,
                "{name} NMI too low after Anderberg: {nmi:.3} < {min_nmi:.3}",
                name = dataset.name,
                nmi = nmi,
                min_nmi = dataset.min_nmi
            );
        }
    }
}
