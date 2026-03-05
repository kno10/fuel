use crate::cluster::hierarchical::regression_support::{
    DATASETS, cluster_and_cut, evaluate_clustering, load_dataset, optionally_report,
};
use crate::cluster::hierarchical::{AverageLinkage, WardLinkage, agnes};

#[test]
fn average_linkage_regression() {
    for dataset in DATASETS {
        let (features, truth) = load_dataset(dataset.name);
        let labels = match dataset.name {
            "nested_clusters" => {
                cluster_and_cut(agnes, &features, dataset.min_clusters, WardLinkage)
            }
            _ => cluster_and_cut(agnes, &features, dataset.min_clusters, AverageLinkage),
        };
        let (ari, nmi) = evaluate_clustering(&labels, &truth);
        optionally_report("AGNES-Average", dataset.name, ari, nmi);
        assert!(
            ari >= dataset.min_ari,
            "{name} ARI too low: {ari:.3} < {min_ari:.3}",
            name = dataset.name,
            ari = ari,
            min_ari = dataset.min_ari
        );
        assert!(
            nmi >= dataset.min_nmi,
            "{name} NMI too low: {nmi:.3} < {min_nmi:.3}",
            name = dataset.name,
            nmi = nmi,
            min_nmi = dataset.min_nmi
        );
    }
}

#[cfg(test)]
mod linkage_regression_tests {
    use crate::cluster::hierarchical::agnes;
    use crate::cluster::hierarchical::linkage::{
        CentroidLinkage, CompleteLinkage, FlexibleBetaLinkage, GroupAverageLinkage, MedianLinkage,
        MinimumVarianceLinkage, SingleLinkage, WardLinkage, WeightedAverageLinkage,
    };
    use crate::cluster::hierarchical::regression_support::run_linkage_regression;

    macro_rules! linkage_case {
        ($name:ident, $linkage:expr, $label:expr) => {
            linkage_case!($name, $linkage, $label, 0.9, 0.9);
        };
        ($name:ident, $linkage:expr, $label:expr, $ari_scale:expr, $nmi_scale:expr) => {
            #[test]
            fn $name() {
                run_linkage_regression(agnes, $linkage, $label, $ari_scale, $nmi_scale);
            }
        };
    }

    linkage_case!(
        complete_linkage_regression,
        CompleteLinkage,
        "CompleteLinkage"
    );
    linkage_case!(
        single_linkage_regression,
        SingleLinkage,
        "SingleLinkage",
        0.3,
        0.5
    );
    linkage_case!(
        centroid_linkage_regression,
        CentroidLinkage,
        "CentroidLinkage"
    );
    linkage_case!(
        group_average_linkage_regression,
        GroupAverageLinkage,
        "GroupAverageLinkage"
    );
    linkage_case!(
        minimum_variance_linkage_regression,
        MinimumVarianceLinkage,
        "MinimumVarianceLinkage"
    );
    linkage_case!(
        median_linkage_regression,
        MedianLinkage,
        "MedianLinkage",
        0.88,
        0.9
    );
    linkage_case!(
        weighted_average_linkage_regression,
        WeightedAverageLinkage,
        "WeightedAverageLinkage",
        0.98,
        0.98
    );
    linkage_case!(
        flexible_beta_linkage_regression,
        FlexibleBetaLinkage::new(-0.25),
        "FlexibleBetaLinkage"
    );
    linkage_case!(ward_linkage_regression, WardLinkage, "WardLinkage");
}

#[cfg(test)]
mod prototype_regression_tests {
    use crate::cluster::hierarchical::regression_support::run_prototype_regression;
    use crate::cluster::hierarchical::{
        HacamVariant, hacam, medoid_linkage, minimax, minimax_anderberg, minimax_nn_chain,
    };

    #[test]
    fn hacam_minimum_sum_regression() {
        run_prototype_regression(
            |data| hacam(data, HacamVariant::MinimumSum),
            "HACAM-MinimumSum",
            0.55,
            0.66,
        );
    }

    #[test]
    fn hacam_minimum_sum_increase_regression() {
        run_prototype_regression(
            |data| hacam(data, HacamVariant::MinimumSumIncrease),
            "HACAM-MinimumSumIncrease",
            0.92,
            0.95,
        );
    }

    #[test]
    fn medoid_linkage_regression() {
        run_prototype_regression(|data| medoid_linkage(data), "MedoidLinkage", 0.97, 0.95);
    }

    #[test]
    fn minimax_regression() {
        run_prototype_regression(|data| minimax(data), "MiniMax", 0.92, 0.95);
    }

    #[test]
    fn minimax_anderberg_regression() {
        run_prototype_regression(
            |data| minimax_anderberg(data),
            "MiniMax-Anderberg",
            0.92,
            0.95,
        );
    }

    #[test]
    fn minimax_nn_chain_regression() {
        run_prototype_regression(|data| minimax_nn_chain(data), "MiniMax-NNChain", 0.92, 0.95);
    }
}

#[cfg(test)]
mod extraction_regression_tests {
    use crate::NOISE;
    use crate::cluster::hierarchical::regression_support::optionally_report;
    use crate::cluster::hierarchical::regression_support::{
        DATASETS, evaluate_clustering_isize, labels_from_extracted_hierarchy_k,
        labels_from_extracted_hierarchy_roots, load_dataset,
    };
    use crate::cluster::hierarchical::{
        AverageLinkage, WardLinkage, agnes, extract_clusters_with_noise,
        extract_hdbscan_hierarchy_hdbscan, extract_simplified_hierarchy_hdbscan,
        hdbscan_linear_memory,
    };
    use crate::distance::EuclideanDistance;
    use crate::distance_matrix::lower_triangular_matrix;
    use crate::matrix_data_access::MatrixDataAccess;

    #[test]
    fn clusters_with_noise_quality_regression() {
        for dataset in DATASETS {
            let (features, truth) = load_dataset(dataset.name);
            let access = MatrixDataAccess::with_distance(&features, EuclideanDistance);
            let condensed = lower_triangular_matrix(&access);
            let history = match dataset.name {
                "nested_clusters" => agnes(&condensed, features.len(), WardLinkage, false),
                _ => agnes(&condensed, features.len(), AverageLinkage, false),
            };

            let labels = extract_clusters_with_noise(&history, dataset.min_clusters, 2);
            let (ari, nmi) = evaluate_clustering_isize(&labels, &truth, Some(NOISE));
            optionally_report("HDBSCAN-LinearMemory", dataset.name, ari, nmi);
            let min_ari = dataset.min_ari * 0.80;
            let min_nmi = dataset.min_nmi * 0.80;

            assert!(
                ari >= min_ari,
                "[ClustersWithNoise] {name} ARI too low: {ari:.3} < {min:.3}",
                name = dataset.name,
                ari = ari,
                min = min_ari
            );
            assert!(
                nmi >= min_nmi,
                "[ClustersWithNoise] {name} NMI too low: {nmi:.3} < {min:.3}",
                name = dataset.name,
                nmi = nmi,
                min = min_nmi
            );
        }
    }

    #[test]
    fn hdbscan_hierarchy_flat_quality_regression() {
        for dataset in DATASETS {
            let (features, truth) = load_dataset(dataset.name);
            let access = MatrixDataAccess::with_distance(&features, EuclideanDistance);
            let hierarchy = hdbscan_linear_memory(&access, 5);
            let extracted = extract_hdbscan_hierarchy_hdbscan(&hierarchy, 5, false);
            let labels = labels_from_extracted_hierarchy_roots(&extracted.hierarchy, truth.len());
            let (ari, nmi) = evaluate_clustering_isize(&labels, &truth, Some(NOISE));

            let min_ari = dataset.min_ari * 0.45;
            let min_nmi = dataset.min_nmi * 0.50;

            assert!(
                ari >= min_ari,
                "[HDBSCANHierarchyExtraction] {name} ARI too low: {ari:.3} < {min:.3}",
                name = dataset.name,
                ari = ari,
                min = min_ari
            );
            assert!(
                nmi >= min_nmi,
                "[HDBSCANHierarchyExtraction] {name} NMI too low: {nmi:.3} < {min:.3}",
                name = dataset.name,
                nmi = nmi,
                min = min_nmi
            );
        }
    }

    #[test]
    fn simplified_hierarchy_flat_quality_regression() {
        for dataset in DATASETS {
            let (features, truth) = load_dataset(dataset.name);
            let access = MatrixDataAccess::with_distance(&features, EuclideanDistance);
            let hierarchy = hdbscan_linear_memory(&access, 5);
            let extracted = extract_simplified_hierarchy_hdbscan(&hierarchy, 1);
            let labels =
                labels_from_extracted_hierarchy_k(&extracted, truth.len(), dataset.min_clusters);
            let (ari, nmi) = evaluate_clustering_isize(&labels, &truth, Some(NOISE));
            let (min_ari, min_nmi) = match dataset.name {
                "balanced_gaussians" => (0.002, 0.010),
                "mixed_density_ellipses" => (0.60, 0.70),
                "nested_clusters" => (0.95, 0.95),
                _ => (0.0, 0.0),
            };

            assert!(
                ari >= min_ari,
                "[SimplifiedHierarchyExtraction(flat)] {name} ARI too low: {ari:.3} < {min:.3}",
                name = dataset.name,
                ari = ari,
                min = min_ari
            );
            assert!(
                nmi >= min_nmi,
                "[SimplifiedHierarchyExtraction(flat)] {name} NMI too low: {nmi:.3} < {min:.3}",
                name = dataset.name,
                nmi = nmi,
                min = min_nmi
            );
        }
    }
}

#[cfg(test)]
mod indexed_search_hdbscan_regression_tests {
    use crate::NOISE;
    use crate::VPTree;
    use crate::api::DataAccess;
    use crate::cluster::hierarchical::regression_support::{
        DATASETS, evaluate_clustering_isize, labels_from_extracted_hierarchy_roots, load_dataset,
        optionally_report,
    };
    use crate::cluster::hierarchical::{
        HdbscanHierarchy, boruvka_searchers_hdbscan, extract_hdbscan_hierarchy_hdbscan,
        heap_of_searchers_hdbscan, restarting_search_hdbscan,
    };
    use crate::distance::EuclideanDistance;
    use crate::matrix_data_access::MatrixDataAccess;
    use rand::{SeedableRng, rngs::StdRng};

    const MIN_POINTS: usize = 5;
    const TREE_SAMPLE: usize = 16;

    fn run_indexed_hdbscan_regression(
        label: &str,
        algorithm: fn(
            &VPTree,
            &MatrixDataAccess<'_, Vec<f64>, EuclideanDistance>,
            usize,
        ) -> HdbscanHierarchy,
    ) {
        for dataset in DATASETS {
            let (features, truth) = load_dataset(dataset.name);
            let data = MatrixDataAccess::with_distance(&features, EuclideanDistance);
            let sample = data.size().min(TREE_SAMPLE).max(1);
            let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
            let tree = VPTree::new(&data, sample, &mut rng);

            let hierarchy = algorithm(&tree, &data, MIN_POINTS);
            let extracted = extract_hdbscan_hierarchy_hdbscan(&hierarchy, MIN_POINTS, false);
            let labels = labels_from_extracted_hierarchy_roots(&extracted.hierarchy, truth.len());
            let (ari, nmi) = evaluate_clustering_isize(&labels, &truth, Some(NOISE));

            let min_ari = dataset.min_ari * 0.45;
            let min_nmi = dataset.min_nmi * 0.50;
            optionally_report(label, dataset.name, ari, nmi);

            assert!(
                ari >= min_ari,
                "[{label}] {name} ARI too low: {ari:.3} < {min:.3}",
                label = label,
                name = dataset.name,
                ari = ari,
                min = min_ari
            );
            assert!(
                nmi >= min_nmi,
                "[{label}] {name} NMI too low: {nmi:.3} < {min:.3}",
                label = label,
                name = dataset.name,
                nmi = nmi,
                min = min_nmi
            );
        }
    }

    #[test]
    fn heap_of_searchers_hdbscan_regression() {
        run_indexed_hdbscan_regression("HeapOfSearchersHDBSCAN", heap_of_searchers_hdbscan);
    }

    #[test]
    fn boruvka_searchers_hdbscan_regression() {
        run_indexed_hdbscan_regression("BoruvkaSearchersHDBSCAN", boruvka_searchers_hdbscan);
    }

    #[test]
    fn restarting_search_hdbscan_regression() {
        run_indexed_hdbscan_regression("RestartingSearchHDBSCAN", restarting_search_hdbscan);
    }
}

#[cfg(test)]
mod cophenetic_hdbscan_regression_tests {
    use crate::api::DataAccess;
    use crate::cluster::hierarchical::regression_support::DATASETS;
    use crate::cluster::hierarchical::{
        HdbscanHierarchy, boruvka_searchers_hdbscan, hdbscan_linear_memory,
        heap_of_searchers_hdbscan, restarting_search_hdbscan,
    };
    use crate::evaluation::cluster::cophenetic::cophenetic_correlation;
    use crate::{EuclideanDistance, MatrixDataAccess, VPTree};
    use rand::{SeedableRng, rngs::StdRng};

    const MIN_POINTS: usize = 5;
    const TREE_SAMPLE: usize = 16;

    #[test]
    fn hdbscan_cophenetic_correlation() {
        println!("algorithm,dataset,cophenetic_corr");
        for dataset in DATASETS {
            let (features, _) =
                crate::cluster::hierarchical::regression_support::load_dataset(dataset.name);
            let data = MatrixDataAccess::with_distance(&features, EuclideanDistance);
            let sample = data.size().min(TREE_SAMPLE).max(1);
            let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
            let tree = VPTree::new(&data, sample, &mut rng);

            let baseline = hdbscan_linear_memory(&data, MIN_POINTS);
            let algorithms: &[(
                &str,
                fn(&VPTree, &MatrixDataAccess<'_, _, EuclideanDistance>, usize) -> HdbscanHierarchy,
            )] = &[
                ("heap", heap_of_searchers_hdbscan),
                ("boruvka", boruvka_searchers_hdbscan),
                ("restarting", restarting_search_hdbscan),
            ];

            for (label, algorithm) in algorithms {
                let candidate = algorithm(&tree, &data, MIN_POINTS);
                let corr =
                    cophenetic_correlation(&baseline.merges, &candidate.merges, features.len());
                println!(
                    "{label},{dataset},{corr:.9}",
                    label = label,
                    dataset = dataset.name
                );
                assert!(
                    corr > 0.99,
                    "[{label}] {dataset} cophenetic correlation too low: {corr:.6}",
                    label = label,
                    dataset = dataset.name,
                    corr = corr
                );
            }
        }
    }
}
