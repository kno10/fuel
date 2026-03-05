use std::fs;
use std::path::PathBuf;

use crate::cluster::dbscan::NOISE;
use crate::cluster::hierarchical::common::{
    Merge, MergeHistory, PrototypeMerge, PrototypeMergeHistory,
};
use crate::cluster::hierarchical::extraction::{
    ExtractedHierarchy, cut_dendrogram_by_number_of_clusters,
};
use crate::cluster::hierarchical::linkage::Linkage;
use crate::distance::EuclideanDistance;
use crate::distance_matrix::lower_triangular_matrix;
use crate::evaluation::cluster::ClusterContingencyTable;
use crate::matrix_data_access::MatrixDataAccess;

#[derive(Clone, Copy, Debug)]
pub(crate) struct DatasetCase {
    pub name: &'static str,
    pub min_clusters: usize,
    pub min_ari: f64,
    pub min_nmi: f64,
}

pub(crate) const DATASETS: [DatasetCase; 3] = [
    DatasetCase {
        name: "balanced_gaussians",
        min_clusters: 5,
        min_ari: 0.88,
        min_nmi: 0.87,
    },
    DatasetCase {
        name: "mixed_density_ellipses",
        min_clusters: 3,
        min_ari: 0.65,
        min_nmi: 0.68,
    },
    DatasetCase {
        name: "nested_clusters",
        min_clusters: 3,
        min_ari: 0.19,
        min_nmi: 0.40,
    },
];

pub(crate) fn load_dataset(name: &str) -> (Vec<Vec<f64>>, Vec<isize>) {
    let path = data_path(name);
    let text = fs::read_to_string(&path).expect("failed to read dataset CSV");
    let mut lines = text.lines();
    let header = lines.next().expect("missing CSV header");
    let dims = header.split(',').count() - 1;

    let mut features = Vec::new();
    let mut labels = Vec::new();
    for line in lines {
        let parts: Vec<_> = line.split(',').collect();
        assert_eq!(parts.len(), dims + 1);
        let point = parts[..dims]
            .iter()
            .map(|v| v.parse::<f64>().expect("invalid feature value"))
            .collect();
        let label = parts[dims].parse().expect("invalid label");
        features.push(point);
        labels.push(label);
    }
    (features, labels)
}

pub(crate) fn evaluate_clustering(labels: &[usize], truth: &[isize]) -> (f64, f64) {
    let prediction: Vec<isize> = labels.iter().map(|&v| v as isize).collect();
    let table = ClusterContingencyTable::from_labels(truth, &prediction, false, false, None, None);
    let ari = table.pair_counting().adjusted_rand_index();
    let nmi = table.entropy().arithmetic_nmi();
    (ari, nmi)
}

pub(crate) fn evaluate_clustering_isize(
    labels: &[isize],
    truth: &[isize],
    noise_label: Option<isize>,
) -> (f64, f64) {
    let table =
        ClusterContingencyTable::from_labels(truth, labels, false, false, None, noise_label);
    let ari = table.pair_counting().adjusted_rand_index();
    let nmi = table.entropy().arithmetic_nmi();
    (ari, nmi)
}

fn collect_subtree_members(node: usize, extracted: &ExtractedHierarchy, out: &mut Vec<usize>) {
    out.extend(extracted.nodes[node].members.iter().copied());
    for &child in &extracted.nodes[node].children {
        collect_subtree_members(child, extracted, out);
    }
}

fn labels_from_frontier(
    extracted: &ExtractedHierarchy,
    frontier: &[usize],
    n: usize,
) -> Vec<isize> {
    let mut labels = vec![NOISE; n];

    for (cid, &node) in frontier.iter().enumerate() {
        let mut members = Vec::new();
        collect_subtree_members(node, extracted, &mut members);
        for point in members {
            if point < n && labels[point] == NOISE {
                labels[point] = cid as isize;
            }
        }
    }
    labels
}

pub(crate) fn labels_from_extracted_hierarchy_roots(
    extracted: &ExtractedHierarchy,
    n: usize,
) -> Vec<isize> {
    labels_from_frontier(extracted, &extracted.roots, n)
}

pub(crate) fn labels_from_extracted_hierarchy_k(
    extracted: &ExtractedHierarchy,
    n: usize,
    min_clusters: usize,
) -> Vec<isize> {
    assert!(min_clusters > 0, "min_clusters must be positive");
    if extracted.roots.is_empty() {
        return vec![NOISE; n];
    }

    let mut frontier = extracted.roots.clone();
    while frontier.len() < min_clusters {
        let mut best_pos = None;
        let mut best_dist = f64::NEG_INFINITY;
        for (i, &node) in frontier.iter().enumerate() {
            if extracted.nodes[node].children.is_empty() {
                continue;
            }
            let d = extracted.nodes[node].distance;
            if d > best_dist {
                best_dist = d;
                best_pos = Some(i);
            }
        }

        let Some(pos) = best_pos else {
            break;
        };
        let node = frontier.swap_remove(pos);
        frontier.extend(extracted.nodes[node].children.iter().copied());
    }

    labels_from_frontier(extracted, &frontier, n)
}

pub(crate) fn cluster_and_cut<L>(
    algorithm: fn(&[f64], usize, L, bool) -> MergeHistory<f64>,
    data: &[Vec<f64>],
    min_clusters: usize,
    linkage: L,
) -> Vec<usize>
where
    L: Linkage<f64> + Copy,
{
    let access = MatrixDataAccess::with_distance(data, EuclideanDistance);
    let condensed = lower_triangular_matrix(&access);
    let history = algorithm(&condensed, data.len(), linkage, false);
    cut_dendrogram_by_number_of_clusters(&history, min_clusters)
}

fn data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("hierarchical")
        .join(format!("{name}.csv"))
}

pub(crate) fn optionally_report(linkage: &str, dataset: &str, ari: f64, nmi: f64) {
    if std::env::var("HIERARCHY_REPORT").is_ok() {
        println!(
            "[{linkage}] {dataset}: ARI {ari:.4}, NMI {nmi:.4}",
            linkage = linkage,
            dataset = dataset,
            ari = ari,
            nmi = nmi
        );
    }
}

pub(crate) fn run_linkage_regression<L>(
    algorithm: fn(&[f64], usize, L, bool) -> MergeHistory<f64>,
    linkage: L,
    linkage_name: &str,
    min_ari_scale: f64,
    min_nmi_scale: f64,
) where
    L: Linkage<f64> + Copy,
{
    let report = std::env::var("HIERARCHY_REPORT").is_ok();
    let mut summary_ari = Vec::new();
    let mut summary_nmi = Vec::new();

    for dataset in DATASETS.iter().filter(|d| d.name != "nested_clusters") {
        let (features, truth) = load_dataset(dataset.name);
        let labels = cluster_and_cut(algorithm, &features, dataset.min_clusters, linkage);
        let (ari, nmi) = evaluate_clustering(&labels, &truth);
        let min_ari = dataset.min_ari * min_ari_scale;
        let min_nmi = dataset.min_nmi * min_nmi_scale;
        assert!(
            ari >= min_ari,
            "[{linkage_name}] {dataset} ARI too low: {ari:.3} < {min_ari:.3}",
            linkage_name = linkage_name,
            dataset = dataset.name,
            ari = ari,
            min_ari = min_ari
        );
        assert!(
            nmi >= min_nmi,
            "[{linkage_name}] {dataset} NMI too low: {nmi:.3} < {min_nmi:.3}",
            linkage_name = linkage_name,
            dataset = dataset.name,
            nmi = nmi,
            min_nmi = min_nmi
        );
        summary_ari.push((dataset.name, ari));
        summary_nmi.push((dataset.name, nmi));
        optionally_report(linkage_name, dataset.name, ari, nmi);
    }

    if report {
        let avg_ari: f64 =
            summary_ari.iter().map(|(_, v)| v).sum::<f64>() / summary_ari.len() as f64;
        let avg_nmi: f64 =
            summary_nmi.iter().map(|(_, v)| v).sum::<f64>() / summary_nmi.len() as f64;
        println!(
            "[{linkage_name}] mean ARI {avg_ari:.4}, mean NMI {avg_nmi:.4} over {} datasets",
            summary_ari.len(),
            linkage_name = linkage_name,
            avg_ari = avg_ari,
            avg_nmi = avg_nmi
        );
    }
}

fn prototype_history_to_merge_history(history: &[PrototypeMerge<f64>]) -> Vec<Merge<f64>> {
    history
        .iter()
        .map(|entry| Merge {
            idx1: entry.idx1,
            idx2: entry.idx2,
            distance: entry.distance,
            size: entry.size,
        })
        .collect()
}

fn cut_prototype_history_by_number_of_clusters(
    history: &[PrototypeMerge<f64>],
    min_clusters: usize,
) -> Vec<usize> {
    let merges = prototype_history_to_merge_history(history);
    cut_dendrogram_by_number_of_clusters(&merges, min_clusters)
}

pub(crate) fn run_prototype_regression<A>(
    algorithm: A,
    algorithm_name: &str,
    min_ari_scale: f64,
    min_nmi_scale: f64,
) where
    A: for<'a> Fn(&MatrixDataAccess<'a, Vec<f64>, EuclideanDistance>) -> PrototypeMergeHistory<f64>,
{
    let report = std::env::var("HIERARCHY_REPORT").is_ok();
    let mut summary_ari = Vec::new();
    let mut summary_nmi = Vec::new();

    for dataset in DATASETS.iter().filter(|d| d.name != "nested_clusters") {
        let (features, truth) = load_dataset(dataset.name);
        let access = MatrixDataAccess::with_distance(&features, EuclideanDistance);
        let history = algorithm(&access);
        let labels = cut_prototype_history_by_number_of_clusters(&history, dataset.min_clusters);
        let (ari, nmi) = evaluate_clustering(&labels, &truth);
        let min_ari = dataset.min_ari * min_ari_scale;
        let min_nmi = dataset.min_nmi * min_nmi_scale;
        assert!(
            ari >= min_ari,
            "[{algorithm_name}] {dataset} ARI too low: {ari:.3} < {min_ari:.3}",
            algorithm_name = algorithm_name,
            dataset = dataset.name,
            ari = ari,
            min_ari = min_ari
        );
        assert!(
            nmi >= min_nmi,
            "[{algorithm_name}] {dataset} NMI too low: {nmi:.3} < {min_nmi:.3}",
            algorithm_name = algorithm_name,
            dataset = dataset.name,
            nmi = nmi,
            min_nmi = min_nmi
        );
        summary_ari.push((dataset.name, ari));
        summary_nmi.push((dataset.name, nmi));
        optionally_report(algorithm_name, dataset.name, ari, nmi);
    }

    if report {
        let avg_ari: f64 =
            summary_ari.iter().map(|(_, v)| v).sum::<f64>() / summary_ari.len() as f64;
        let avg_nmi: f64 =
            summary_nmi.iter().map(|(_, v)| v).sum::<f64>() / summary_nmi.len() as f64;
        println!(
            "[{algorithm_name}] mean ARI {avg_ari:.4}, mean NMI {avg_nmi:.4} over {} datasets",
            summary_ari.len(),
            algorithm_name = algorithm_name,
            avg_ari = avg_ari,
            avg_nmi = avg_nmi
        );
    }
}
