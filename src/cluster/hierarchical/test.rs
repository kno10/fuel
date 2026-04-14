use std::fs;
use std::path::PathBuf;

use crate::distance::DistanceFunction;
use crate::evaluation::cluster::external::ClusterContingencyTable;
use crate::{CondensedDistanceMatrix, TableWithDistance};

#[derive(Clone, Copy, Debug)]
pub struct DatasetCase {
    pub name: &'static str,
    pub min_clusters: usize,
    pub min_ari: f64,
    pub min_nmi: f64,
}

pub const DATASETS: [DatasetCase; 3] = [
    DatasetCase { name: "balanced_gaussians", min_clusters: 5, min_ari: 0.88, min_nmi: 0.87 },
    DatasetCase { name: "mixed_density_ellipses", min_clusters: 3, min_ari: 0.65, min_nmi: 0.68 },
    DatasetCase { name: "nested_clusters", min_clusters: 3, min_ari: 0.19, min_nmi: 0.40 },
];

pub(crate) struct ScalarDistance;

impl crate::distance::DistanceFunction<[f64], f64> for ScalarDistance {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let ai = a.first().copied().unwrap_or(0.0);
        let bi = b.first().copied().unwrap_or(0.0);
        (ai - bi).abs()
    }
}

pub fn load_dataset(name: &str) -> (Vec<Vec<f64>>, Vec<isize>) {
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

pub fn evaluate_clustering(labels: &[usize], truth: &[isize]) -> (f64, f64) {
    debug_assert!(labels.len() <= 100000);
    let prediction: Vec<isize> = labels
        .iter()
        .map(|&v| {
            debug_assert!(v <= 100000);
            isize::try_from(v).unwrap()
        })
        .collect();
    let table = ClusterContingencyTable::from_labels(truth, &prediction, false, false, None, None);
    let ari = table.pair_counting().adjusted_rand_index();
    let nmi = table.entropy().arithmetic_nmi();
    (ari, nmi)
}

pub fn evaluate_clustering_isize(
    labels: &[isize], truth: &[isize], noise_label: Option<isize>,
) -> (f64, f64) {
    let table =
        ClusterContingencyTable::from_labels(truth, labels, false, false, None, noise_label);
    let ari = table.pair_counting().adjusted_rand_index();
    let nmi = table.entropy().arithmetic_nmi();
    (ari, nmi)
}

fn data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("hierarchical")
        .join(format!("{name}.csv"))
}

pub fn test_clustering_condensed<A, D>(
    method_name: &str, linkage_key: &str, distance_metric: D, algorithm: A,
) where
    A: Fn(&CondensedDistanceMatrix<f64>, usize) -> Vec<usize>,
    D: DistanceFunction<[f64], f64> + Sync + Copy,
{
    let tolerance = 1e-6;
    for dataset in DATASETS {
        let (features, truth) = load_dataset(dataset.name);
        let labels = {
            let algorithm = &algorithm;
            let data: &[Vec<f64>] = &features;
            let min_clusters = dataset.min_clusters;
            let access = TableWithDistance::with_distance(data, distance_metric);
            let condensed = CondensedDistanceMatrix::new_from_data(&access);
            algorithm(&condensed, min_clusters)
        };
        assert_clustering_quality(
            method_name,
            linkage_key,
            &dataset,
            labels,
            &truth,
            None,
            tolerance,
        );
    }
}

pub fn test_clustering_table<A, D>(method_name: &str, linkage_key: &str, distance: D, algorithm: A)
where
    A: for<'a> Fn(&TableWithDistance<'a, f64, Vec<f64>, D, f64>, usize) -> Vec<usize>,
    D: DistanceFunction<[f64], f64> + Copy,
{
    let tolerance = 1e-6;
    for dataset in DATASETS {
        let (features, truth) = load_dataset(dataset.name);
        let access = TableWithDistance::with_distance(&features, distance);

        let labels = algorithm(&access, dataset.min_clusters);
        assert_clustering_quality(
            method_name,
            linkage_key,
            &dataset,
            labels,
            &truth,
            None,
            tolerance,
        );
    }
}

pub fn baseline_ari_nmi(method: &str, linkage: &str, dataset: &str) -> Option<(f64, f64)> {
    match (method, linkage, dataset) {
        // from scipy, do not change
        (_, "single", "balanced_gaussians") => Some((0.26426933, 0.50042054)),
        (_, "single", "mixed_density_ellipses") => Some((0.65177664, 0.75029378)),
        (_, "single", "nested_clusters") => Some((1.0, 1.0)),

        // from scipy, do not change
        (_, "complete", "balanced_gaussians") => Some((0.84299128, 0.84351808)),
        (_, "complete", "mixed_density_ellipses") => Some((0.68522378, 0.70281270)),
        (_, "complete", "nested_clusters") => Some((0.11555136, 0.34400888)),

        // ELKI reference values
        (_, "clink", "balanced_gaussians") => Some((0.02248265, 0.09479794)),
        (_, "clink", "mixed_density_ellipses") => Some((0.37483306, 0.37220926)),
        (_, "clink", "nested_clusters") => Some((-0.00709926, 0.01595459)),

        // Geometric uses squared Euclidean, hence differs
        ("GeometricNNChain" | "IncrementalNNChain", "average", "balanced_gaussians") => {
            Some((0.89744181, 0.87827254))
        }
        ("GeometricNNChain" | "IncrementalNNChain", "average", "mixed_density_ellipses") => {
            Some((0.66619931, 0.69208670))
        }
        ("GeometricNNChain" | "IncrementalNNChain", "average", "nested_clusters") => {
            Some((0.05542034, 0.28138375))
        }
        // from scipy, do not change
        (_, "average", "balanced_gaussians") => Some((0.89744181, 0.87827254)),
        (_, "average", "mixed_density_ellipses") => Some((0.66619931, 0.69208670)),
        (_, "average", "nested_clusters") => Some((0.05446688, 0.29094847)),

        // from scipy, do not change
        (_, "weighted_average", "balanced_gaussians") => Some((0.89749751, 0.87619363)),
        (_, "weighted_average", "mixed_density_ellipses") => Some((0.67559447, 0.67996287)),
        (_, "weighted_average", "nested_clusters") => Some((0.11555136, 0.34400888)),

        // from scipy, do not change
        (_, "ward", "balanced_gaussians") => Some((0.82510142, 0.82806881)),
        (_, "ward", "mixed_density_ellipses") => Some((0.66619931, 0.69208670)),
        (_, "ward", "nested_clusters") => Some((0.20169213, 0.41483921)),

        // from scipy, do not change
        (_, "centroid", "balanced_gaussians") => Some((0.89732150, 0.87940515)),
        (_, "centroid", "mixed_density_ellipses") => Some((0.66619931, 0.69208670)),
        // NOT from scipy, because it yields 2 clusters due to a bug.
        (_, "centroid", "nested_clusters") => Some((0.05542033, 0.28138374)),

        // NN-chain differs due to inversions
        ("NNChain" | "GeometricNNChain" | "IncrementalNNChain", "median", "nested_clusters") => {
            Some((0.00742410, 0.24227713))
        }
        ("IncrementalNNChain", "median", "mixed_density_ellipses") => {
            Some((0.67559447, 0.67996287))
        }
        // from scipy, do not change
        (_, "median", "balanced_gaussians") => Some((0.89749751, 0.87619363)),
        (_, "median", "mixed_density_ellipses") => Some((0.66619931, 0.69208670)),
        (_, "median", "nested_clusters") => Some((0.06142364, 0.29944206)),

        (_, "flexible", "balanced_gaussians") => Some((0.83212015, 0.83584186)),
        (_, "flexible", "mixed_density_ellipses") => Some((0.67854638, 0.69901805)),
        (_, "flexible", "nested_clusters") => Some((0.49012295, 0.67360707)),

        // Podani minimum variance increase (similar, but not the same as Ward)
        // NN-chain differs due to inversions
        ("NNChain" | "SetNNChain", "mivar", "balanced_gaussians") => Some((0.82313945, 0.82521275)),
        ("NNChain" | "SetNNChain", "mivar", "mixed_density_ellipses") => {
            Some((0.71527067, 0.72036622))
        }
        ("NNChain" | "SetNNChain", "mivar", "nested_clusters") => Some((-0.00305784, 0.22616914)),

        ("GeometricNNChain" | "IncrementalNNChain", "mivar", "balanced_gaussians") => {
            Some((0.8231394536846123, 0.8252127518781317))
        }
        ("GeometricNNChain" | "IncrementalNNChain", "mivar", "mixed_density_ellipses") => {
            Some((0.7152706729485783, 0.7203662209989005))
        }
        ("GeometricNNChain" | "IncrementalNNChain", "mivar", "nested_clusters") => {
            Some((-0.0030578414334129144, 0.2261691486893834))
        }

        // ELKI reference values
        (_, "mivar", "balanced_gaussians") => Some((0.83374913, 0.83743840)),
        (_, "mivar", "mixed_density_ellipses") => Some((0.66619931, 0.69208670)),
        (_, "mivar", "nested_clusters") => Some((-0.03152302, 0.17016834)),

        //        ("GeometricNNChain" | "IncrementalNNChain", "mnssq", "balanced_gaussians") => {
        //            Some((0.8251014224250723, 0.8280688171474802))
        //        }
        // Podani-style least sum of squares.
        (_, "mnssq", "balanced_gaussians") => Some((0.8312691585239753, 0.8304105914840773)),
        (_, "mnssq", "mixed_density_ellipses") => Some((0.6547179131082174, 0.6608153458816465)),
        (_, "mnssq", "nested_clusters") => Some((0.21652100610164282, 0.3407457659712788)),

        ("NNChain" | "SetNNChain", "mnvar", "balanced_gaussians") => {
            Some((0.827731570835474, 0.8290832556093919))
        }
        ("GeometricNNChain" | "IncrementalNNChain", "mnvar", "balanced_gaussians") => {
            Some((0.827731570835474, 0.8290832556093919))
        }
        ("GeometricNNChain" | "IncrementalNNChain", "mnvar", "mixed_density_ellipses") => {
            Some((0.6670989234090616, 0.6986025832358843))
        }
        ("GeometricNNChain" | "IncrementalNNChain", "mnvar", "nested_clusters") => {
            Some((0.028812197877294544, 0.2677520764721653))
        }
        // Podani-style least variance.
        (_, "mnvar", "balanced_gaussians") => Some((0.8377527295473177, 0.8413641230660861)),
        (_, "mnvar", "mixed_density_ellipses") => Some((0.6670989234090616, 0.6986025832358843)),
        (_, "mnvar", "nested_clusters") => Some((0.028812197877294544, 0.2677520764721653)),

        (_, "minimax", "balanced_gaussians") => Some((0.83315437, 0.83942347)),
        (_, "minimax", "mixed_density_ellipses") => Some((0.66619931, 0.69208670)),
        (_, "minimax", "nested_clusters") => Some((0.21762638, 0.42722050)),

        (_, "medoid", "balanced_gaussians") => Some((0.89704504, 0.87812654)),
        (_, "medoid", "mixed_density_ellipses") => Some((0.73227740, 0.73071549)),
        (_, "medoid", "nested_clusters") => Some((0.07732278, 0.30354060)),

        // HACAM minimum sum
        ("SetNNChain", "minimum_sum", "balanced_gaussians") => Some((0.82773157, 0.82908325)),
        ("SetNNChain", "minimum_sum", "mixed_density_ellipses") => Some((0.67854638, 0.69901805)),
        ("SetNNChain", "minimum_sum", "nested_clusters") => Some((0.21720967, 0.34088862)),
        (_, "minimum_sum", "balanced_gaussians") => Some((0.82773157, 0.82908325)),
        (_, "minimum_sum", "mixed_density_ellipses") => Some((0.67854638, 0.69901805)),
        (_, "minimum_sum", "nested_clusters") => Some((0.21720967, 0.34088862)),

        // HACAM minimum sum increase
        ("SetNNChain", "minimum_sum_increase", "balanced_gaussians") => {
            Some((0.83161034, 0.83904762))
        }
        ("SetNNChain", "minimum_sum_increase", "mixed_density_ellipses") => {
            Some((0.71527067, 0.72036622))
        }
        ("SetNNChain", "minimum_sum_increase", "nested_clusters") => Some((0.47800173, 0.67001659)),
        // ELKI reference values
        (_, "minimum_sum_increase", "balanced_gaussians") => Some((0.83161034, 0.83904762)),
        (_, "minimum_sum_increase", "mixed_density_ellipses") => Some((0.66447973, 0.66629218)),
        (_, "minimum_sum_increase", "nested_clusters") => Some((0.47800173, 0.67001659)),

        // NN-chain differs due to inversions
        ("SetNNChain", "hausdorff", "mixed_density_ellipses") => Some((0.74114891, 0.73213295)),
        (_, "hausdorff", "balanced_gaussians") => Some((0.90162568, 0.88080778)),
        (_, "hausdorff", "mixed_density_ellipses") => Some((0.65177664, 0.75029378)),
        (_, "hausdorff", "nested_clusters") => Some((0.05446688, 0.29094847)),

        _ => None,
    }
}

pub fn assert_clustering_quality(
    algorithm_name: &str, linkage_key: &str, dataset: &DatasetCase, labels: Vec<usize>,
    truth: &[isize], noise_label: Option<isize>, tolerance: f64,
) {
    let (ari, nmi) = if let Some(noise_label) = noise_label {
        let labels_isize: Vec<isize> = labels.into_iter().map(|v| v as isize).collect();
        evaluate_clustering_isize(&labels_isize, truth, Some(noise_label))
    } else {
        evaluate_clustering(&labels, truth)
    };

    let baseline = baseline_ari_nmi(algorithm_name, linkage_key, dataset.name);

    println!(
        "{} {} {}: ARI={} NMI={} (baseline={:?})",
        algorithm_name, linkage_key, dataset.name, ari, nmi, baseline
    );

    if let Some((ref_ari, ref_nmi)) = baseline {
        assert!(
            (ari - ref_ari).abs() <= tolerance,
            "[{algorithm}] {dataset} ARI differs: {ari} vs {ref_ari} ({err:e})",
            algorithm = algorithm_name,
            dataset = dataset.name,
            ari = ari,
            ref_ari = ref_ari,
            err = (ari - ref_ari).abs(),
        );
        assert!(
            (nmi - ref_nmi).abs() <= tolerance,
            "[{algorithm}] {dataset} NMI differs: {nmi} vs {ref_nmi} ({err:e})",
            algorithm = algorithm_name,
            dataset = dataset.name,
            nmi = nmi,
            ref_nmi = ref_nmi,
            err = (nmi - ref_nmi).abs(),
        );
    } else {
        println!(
            "Skipping baseline validation for {}/{} because no reference values are available.",
            linkage_key, dataset.name
        );
    }
}
