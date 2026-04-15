use std::fs;
use std::path::PathBuf;

use crate::distance::DistanceFunction;
use crate::evaluation::cluster::external::ClusterContingencyTable;
use crate::{CondensedDistanceMatrix, TableWithDistance};

#[derive(Clone, Copy, Debug)]
pub struct DatasetCase {
    pub name: &'static str,
    pub clusters: usize,
}

pub const DATASETS: [DatasetCase; 3] = [
    DatasetCase { name: "balanced_gaussians", clusters: 5 },
    DatasetCase { name: "mixed_density_ellipses", clusters: 3 },
    DatasetCase { name: "nested_clusters", clusters: 3 },
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

pub type ClusterQuality = (f64, f64, f64);

pub fn test_clustering_condensed<A, D>(
    method_name: &str, linkage_key: &str, distance: D, algorithm: A,
) where
    A: Fn(&CondensedDistanceMatrix<f64>, usize) -> (Vec<usize>, f64),
    D: DistanceFunction<[f64], f64> + Sync + Copy,
{
    let tolerance = 1e-6;
    for dataset in DATASETS {
        let (features, truth) = load_dataset(dataset.name);
        let (labels, last_merge) = {
            let algorithm = &algorithm;
            let data: &[Vec<f64>] = &features;
            let min_clusters = dataset.clusters;
            let access = TableWithDistance::with_distance(data, distance);
            let condensed = CondensedDistanceMatrix::new_from_data(&access);
            algorithm(&condensed, min_clusters)
        };
        let last_merge =
            if distance.is_squared_distance() { last_merge.sqrt() } else { last_merge };
        let expected_quality = expected_quality(method_name, linkage_key, dataset.name);

        assert_clustering_quality(
            method_name,
            linkage_key,
            &dataset,
            labels,
            &truth,
            None,
            last_merge,
            expected_quality,
            tolerance,
        );
    }
}

pub fn test_clustering_table<A, D>(method_name: &str, linkage_key: &str, distance: D, algorithm: A)
where
    A: for<'a> Fn(&TableWithDistance<'a, f64, Vec<f64>, D, f64>, usize) -> (Vec<usize>, f64),
    D: DistanceFunction<[f64], f64> + Copy + Sync,
{
    let tolerance = 1e-6;
    for dataset in DATASETS {
        let (features, truth) = load_dataset(dataset.name);
        let access = TableWithDistance::with_distance(&features, distance);
        let (labels, last_merge) = algorithm(&access, dataset.clusters);
        let last_merge =
            if distance.is_squared_distance() { last_merge.sqrt() } else { last_merge };
        let expected_quality = expected_quality(method_name, linkage_key, dataset.name);

        assert_clustering_quality(
            method_name,
            linkage_key,
            &dataset,
            labels,
            &truth,
            None,
            last_merge,
            expected_quality,
            tolerance,
        );
    }
}

pub fn expected_quality(method: &str, linkage: &str, dataset: &str) -> ClusterQuality {
    match (method, linkage, dataset) {
        (_, "single", "balanced_gaussians") => {
            (0.2642693396251441, 0.5004205460950993, 1.5462335738972)
        }
        (_, "single", "mixed_density_ellipses") => {
            (0.6517766497461929, 0.7502937881452567, 3.0616196474803674)
        }
        (_, "single", "nested_clusters") => (1.0, 1.0, 3.446743063114394),

        (_, "complete", "balanced_gaussians") => {
            (0.8429912837725534, 0.8435180808760631, 11.477741481925124)
        }
        (_, "complete", "mixed_density_ellipses") => {
            (0.6852237829129537, 0.7028127099509311, 9.527291136286765)
        }
        (_, "complete", "nested_clusters") => {
            (0.1155513688983255, 0.34400888501379545, 12.844942028175522)
        }

        (_, "clink", "balanced_gaussians") => {
            (0.02248265566838501, 0.09479794676091588, 11.477741481925124)
        }
        (_, "clink", "mixed_density_ellipses") => {
            (0.3748330626240751, 0.3722092676844102, 9.527291136286765)
        }
        (_, "clink", "nested_clusters") => {
            (-0.007099266985542663, 0.015954599113258524, 12.844942028175522)
        }

        // These two optimize for squared euclidean, not euclidean as the others!
        ("GeometricNNChain" | "IncrementalNNChain", "average", "balanced_gaussians") => {
            (0.8974418173545886, 0.8782725401340807, 6.246983926053769)
        }
        ("GeometricNNChain" | "IncrementalNNChain", "average", "mixed_density_ellipses") => {
            (0.6661993133059516, 0.6920867067073396, 6.07485916980104)
        }
        ("GeometricNNChain" | "IncrementalNNChain", "average", "nested_clusters") => {
            (0.05542033946481422, 0.28138374507097075, 8.571772740923933)
        }
        (_, "average", "balanced_gaussians") => {
            (0.8974418173545886, 0.8782725401340807, 6.073161190678873)
        }
        (_, "average", "mixed_density_ellipses") => {
            (0.666199313306, 0.692086706707, 6.003013131008778)
        }
        (_, "average", "nested_clusters") => {
            (0.054466885229454656, 0.2909484733640857, 8.26425595916937)
        }

        (_, "weighted_average", "balanced_gaussians") => {
            (0.897497510884, 0.876193639191, 6.529066781308216)
        }
        (_, "weighted_average", "mixed_density_ellipses") => {
            (0.675594477116, 0.679962876062, 6.65683801410044)
        }
        (_, "weighted_average", "nested_clusters") => {
            (0.115551368898, 0.344008885014, 9.246720310791597)
        }

        (_, "ward", "balanced_gaussians") => (0.825101422425, 0.828068817147, 57.53945993063017),
        (_, "ward", "mixed_density_ellipses") => {
            (0.666199313306, 0.692086706707, 52.09125215802077)
        }
        (_, "ward", "nested_clusters") => (0.201692136753, 0.414839216923, 56.51236090145837),

        // Because of inversions, nnchain may differ
        ("NNChain" | "GeometricNNChain" | "IncrementalNNChain", "centroid", "nested_clusters") => {
            (0.05542033946481422, 0.28138374507097075, 6.412057929939861)
        }
        (_, "centroid", "balanced_gaussians") => (0.897321501719, 0.87940515035, 4.972868339252531),
        (_, "centroid", "mixed_density_ellipses") => {
            (0.666199313306, 0.692086706707, 5.683621572837187)
        }
        (_, "centroid", "nested_clusters") => (0.055420339465, 0.281383745071, 6.637839128788325),

        // Because of inversions, nnchain may differ
        ("NNChain" | "GeometricNNChain" | "IncrementalNNChain", "median", "balanced_gaussians") => {
            (0.8974975108841875, 0.8761936391912916, 6.156002350383626)
        }
        // INNC for centroid and median may differ even more
        ("IncrementalNNChain", "median", "mixed_density_ellipses") => {
            (0.6755944771158272, 0.6799628760623648, 6.163447771319738)
        }
        ("NNChain" | "GeometricNNChain" | "IncrementalNNChain", "median", "nested_clusters") => {
            (0.007424109373891063, 0.2422771322015568, 7.933565875096066)
        }
        (_, "median", "balanced_gaussians") => {
            (0.8974975108841875, 0.8761936391912916, 5.9698730136502842)
        }
        (_, "median", "mixed_density_ellipses") => {
            (0.6661993133059516, 0.6920867067073396, 6.0588692662669725)
        }
        (_, "median", "nested_clusters") => {
            (0.06142364950063201, 0.2994420626458651, 7.8125336308334452)
        }

        (_, "flexible", "balanced_gaussians") => (0.832120156519, 0.835841860529, 67.8224943001324),
        (_, "flexible", "mixed_density_ellipses") => {
            (0.678546384642, 0.699018053173, 67.56633266461665)
        }
        (_, "flexible", "nested_clusters") => (0.490122955485, 0.673607075192, 69.24369383665126),

        ("NNChain" | "SetNNChain", "mivar", "balanced_gaussians") => {
            (0.8231394536846123, 0.8252127518781317, 4.970652265907525)
        }
        ("NNChain" | "SetNNChain", "mivar", "mixed_density_ellipses") => {
            (0.7152706729485783, 0.7203662209989005, 5.568789097810354)
        }
        ("NNChain" | "SetNNChain", "mivar", "nested_clusters") => {
            (-0.0030578414334129144, 0.2261691486893834, 4.3822005750172)
        }
        ("GeometricNNChain" | "IncrementalNNChain", "mivar", "balanced_gaussians") => {
            (0.8231394536846123, 0.8252127518781317, 6.098108799533863)
        }
        // Verify why these two diverge more than usual here:
        ("GeometricNNChain", "mivar", "mixed_density_ellipses") => {
            (0.7152706729485783, 0.7203662209989005, 5.848911853649832)
        }
        ("IncrementalNNChain", "mivar", "mixed_density_ellipses") => {
            (0.7152706729485783, 0.7203662209989005, 5.844020122335656)
        }
        ("GeometricNNChain" | "IncrementalNNChain", "mivar", "nested_clusters") => {
            (-0.0030578414334129144, 0.2261691486893834, 4.763234631451172)
        }
        (_, "mivar", "balanced_gaussians") => {
            (0.8337491307944175, 0.8374384079747269, 4.970652265907525)
        }
        (_, "mivar", "mixed_density_ellipses") => {
            (0.6661993133059516, 0.6920867067073396, 5.568789097810354)
        }
        (_, "mivar", "nested_clusters") => {
            (-0.031523021317249225, 0.1701683459816219, 3.6378358157241957)
        }

        (_, "mnssq", "balanced_gaussians") => (0.831269158524, 0.830410591484, 84.47839073867756),
        (_, "mnssq", "mixed_density_ellipses") => {
            (0.654717913108, 0.660815345882, 60.46349505679712)
        }
        (_, "mnssq", "nested_clusters") => (0.216521006102, 0.340745765971, 109.31548524538178),

        (
            "NNChain" | "SetNNChain" | "GeometricNNChain" | "IncrementalNNChain",
            "mnvar",
            "balanced_gaussians",
        ) => (0.827731570835, 0.829083255609, 7.2978214403068336),
        (_, "mnvar", "balanced_gaussians") => (0.837752729547, 0.841364123066, 7.297821440306833),
        (_, "mnvar", "mixed_density_ellipses") => {
            (0.667098923409, 0.698602583236, 6.463819511698133)
        }
        (_, "mnvar", "nested_clusters") => (0.028812197877, 0.267752076472, 9.587602296890132),

        (_, "minimax", "balanced_gaussians") => (0.833154377259, 0.839423475743, 7.197231518802146),
        (_, "minimax", "mixed_density_ellipses") => {
            (0.666199313306, 0.692086706707, 5.660245703240438)
        }
        (_, "minimax", "nested_clusters") => (0.217626387283, 0.427220509987, 8.179028071258621),

        (_, "medoid", "balanced_gaussians") => (0.897045041373, 0.878126546181, 5.828871770212825),
        (_, "medoid", "mixed_density_ellipses") => {
            (0.732277402207, 0.730715498497, 5.439458677563012)
        }
        (_, "medoid", "nested_clusters") => (0.077322789236, 0.303540604612, 7.866097989992999),

        (_, "minimum_sum", "balanced_gaussians") => {
            (0.827731570835, 0.829083255609, 1003.183082343633)
        }
        (_, "minimum_sum", "mixed_density_ellipses") => {
            (0.678546384642, 0.699018053173, 546.8952999652028)
        }
        (_, "minimum_sum", "nested_clusters") => {
            (0.217209678126, 0.340888626523, 1180.099661187916)
        }

        // Supposedly due to inversions
        ("SetNNChain", "minimum_sum_increase", "mixed_density_ellipses") => {
            (0.7152706729485783, 0.7203662209989005, 317.9743714379554)
        }
        (_, "minimum_sum_increase", "balanced_gaussians") => {
            (0.831610348971, 0.839047621108, 309.13263153814)
        }
        (_, "minimum_sum_increase", "mixed_density_ellipses") => {
            (0.664479732553, 0.666292183739, 317.9743714379554)
        }
        (_, "minimum_sum_increase", "nested_clusters") => {
            (0.478001730032, 0.670016592455, 239.80362670346005)
        }

        // Supposedly due to inversions
        ("SetNNChain", "hausdorff", "balanced_gaussians") => {
            (0.9016256801765656, 0.8808077864392627, 8.943032149967975)
        }
        ("SetNNChain", "hausdorff", "mixed_density_ellipses") => {
            (0.741148916791, 0.732132957357, 6.905970528815897)
        }
        (_, "hausdorff", "balanced_gaussians") => {
            (0.9016256801765656, 0.8808077864392627, 5.808795233475465)
        }
        (_, "hausdorff", "mixed_density_ellipses") => {
            (0.6517766497461929, 0.7502937881452567, 8.91780927382765)
        }
        (_, "hausdorff", "nested_clusters") => {
            (0.054466885229454656, 0.2909484733640857, 8.047001604386724)
        }

        (_, "hdbscan", "balanced_gaussians") => (0.9016423642092285, 0.8811481710192761, f64::NAN),
        (_, "hdbscan", "mixed_density_ellipses") => {
            (0.6661993133059516, 0.6920867067073396, f64::NAN)
        }
        (_, "hdbscan", "nested_clusters") => (0.2016921367533269, 0.4148392169225281, f64::NAN),
        _ => (f64::NAN, f64::NAN, f64::NAN),
    }
}

pub fn assert_clustering_quality(
    algorithm_name: &str, linkage_key: &str, dataset: &DatasetCase, labels: Vec<usize>,
    truth: &[isize], noise_label: Option<isize>, actual_last_merge: f64,
    expected_quality: ClusterQuality, tolerance: f64,
) {
    let (ari, nmi) = if let Some(noise_label) = noise_label {
        let labels_isize: Vec<isize> = labels.into_iter().map(|v| v as isize).collect();
        evaluate_clustering_isize(&labels_isize, truth, Some(noise_label))
    } else {
        evaluate_clustering(&labels, truth)
    };

    let actual_quality = (ari, nmi, actual_last_merge);

    println!(
        "{} {} {}: quality={:?} (expect {:?})",
        algorithm_name, linkage_key, dataset.name, actual_quality, expected_quality,
    );

    let (ref_ari, ref_nmi, ref_last_merge) = expected_quality;
    if !ref_ari.is_nan() {
        assert!(
            (ari - ref_ari).abs() <= tolerance,
            "[{algorithm}] {dataset} ARI differs: {ari} vs {ref_ari} ({err:e})",
            algorithm = algorithm_name,
            dataset = dataset.name,
            ari = ari,
            ref_ari = ref_ari,
            err = (ari - ref_ari).abs(),
        );
    }
    if !ref_nmi.is_nan() {
        assert!(
            (nmi - ref_nmi).abs() <= tolerance,
            "[{algorithm}] {dataset} NMI differs: {nmi} vs {ref_nmi} ({err:e})",
            algorithm = algorithm_name,
            dataset = dataset.name,
            nmi = nmi,
            ref_nmi = ref_nmi,
            err = (nmi - ref_nmi).abs(),
        );
    }
    if !ref_last_merge.is_nan() {
        assert!(
            (actual_last_merge - ref_last_merge).abs() <= tolerance,
            "[{algorithm}] {dataset} last merge differs: {actual_last_merge} vs {ref_last_merge} ({err:e})",
            algorithm = algorithm_name,
            dataset = dataset.name,
            actual_last_merge = actual_last_merge,
            ref_last_merge = ref_last_merge,
            err = (actual_last_merge - ref_last_merge).abs(),
        );
    }
}
