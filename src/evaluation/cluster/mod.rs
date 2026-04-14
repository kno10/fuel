#![allow(unused)]
use std::collections::BTreeMap;

pub mod external;
pub mod internal;

use crate::evaluation::cluster::external::*;
use crate::evaluation::cluster::internal::*;

#[inline]
pub(crate) fn f1_measure(precision: f64, recall: f64) -> f64 {
    let denom = precision + recall;
    if denom == 0.0 { 0.0 } else { 2.0 * precision * recall / denom }
}

pub struct ClusteringEvaluation {
    pub pair_counting: PairCounting,
    pub entropy: Entropy,
    pub bcubed: BCubed,
    pub set_matching_purity: SetMatchingPurity,
    pub maximum_matching_accuracy: MaximumMatchingAccuracy,
    pub pair_sets_index: PairSetsIndex,
}

pub fn evaluate_clustering(
    labels1: &[isize], labels2: &[isize], self_pairing: bool, break_noise_clusters: bool,
    noise_label1: Option<isize>, noise_label2: Option<isize>,
) -> ClusteringEvaluation {
    let table = ClusterContingencyTable::from_labels(
        labels1,
        labels2,
        self_pairing,
        break_noise_clusters,
        noise_label1,
        noise_label2,
    );
    ClusteringEvaluation {
        pair_counting: table.pair_counting(),
        entropy: table.entropy(),
        bcubed: table.bcubed(),
        set_matching_purity: table.set_matching_purity(),
        maximum_matching_accuracy: table.maximum_matching_accuracy(),
        pair_sets_index: table.pair_sets_index(),
    }
}

pub fn cluster_sizes(labels: &[isize]) -> BTreeMap<isize, usize> {
    let mut sizes = BTreeMap::new();
    for &l in labels {
        *sizes.entry(l).or_insert(0) += 1;
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKLEARNA: [isize; 17] = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3];
    const SKLEARNB: [isize; 17] = [1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2];
    const SAMEA: [isize; 6] = [0, 0, 1, 1, 2, 2];
    const SAMEB: [isize; 6] = [2, 2, 1, 1, 0, 0];

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "{a} !~= {b}");
    }

    #[test]
    fn pair_counting_same_example() {
        let table = ClusterContingencyTable::from_labels(&SAMEA, &SAMEB, false, false, None, None);
        let pc = table.pair_counting();

        approx(pc.precision(), 1.0, 0.0);
        approx(pc.recall(), 1.0, 0.0);
        approx(pc.rand_index(), 1.0, 0.0);
        approx(pc.fowlkes_mallows(), 1.0, 0.0);
        approx(pc.jaccard(), 1.0, 0.0);
        approx(pc.f1_measure(), 1.0, 0.0);
        approx(pc.f_measure(5.0), 1.0, 0.0);
        approx(pc.adjusted_rand_index(), 1.0, 0.0);
        assert_eq!(pc.mirkin(), 0);
    }

    #[test]
    fn pair_counting_sklearn_example() {
        let table =
            ClusterContingencyTable::from_labels(&SKLEARNA, &SKLEARNB, false, false, None, None);
        let pc = table.pair_counting();

        approx(pc.precision(), 0.476190476190476, 1e-15);
        approx(pc.recall(), 0.5, 1e-15);
        approx(pc.rand_index(), 0.691176470588235, 1e-15);
        approx(pc.fowlkes_mallows(), 0.487950036474267, 1e-15);
        approx(pc.jaccard(), 0.32258064516129, 1e-15);
        approx(pc.f1_measure(), 0.487804878048781, 1e-15);
        approx(pc.f_measure(5.0), 0.499040307101727, 1e-15);
        approx(pc.adjusted_rand_index(), 0.26694045174538, 1e-15);
        assert_eq!(pc.mirkin(), 84);
    }

    #[test]
    fn entropy_same_example() {
        let table = ClusterContingencyTable::from_labels(&SAMEA, &SAMEB, false, false, None, None);
        let e = table.entropy();

        approx(e.mutual_information, e.upper_bound_mi(), 0.0);
        approx(e.joint_nmi(), 1.0, 0.0);
        approx(e.min_nmi(), 1.0, 0.0);
        approx(e.max_nmi(), 1.0, 0.0);
        approx(e.arithmetic_nmi(), 1.0, 0.0);
        approx(e.geometric_nmi(), 1.0, 0.0);
        approx(e.expected_mutual_information, 0.5441, 1e-5);
    }

    #[test]
    fn entropy_sklearn_example() {
        let table =
            ClusterContingencyTable::from_labels(&SKLEARNA, &SKLEARNB, false, false, None, None);
        let e = table.entropy();

        approx(e.mutual_information, 0.41022, 1e-5);
        approx(e.expected_mutual_information, 0.15042, 1e-5);
    }

    #[test]
    fn bcubed_sklearn_example() {
        let table =
            ClusterContingencyTable::from_labels(&SKLEARNA, &SKLEARNB, true, false, None, None);
        let bc = table.bcubed();

        approx(bc.precision, 0.57843137254902, 1e-15);
        approx(bc.recall, 0.584313725490196, 1e-15);
        approx(bc.f1_measure(), 0.5813576695433655, 1e-15);
    }

    #[test]
    fn maximum_matching_accuracy_sklearn_example() {
        let table =
            ClusterContingencyTable::from_labels(&SKLEARNA, &SKLEARNB, true, false, None, None);
        let mm = table.maximum_matching_accuracy();

        approx(mm.accuracy, 12.0 / 17.0, 1e-15);
    }

    #[test]
    fn evaluate_clustering_sklearn_example() {
        let e = evaluate_clustering(&SKLEARNA, &SKLEARNB, false, false, None, None);

        approx(e.pair_counting.adjusted_rand_index(), 0.26694045174538, 1e-15);
        approx(e.pair_counting.f1_measure(), 0.487804878048781, 1e-15);
        approx(e.entropy.mutual_information, 0.41022, 1e-5);
        approx(e.entropy.expected_mutual_information, 0.15042, 1e-5);
    }
}
