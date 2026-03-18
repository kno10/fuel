#![allow(unused)]
use std::collections::BTreeMap;

pub mod external;
pub mod internal;

use crate::evaluation::cluster::external::*;
use crate::evaluation::cluster::internal::*;

#[inline]
pub(crate) fn f1_measure(precision: f64, recall: f64) -> f64 {
    let denom = precision + recall;
    if denom == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / denom
    }
}

#[allow(dead_code)]
pub struct ClusteringEvaluation {
    pub pair_counting: PairCounting,
    pub entropy: Entropy,
    pub bcubed: BCubed,
    pub set_matching_purity: SetMatchingPurity,
    pub maximum_matching_accuracy: MaximumMatchingAccuracy,
    pub pair_sets_index: PairSetsIndex,
}

#[allow(dead_code)]
pub fn evaluate_clustering(
    labels1: &[isize],
    labels2: &[isize],
    self_pairing: bool,
    break_noise_clusters: bool,
    noise_label1: Option<isize>,
    noise_label2: Option<isize>,
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

#[allow(dead_code)]
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

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "{a} !~= {b}");
    }

    #[test]
    fn perfect_match_scores_are_perfect() {
        let l1 = [0, 0, 1, 1, 2, 2];
        let l2 = [2, 2, 1, 1, 0, 0];
        let e: ClusteringEvaluation = evaluate_clustering(&l1, &l2, false, false, None, None);

        approx(e.pair_counting.f1_measure(), 1.0, 1e-12);
        approx(e.pair_counting.rand_index(), 1.0, 1e-12);
        approx(e.pair_counting.adjusted_rand_index(), 1.0, 1e-12);
        approx(e.entropy.mutual_information, e.entropy.entropy_first, 1e-12);
        approx(e.entropy.variation_of_information, 0.0, 1e-12);
        approx(e.bcubed.f1_measure(), 1.0, 1e-12);
        approx(e.set_matching_purity.f1_measure(), 1.0, 1e-12);
        approx(e.maximum_matching_accuracy.accuracy, 1.0, 1e-12);
    }

    #[test]
    fn split_vs_single_cluster_is_imperfect() {
        let l1 = [0, 0, 1, 1];
        let l2 = [0, 0, 0, 0];
        let e = evaluate_clustering(&l1, &l2, false, false, None, None);

        assert!(e.pair_counting.f1_measure() < 1.0);
        assert!(e.entropy.variation_of_information > 0.0);
        assert!(e.maximum_matching_accuracy.accuracy < 1.0);
        assert!(e.set_matching_purity.purity <= 1.0);
        assert!(e.set_matching_purity.inverse_purity <= 1.0);
    }

    #[test]
    fn break_noise_clusters_changes_pair_counting() {
        let l1 = [0, 0, -1, -1, -1, 1, 1];
        let l2 = [0, 0, 2, 2, 2, 1, 1];

        let a = evaluate_clustering(&l1, &l2, false, false, Some(-1), Some(-1));
        let b = evaluate_clustering(&l1, &l2, false, true, Some(-1), Some(-1));

        assert!(b.pair_counting.in_both <= a.pair_counting.in_both);
        assert!(b.pair_counting.rand_index() <= 1.0);
    }
}
