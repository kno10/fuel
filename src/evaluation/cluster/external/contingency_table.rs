use std::collections::HashMap;

use super::{
    BCubed, Entropy, MaximumMatchingAccuracy, PairCounting, PairSetsIndex, SetMatchingPurity,
};

/// Contingency table representation used by a variety of clustering
/// evaluation metrics.
#[derive(Clone, Debug)]
pub struct ClusterContingencyTable {
    pub break_noise_clusters: bool,
    pub self_pairing: bool,
    pub size1: usize,
    pub size2: usize,
    pub contingency: Vec<Vec<usize>>,
    pub noise1: Vec<bool>,
    pub noise2: Vec<bool>,
}

impl ClusterContingencyTable {
    pub fn from_labels(
        labels1: &[isize], labels2: &[isize], self_pairing: bool, break_noise_clusters: bool,
        noise_label1: Option<isize>, noise_label2: Option<isize>,
    ) -> Self {
        assert_eq!(labels1.len(), labels2.len(), "label vectors must have equal length");

        let n = labels1.len();
        let mut map1: HashMap<isize, usize> = HashMap::new();
        let mut map2: HashMap<isize, usize> = HashMap::new();
        let mut next1 = 0usize;
        let mut next2 = 0usize;

        for &l in labels1 {
            map1.entry(l).or_insert_with(|| {
                let cur = next1;
                next1 += 1;
                cur
            });
        }
        for &l in labels2 {
            map2.entry(l).or_insert_with(|| {
                let cur = next2;
                next2 += 1;
                cur
            });
        }

        let size1 = map1.len();
        let size2 = map2.len();
        let mut contingency = vec![vec![0usize; size2 + 2]; size1 + 2];
        let mut noise1 = vec![false; size1];
        let mut noise2 = vec![false; size2];

        if let Some(noise) = noise_label1 && let Some(idx) = map1.get(&noise).copied() {
            noise1[idx] = true;
        }
        if let Some(noise) = noise_label2 && let Some(idx) = map2.get(&noise).copied() {
            noise2[idx] = true;
        }

        for (&label, &i) in &map1 {
            if Some(label) == noise_label1 {
                noise1[i] = true;
            }
        }
        for (&label, &j) in &map2 {
            if Some(label) == noise_label2 {
                noise2[j] = true;
            }
        }

        for k in 0..n {
            let i = map1[&labels1[k]];
            let j = map2[&labels2[k]];
            contingency[i][j] += 1;
            contingency[i][size2] += 1;
            contingency[size1][j] += 1;
            contingency[size1][size2] += 1;
        }

        for i in 0..size1 {
            contingency[i][size2 + 1] = contingency[i][size2];
            contingency[size1][size2 + 1] += contingency[i][size2];
        }
        for j in 0..size2 {
            contingency[size1 + 1][j] = contingency[size1][j];
            contingency[size1 + 1][size2] += contingency[size1][j];
        }

        Self { break_noise_clusters, self_pairing, size1, size2, contingency, noise1, noise2 }
    }

    pub fn is_strict_partitioning(&self) -> bool {
        let expected = self.contingency[self.size1][self.size2];
        self.contingency[self.size1][self.size2 + 1] == expected
            && self.contingency[self.size1 + 1][self.size2] == expected
    }

    pub fn pair_counting(&self) -> PairCounting { PairCounting::new(self) }

    pub fn entropy(&self) -> Entropy { Entropy::new(self) }

    pub fn bcubed(&self) -> BCubed { BCubed::new(self) }

    pub fn set_matching_purity(&self) -> SetMatchingPurity { SetMatchingPurity::new(self) }

    pub fn maximum_matching_accuracy(&self) -> MaximumMatchingAccuracy {
        MaximumMatchingAccuracy::new(self)
    }

    pub fn pair_sets_index(&self) -> PairSetsIndex { PairSetsIndex::new(self) }
}
