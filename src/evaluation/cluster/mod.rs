use std::collections::{BTreeMap, HashMap};

pub mod cophenetic;
pub mod extractor;
pub mod internal;
pub mod pairsegments;

#[inline]
fn f1_measure(precision: f64, recall: f64) -> f64 {
    let denom = precision + recall;
    if denom == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / denom
    }
}

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
        labels1: &[isize],
        labels2: &[isize],
        self_pairing: bool,
        break_noise_clusters: bool,
        noise_label1: Option<isize>,
        noise_label2: Option<isize>,
    ) -> Self {
        assert_eq!(
            labels1.len(),
            labels2.len(),
            "label vectors must have equal length"
        );

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

        if let Some(noise) = noise_label1 {
            if let Some(&idx) = map1.get(&noise) {
                noise1[idx] = true;
            }
        }
        if let Some(noise) = noise_label2 {
            if let Some(&idx) = map2.get(&noise) {
                noise2[idx] = true;
            }
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

        Self {
            break_noise_clusters,
            self_pairing,
            size1,
            size2,
            contingency,
            noise1,
            noise2,
        }
    }

    pub fn is_strict_partitioning(&self) -> bool {
        let expected = self.contingency[self.size1][self.size2];
        self.contingency[self.size1][self.size2 + 1] == expected
            && self.contingency[self.size1 + 1][self.size2] == expected
    }

    pub fn pair_counting(&self) -> PairCounting {
        PairCounting::new(self)
    }

    pub fn entropy(&self) -> Entropy {
        Entropy::new(self)
    }

    pub fn edit_distance(&self) -> EditDistance {
        EditDistance::new(self)
    }

    pub fn bcubed(&self) -> BCubed {
        BCubed::new(self)
    }

    pub fn set_matching_purity(&self) -> SetMatchingPurity {
        SetMatchingPurity::new(self)
    }

    pub fn maximum_matching_accuracy(&self) -> MaximumMatchingAccuracy {
        MaximumMatchingAccuracy::new(self)
    }

    pub fn pair_sets_index(&self) -> PairSetsIndex {
        PairSetsIndex::new(self)
    }

    pub fn average_symmetric_gini(&self) -> WeightedMoments {
        let mut mv = WeightedMoments::default();
        for i1 in 0..self.size1 {
            let row_sum = self.contingency[i1][self.size2] as f64;
            if row_sum > 0.0 {
                let mut purity = 0.0;
                for i2 in 0..self.size2 {
                    let rel = self.contingency[i1][i2] as f64 / row_sum;
                    purity += rel * rel;
                }
                mv.put(purity, row_sum);
            }
        }
        for i2 in 0..self.size2 {
            let col_sum = self.contingency[self.size1][i2] as f64;
            if col_sum > 0.0 {
                let mut purity = 0.0;
                for i1 in 0..self.size1 {
                    let rel = self.contingency[i1][i2] as f64 / col_sum;
                    purity += rel * rel;
                }
                mv.put(purity, col_sum);
            }
        }
        mv
    }

    pub fn adjusted_symmetric_gini(&self) -> WeightedMoments {
        let mut mv = WeightedMoments::default();
        let total = self.contingency[self.size1][self.size2] as f64;
        if total <= 0.0 {
            return mv;
        }

        for i1 in 0..self.size1 {
            let row_sum = self.contingency[i1][self.size2] as f64;
            if row_sum > 0.0 {
                let mut purity = 0.0;
                let mut exp = 0.0;
                for i2 in 0..self.size2 {
                    let rel = self.contingency[i1][i2] as f64 / row_sum;
                    purity += rel * rel;
                    let e = self.contingency[self.size1][i2] as f64 / total;
                    exp += e * e;
                }
                let denom = 1.0 - exp;
                if denom != 0.0 {
                    mv.put((purity - exp) / denom, row_sum);
                }
            }
        }

        for i2 in 0..self.size2 {
            let col_sum = self.contingency[self.size1][i2] as f64;
            if col_sum > 0.0 {
                let mut purity = 0.0;
                let mut exp = 0.0;
                for i1 in 0..self.size1 {
                    let rel = self.contingency[i1][i2] as f64 / col_sum;
                    purity += rel * rel;
                    let e = self.contingency[i1][self.size2] as f64 / total;
                    exp += e * e;
                }
                let denom = 1.0 - exp;
                if denom != 0.0 {
                    mv.put((purity - exp) / denom, col_sum);
                }
            }
        }
        mv
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WeightedMoments {
    sum_w: f64,
    sum_wx: f64,
    sum_wx2: f64,
}

impl WeightedMoments {
    pub fn put(&mut self, x: f64, w: f64) {
        self.sum_w += w;
        self.sum_wx += w * x;
        self.sum_wx2 += w * x * x;
    }

    pub fn weight(&self) -> f64 {
        self.sum_w
    }

    pub fn mean(&self) -> f64 {
        if self.sum_w == 0.0 {
            0.0
        } else {
            self.sum_wx / self.sum_w
        }
    }

    pub fn variance(&self) -> f64 {
        if self.sum_w == 0.0 {
            0.0
        } else {
            let mean = self.mean();
            (self.sum_wx2 / self.sum_w) - mean * mean
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PairCounting {
    pub in_both: u64,
    pub in_first: u64,
    pub in_second: u64,
    pub in_none: u64,
}

impl PairCounting {
    pub fn new(table: &ClusterContingencyTable) -> Self {
        let mut in_b = 0u64;
        let mut in1 = 0u64;
        let mut in2 = 0u64;

        for i in 0..table.size1 {
            let size = table.contingency[i][table.size2 + 1] as u64;
            if table.break_noise_clusters && table.noise1[i] {
                if table.self_pairing {
                    in1 += size;
                }
            } else {
                in1 += size
                    * if table.self_pairing {
                        size
                    } else {
                        size.saturating_sub(1)
                    };
            }
        }

        for j in 0..table.size2 {
            let size = table.contingency[table.size1 + 1][j] as u64;
            if table.break_noise_clusters && table.noise2[j] {
                if table.self_pairing {
                    in2 += size;
                }
            } else {
                in2 += size
                    * if table.self_pairing {
                        size
                    } else {
                        size.saturating_sub(1)
                    };
            }
        }

        for i in 0..table.size1 {
            for j in 0..table.size2 {
                let size = table.contingency[i][j] as u64;
                if table.break_noise_clusters && (table.noise1[i] || table.noise2[j]) {
                    if table.self_pairing {
                        in_b += size;
                    }
                } else {
                    in_b += size
                        * if table.self_pairing {
                            size
                        } else {
                            size.saturating_sub(1)
                        };
                }
            }
        }

        let tsize = table.contingency[table.size1][table.size2] as u64;
        let total = tsize
            * if table.self_pairing {
                tsize
            } else {
                tsize.saturating_sub(1)
            };
        let in_first = in1.saturating_sub(in_b);
        let in_second = in2.saturating_sub(in_b);
        let in_none = total.saturating_sub(in_b + in_first + in_second);

        Self {
            in_both: in_b,
            in_first,
            in_second,
            in_none,
        }
    }

    pub fn f_measure(&self, beta: f64) -> f64 {
        let beta2 = beta * beta;
        let a = (1.0 + beta2) * self.in_both as f64;
        let b = a + beta2 * self.in_first as f64 + self.in_second as f64;
        if b == 0.0 { 0.0 } else { a / b }
    }

    pub fn f1_measure(&self) -> f64 {
        let a = 2.0 * self.in_both as f64;
        let b = a + self.in_first as f64 + self.in_second as f64;
        if b == 0.0 { 0.0 } else { a / b }
    }

    pub fn precision(&self) -> f64 {
        let d = (self.in_both + self.in_second) as f64;
        if d == 0.0 {
            0.0
        } else {
            self.in_both as f64 / d
        }
    }

    pub fn recall(&self) -> f64 {
        let d = (self.in_both + self.in_first) as f64;
        if d == 0.0 {
            0.0
        } else {
            self.in_both as f64 / d
        }
    }

    pub fn fowlkes_mallows(&self) -> f64 {
        (self.precision() * self.recall()).sqrt()
    }

    pub fn rand_index(&self) -> f64 {
        let d = (self.in_both + self.in_first + self.in_second + self.in_none) as f64;
        if d == 0.0 {
            0.0
        } else {
            (self.in_both + self.in_none) as f64 / d
        }
    }

    pub fn adjusted_rand_index(&self) -> f64 {
        let d = ((self.in_both + self.in_first + self.in_second + self.in_none) as f64).sqrt();
        if d == 0.0 {
            return 0.0;
        }
        let exp =
            (self.in_both + self.in_first) as f64 / d * (self.in_both + self.in_second) as f64 / d;
        let opt = self.in_both as f64 + 0.5 * (self.in_first + self.in_second) as f64;
        let denom = opt - exp;
        if denom == 0.0 {
            0.0
        } else {
            (self.in_both as f64 - exp) / denom
        }
    }

    pub fn jaccard(&self) -> f64 {
        let d = (self.in_both + self.in_first + self.in_second) as f64;
        if d == 0.0 {
            0.0
        } else {
            self.in_both as f64 / d
        }
    }

    pub fn mirkin(&self) -> u64 {
        self.in_first + self.in_second
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BCubed {
    pub precision: f64,
    pub recall: f64,
}

impl BCubed {
    pub fn new(table: &ClusterContingencyTable) -> Self {
        let selfpair = if table.self_pairing { 0usize } else { 1usize };
        let mut agg_prec = 0.0;
        let mut agg_rec = 0.0;

        for i1 in 0..table.size1 {
            let row = &table.contingency[i1];
            for i2 in 0..table.size2 {
                let c = row[i2];
                if c > selfpair {
                    agg_prec += c as f64 * (c - selfpair) as f64
                        / (table.contingency[table.size1][i2] - selfpair) as f64;
                    agg_rec +=
                        c as f64 * (c - selfpair) as f64 / (row[table.size2] - selfpair) as f64;
                }
            }
        }

        let total = table.contingency[table.size1][table.size2] as f64;
        let precision = if total == 0.0 { 0.0 } else { agg_prec / total };
        let recall = if total == 0.0 { 0.0 } else { agg_rec / total };
        Self { precision, recall }
    }

    pub fn f1_measure(&self) -> f64 {
        f1_measure(self.precision, self.recall)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EditDistance {
    pub edit_first: usize,
    pub edit_second: usize,
    pub baseline: usize,
}

impl EditDistance {
    pub fn new(table: &ClusterContingencyTable) -> Self {
        let r = table.size1;
        let c = table.size2;
        let mut ed1 = 0usize;
        let mut ed2 = 0usize;

        for i in 0..c {
            let csize = table.contingency[r][i];
            if csize > 0 {
                let mut largest = 0usize;
                for j in 0..r {
                    largest = largest.max(table.contingency[j][i]);
                }
                ed1 += 1 + csize - largest;
            }
        }

        for i in 0..r {
            let csize = table.contingency[i][c];
            if csize > 0 {
                let mut largest = 0usize;
                for j in 0..c {
                    largest = largest.max(table.contingency[i][j]);
                }
                ed2 += 1 + csize - largest;
            }
        }

        Self {
            edit_first: ed1,
            edit_second: ed2,
            baseline: table.contingency[r][c],
        }
    }

    pub fn edit_distance_first(&self) -> f64 {
        if self.baseline == 0 {
            0.0
        } else {
            1.0 - self.edit_first as f64 / self.baseline as f64
        }
    }

    pub fn edit_distance_second(&self) -> f64 {
        if self.baseline == 0 {
            0.0
        } else {
            1.0 - self.edit_second as f64 / self.baseline as f64
        }
    }

    pub fn f1_measure(&self) -> f64 {
        f1_measure(self.edit_distance_first(), self.edit_distance_second())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SetMatchingPurity {
    pub purity: f64,
    pub inverse_purity: f64,
    pub f_first: f64,
    pub f_second: f64,
}

impl SetMatchingPurity {
    pub fn new(table: &ClusterContingencyTable) -> Self {
        let r = table.size1;
        let c = table.size2;
        let mut agg_purity = 0.0;
        let mut agg_first = 0.0;

        for i in 0..r {
            let mut precision_max = 0.0f64;
            let mut f_max = 0.0f64;
            for j in 0..c {
                let x = table.contingency[i][j] as f64;
                precision_max = precision_max.max(x);
                let denom = (table.contingency[i][c] + table.contingency[r][j]) as f64;
                if denom > 0.0 {
                    f_max = f_max.max((2.0 * x) / denom);
                }
            }
            agg_purity += precision_max;
            agg_first += table.contingency[i][c] as f64 * f_max;
        }

        let mut agg_inv_p = 0.0;
        let mut agg_second = 0.0;
        for i in 0..c {
            let mut recall_max = 0.0f64;
            let mut f_max = 0.0f64;
            for j in 0..r {
                let x = table.contingency[j][i] as f64;
                recall_max = recall_max.max(x);
                let denom = (table.contingency[j][c] + table.contingency[r][i]) as f64;
                if denom > 0.0 {
                    f_max = f_max.max((2.0 * x) / denom);
                }
            }
            agg_inv_p += recall_max;
            agg_second += table.contingency[r][i] as f64 * f_max;
        }

        let n = table.contingency[r][c] as f64;
        if n == 0.0 {
            return Self {
                purity: 0.0,
                inverse_purity: 0.0,
                f_first: 0.0,
                f_second: 0.0,
            };
        }

        Self {
            purity: agg_purity / n,
            inverse_purity: agg_inv_p / n,
            f_first: agg_first / n,
            f_second: agg_second / n,
        }
    }

    pub fn f1_measure(&self) -> f64 {
        f1_measure(self.purity, self.inverse_purity)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MaximumMatchingAccuracy {
    pub accuracy: f64,
}

impl MaximumMatchingAccuracy {
    pub fn new(table: &ClusterContingencyTable) -> Self {
        let rowlen = table.size1;
        let collen = table.size2;
        let maxlen = rowlen.max(collen);
        if maxlen == 0 {
            return Self { accuracy: 0.0 };
        }

        let mut costs = vec![vec![0.0; maxlen]; maxlen];
        for (i, row) in costs.iter_mut().enumerate().take(rowlen) {
            for (j, cell) in row.iter_mut().enumerate().take(collen) {
                *cell = -(table.contingency[i][j] as f64);
            }
        }

        let chosen = hungarian_min_cost_assignment(&costs);
        let mut correct = 0.0;
        for (i, &j) in chosen.iter().enumerate().take(rowlen) {
            if j < collen {
                correct += table.contingency[i][j] as f64;
            }
        }
        let n = table.contingency[rowlen][collen] as f64;
        let accuracy = if n == 0.0 { 0.0 } else { correct / n };
        Self { accuracy }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PairSetsIndex {
    pub simplified_psi: f64,
    pub psi: f64,
}

impl PairSetsIndex {
    pub fn new(table: &ClusterContingencyTable) -> Self {
        let rowlen = table.size1;
        let collen = table.size2;

        if rowlen == 1 && collen == 1 {
            return Self {
                simplified_psi: 1.0,
                psi: 1.0,
            };
        }

        let maxlen = rowlen.max(collen);
        if maxlen == 0 {
            return Self {
                simplified_psi: 0.0,
                psi: 0.0,
            };
        }

        let mut costs = vec![vec![0.0; maxlen]; maxlen];
        for (i, row) in costs.iter_mut().enumerate().take(rowlen) {
            let rowsum = table.contingency[i][collen];
            if rowsum > 0 {
                for (j, cell) in row.iter_mut().enumerate().take(collen) {
                    let x = table.contingency[i][j];
                    if x > 0 {
                        *cell = -(x as f64 / table.contingency[rowlen][j].max(rowsum) as f64);
                    }
                }
            }
        }

        let chosen = hungarian_min_cost_assignment(&costs);
        let mut s = 0.0;
        for i in 0..maxlen {
            s += -costs[i][chosen[i]];
        }

        let mut first_sizes: Vec<usize> =
            (0..rowlen).map(|i| table.contingency[i][collen]).collect();
        let mut second_sizes: Vec<usize> =
            (0..collen).map(|j| table.contingency[rowlen][j]).collect();
        first_sizes.sort_unstable();
        second_sizes.sort_unstable();

        let n = table.contingency[rowlen][collen] as f64;
        let minlength = rowlen.min(collen);
        let mut e = 0.0;
        if n > 0.0 {
            for i in 0..minlength {
                let a = first_sizes[i] as f64;
                let b = second_sizes[i] as f64;
                let denom = a.max(b);
                if denom > 0.0 {
                    e += (a * b / n) / denom;
                }
            }
        }

        let denom_simple = maxlen as f64 - 1.0;
        let simplified_psi = if s < 1.0 || denom_simple <= 0.0 {
            0.0
        } else {
            (s - 1.0) / denom_simple
        };

        let denom = maxlen as f64 - e;
        let psi = if s < e || denom <= 0.0 {
            0.0
        } else {
            (s - e) / denom
        };

        Self {
            simplified_psi,
            psi,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Entropy {
    pub entropy_first: f64,
    pub entropy_second: f64,
    pub entropy_joint: f64,
    pub mutual_information: f64,
    pub variation_of_information: f64,
    pub expected_mutual_information: f64,
    vibound: f64,
}

impl Entropy {
    pub fn new(table: &ClusterContingencyTable) -> Self {
        let r = table.size1;
        let c = table.size2;
        let n = table.contingency[r][c];
        if n == 0 {
            return Self {
                entropy_first: 0.0,
                entropy_second: 0.0,
                entropy_joint: 0.0,
                mutual_information: 0.0,
                variation_of_information: 0.0,
                expected_mutual_information: 0.0,
                vibound: 0.0,
            };
        }

        let byn = 1.0 / n as f64;
        let mlogn = -(n as f64).ln();

        if n <= 10_000 {
            let m = max_cluster_size(&table.contingency, r, c);
            let mut logs = vec![0.0f64; m.max(2)];
            let entropy_first =
                compute_entropy_first(&table.contingency, r, c, byn, mlogn, &mut logs);
            let entropy_second =
                compute_entropy_second(&table.contingency, r, c, byn, mlogn, &mut logs);
            let (
                entropy_joint,
                mutual_information,
                variation_of_information,
                expected_mutual_information,
            ) = compute_mi_full(&table.contingency, r, c, n, byn, mlogn, &mut logs);
            let vibound = (-mlogn).min(2.0 * (r.max(c) as f64).ln());
            Self {
                entropy_first,
                entropy_second,
                entropy_joint,
                mutual_information,
                variation_of_information,
                expected_mutual_information,
                vibound,
            }
        } else {
            let (
                entropy_first,
                entropy_second,
                entropy_joint,
                mutual_information,
                variation_of_information,
            ) = compute_mi_large(&table.contingency, r, c, byn, mlogn);
            let vibound = (-mlogn).min(2.0 * (r.max(c) as f64).ln());
            Self {
                entropy_first,
                entropy_second,
                entropy_joint,
                mutual_information,
                variation_of_information,
                expected_mutual_information: 0.0,
                vibound,
            }
        }
    }

    pub fn conditional_entropy_first(&self) -> f64 {
        self.entropy_joint - self.entropy_second
    }

    pub fn conditional_entropy_second(&self) -> f64 {
        self.entropy_joint - self.entropy_first
    }

    pub fn entropy_powers(&self) -> f64 {
        let denom = self.entropy_first + self.entropy_second;
        if denom == 0.0 {
            0.0
        } else {
            2.0 * self.entropy_joint / denom - 1.0
        }
    }

    pub fn upper_bound_mi(&self) -> f64 {
        self.entropy_first.min(self.entropy_second)
    }

    pub fn joint_nmi(&self) -> f64 {
        if self.entropy_joint == 0.0 {
            0.0
        } else {
            self.mutual_information / self.entropy_joint
        }
    }

    pub fn min_nmi(&self) -> f64 {
        let denom = self.entropy_first.min(self.entropy_second);
        if denom == 0.0 {
            0.0
        } else {
            self.mutual_information / denom
        }
    }

    pub fn max_nmi(&self) -> f64 {
        let denom = self.entropy_first.max(self.entropy_second);
        if denom == 0.0 {
            0.0
        } else {
            self.mutual_information / denom
        }
    }

    pub fn arithmetic_nmi(&self) -> f64 {
        let denom = self.entropy_first + self.entropy_second;
        if denom == 0.0 {
            0.0
        } else {
            2.0 * self.mutual_information / denom
        }
    }

    pub fn geometric_nmi(&self) -> f64 {
        if self.entropy_first * self.entropy_second <= 0.0 {
            self.mutual_information
        } else {
            self.mutual_information / (self.entropy_first * self.entropy_second).sqrt()
        }
    }

    pub fn upper_bound_vi(&self) -> f64 {
        self.vibound
    }

    pub fn normalized_variation_of_information(&self) -> f64 {
        if self.entropy_joint == 0.0 {
            0.0
        } else {
            1.0 - self.mutual_information / self.entropy_joint
        }
    }

    pub fn normalized_information_distance(&self) -> f64 {
        let denom = self.entropy_first.max(self.entropy_second);
        if denom == 0.0 {
            0.0
        } else {
            1.0 - self.mutual_information / denom
        }
    }

    pub fn adjusted_joint_mi(&self) -> f64 {
        let denom = self.entropy_joint - self.expected_mutual_information;
        if denom == 0.0 {
            0.0
        } else {
            (self.mutual_information - self.expected_mutual_information) / denom
        }
    }

    pub fn adjusted_arithmetic_mi(&self) -> f64 {
        let denom =
            0.5 * (self.entropy_first + self.entropy_second) - self.expected_mutual_information;
        if denom == 0.0 {
            0.0
        } else {
            (self.mutual_information - self.expected_mutual_information) / denom
        }
    }

    pub fn adjusted_geometric_mi(&self) -> f64 {
        if self.entropy_first * self.entropy_second <= 0.0 {
            self.mutual_information - self.expected_mutual_information
        } else {
            let denom = (self.entropy_first * self.entropy_second).sqrt()
                - self.expected_mutual_information;
            if denom == 0.0 {
                0.0
            } else {
                (self.mutual_information - self.expected_mutual_information) / denom
            }
        }
    }

    pub fn adjusted_min_mi(&self) -> f64 {
        let denom = self.entropy_first.min(self.entropy_second) - self.expected_mutual_information;
        if denom == 0.0 {
            0.0
        } else {
            (self.mutual_information - self.expected_mutual_information) / denom
        }
    }

    pub fn adjusted_max_mi(&self) -> f64 {
        let denom = self.entropy_first.max(self.entropy_second) - self.expected_mutual_information;
        if denom == 0.0 {
            0.0
        } else {
            (self.mutual_information - self.expected_mutual_information) / denom
        }
    }
}

fn max_cluster_size(contingency: &[Vec<usize>], r: usize, c: usize) -> usize {
    let mut maxc = 0usize;
    for j in 0..c {
        maxc = maxc.max(contingency[r][j]);
    }
    for row in contingency.iter().take(r) {
        maxc = maxc.max(row[c]);
    }
    maxc
}

fn log_cached(i: usize, logs: &mut [f64]) -> f64 {
    if i <= 1 {
        return 0.0;
    }
    let idx = i - 2;
    if idx >= logs.len() {
        return (i as f64).ln();
    }
    let v = logs[idx];
    if v > 0.0 {
        v
    } else {
        let ln = (i as f64).ln();
        logs[idx] = ln;
        ln
    }
}

fn compute_entropy_first(
    contingency: &[Vec<usize>],
    r: usize,
    c: usize,
    byn: f64,
    mlogn: f64,
    logs: &mut [f64],
) -> f64 {
    let mut e = 0.0;
    for row in contingency.iter().take(r) {
        let v = row[c];
        if v > 0 {
            e += v as f64 * byn * (-log_cached(v, logs) - mlogn);
        }
    }
    e
}

fn compute_entropy_second(
    contingency: &[Vec<usize>],
    r: usize,
    c: usize,
    byn: f64,
    mlogn: f64,
    logs: &mut [f64],
) -> f64 {
    let mut e = 0.0;
    for j in 0..c {
        let v = contingency[r][j];
        if v > 0 {
            e += v as f64 * byn * (-log_cached(v, logs) - mlogn);
        }
    }
    e
}

fn compute_mi_large(
    contingency: &[Vec<usize>],
    r: usize,
    c: usize,
    byn: f64,
    mlogn: f64,
) -> (f64, f64, f64, f64, f64) {
    let mut logs = vec![0.0f64; 14];
    let mut mlogbn = vec![0.0f64; c];
    let mut ent1 = 0.0;
    let mut ent2 = 0.0;
    let mut joint = 0.0;
    let mut mi = 0.0;
    let mut vi = 0.0;

    for (j, val) in contingency[r].iter().copied().enumerate().take(c) {
        if val > 0 {
            mlogbn[j] = -log_cached(val, &mut logs) - mlogn;
            ent2 += val as f64 * byn * mlogbn[j];
        }
    }

    for row in contingency.iter().take(r) {
        let an = row[c];
        if an == 0 {
            continue;
        }
        let mlogain = -log_cached(an, &mut logs) - mlogn;
        ent1 += an as f64 * byn * mlogain;
        for j in 0..c {
            let vij = row[j];
            if vij > 0 {
                let p = vij as f64 * byn;
                let mlogp = -log_cached(vij, &mut logs) - mlogn;
                let mlogbjn = mlogbn[j];
                joint += p * mlogp;
                mi += p * (mlogain + mlogbjn - mlogp);
                vi += p * (mlogp - mlogain + mlogp - mlogbjn);
            }
        }
    }

    (ent1, ent2, joint, mi, vi)
}

fn lfac(i: usize, lfacs: &[f64]) -> f64 {
    if i <= 1 { 0.0 } else { lfacs[i] }
}

fn compute_mi_full(
    contingency: &[Vec<usize>],
    r: usize,
    c: usize,
    n: usize,
    byn: f64,
    mlogn: f64,
    logs: &mut [f64],
) -> (f64, f64, f64, f64) {
    let mut lfacs = vec![0.0f64; n + 1];
    for i in 2..=n {
        lfacs[i] = lfacs[i - 1] + log_cached(i, logs);
    }

    let mut joint = 0.0;
    let mut mi = 0.0;
    let mut vi = 0.0;
    let mut emi = 0.0;

    for row in contingency.iter().take(r) {
        let ai = row[c];
        if ai == 0 {
            continue;
        }
        let mlogain = -log_cached(ai, logs) - mlogn;
        let lfac_ai = lfac(ai, &lfacs);
        let lfac_n_minus_ai = lfac(n - ai, &lfacs);

        for j in 0..c {
            let bj = contingency[r][j];
            if bj == 0 {
                continue;
            }
            let vij = row[j];
            let mlogbjn = -log_cached(bj, logs) - mlogn;
            if vij > 0 {
                let p = vij as f64 * byn;
                let mlogp = -log_cached(vij, logs) - mlogn;
                joint += p * mlogp;
                mi += p * (mlogain + mlogbjn - mlogp);
                vi += p * (mlogp - mlogain + mlogp - mlogbjn);
            }

            let start = (ai + bj).saturating_sub(n).max(1);
            let end = ai.min(bj);
            if start <= end {
                let t1 = mlogain + mlogbjn + mlogn;
                let tail = (n as isize - ai as isize - bj as isize + start as isize) as usize;
                let mut t2 = (lfac_ai + lfac(bj, &lfacs) + lfac_n_minus_ai + lfac(n - bj, &lfacs)
                    - lfac(n, &lfacs)
                    - lfac(start, &lfacs)
                    - lfac(ai - start, &lfacs)
                    - lfac(bj - start, &lfacs)
                    - lfac(tail, &lfacs))
                .exp();

                emi += start as f64 * byn * (t1 + log_cached(start, logs)) * t2;
                for nij in (start + 1)..=end {
                    let tail_term = (n as isize - ai as isize - bj as isize + nij as isize) as f64;
                    t2 *= (ai - nij + 1) as f64 * (bj - nij + 1) as f64 / (nij as f64 * tail_term);
                    if t2 < 0.0 {
                        break;
                    }
                    emi += nij as f64 * byn * (t1 + log_cached(nij, logs)) * t2;
                }
            }
        }
    }

    (joint, mi, vi, emi)
}

fn hungarian_min_cost_assignment(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let m = cost[0].len();
    assert_eq!(n, m, "Hungarian solver requires a square matrix");

    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0usize; n];
    for j in 1..=m {
        if p[j] > 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }
    assignment
}

#[derive(Debug, Clone, Copy)]
pub struct ClusteringEvaluation {
    pub pair_counting: PairCounting,
    pub entropy: Entropy,
    pub bcubed: BCubed,
    pub set_matching_purity: SetMatchingPurity,
    pub maximum_matching_accuracy: MaximumMatchingAccuracy,
    pub pair_sets_index: PairSetsIndex,
    pub edit_distance: EditDistance,
}

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
        edit_distance: table.edit_distance(),
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

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "{a} !~= {b}");
    }

    #[test]
    fn perfect_match_scores_are_perfect() {
        let l1 = [0, 0, 1, 1, 2, 2];
        let l2 = [2, 2, 1, 1, 0, 0];
        let e = evaluate_clustering(&l1, &l2, false, false, None, None);

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
