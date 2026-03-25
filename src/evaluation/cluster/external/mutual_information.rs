use crate::evaluation::cluster::external::contingency_table::ClusterContingencyTable;

/// Entropy and mutual information measures derived from a contingency table.
/// All results are stored as `f64`; this metric is not driven by any distance
/// function so there is no need for a generic float parameter.
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
    pub fn new(table: &ClusterContingencyTable) -> Entropy {
        let r = table.size1;
        let c = table.size2;
        let n = table.contingency[r][c];
        if n == 0 {
            return Entropy {
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
            Entropy {
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
            Entropy {
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

    pub fn conditional_entropy_first(&self) -> f64 { self.entropy_joint - self.entropy_second }

    pub fn conditional_entropy_second(&self) -> f64 { self.entropy_joint - self.entropy_first }

    pub fn entropy_powers(&self) -> f64 {
        let denom = self.entropy_first + self.entropy_second;
        if denom == 0.0 { 0.0 } else { 2.0 * self.entropy_joint / denom - 1.0 }
    }

    pub fn upper_bound_mi(&self) -> f64 { self.entropy_first.min(self.entropy_second) }

    pub fn joint_nmi(&self) -> f64 {
        if self.entropy_joint == 0.0 { 0.0 } else { self.mutual_information / self.entropy_joint }
    }

    pub fn min_nmi(&self) -> f64 {
        let denom = self.entropy_first.min(self.entropy_second);
        if denom == 0.0 { 0.0 } else { self.mutual_information / denom }
    }

    pub fn max_nmi(&self) -> f64 {
        let denom = self.entropy_first.max(self.entropy_second);
        if denom == 0.0 { 0.0 } else { self.mutual_information / denom }
    }

    pub fn arithmetic_nmi(&self) -> f64 {
        let denom = self.entropy_first + self.entropy_second;
        if denom == 0.0 { 0.0 } else { 2.0 * self.mutual_information / denom }
    }

    pub fn geometric_nmi(&self) -> f64 {
        let denom = (self.entropy_first * self.entropy_second).sqrt();
        if denom == 0.0 { 0.0 } else { self.mutual_information / denom }
    }

    pub fn upper_bound_vi(&self) -> f64 { self.vibound }

    pub fn normalized_variation_of_information(&self) -> f64 {
        if self.entropy_joint == 0.0 {
            0.0
        } else {
            self.variation_of_information / self.entropy_joint
        }
    }

    pub fn normalized_information_distance(&self) -> f64 {
        if self.entropy_joint == 0.0 {
            0.0
        } else {
            (2.0 * self.entropy_joint - self.mutual_information) / self.entropy_joint
        }
    }
}

// helper functions previously defined in mod.rs follow.

fn max_cluster_size(contingency: &[Vec<usize>], r: usize, c: usize) -> usize {
    let mut maxc = 0usize;
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
    contingency: &[Vec<usize>], r: usize, c: usize, byn: f64, mlogn: f64, logs: &mut [f64],
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
    contingency: &[Vec<usize>], r: usize, c: usize, byn: f64, mlogn: f64, logs: &mut [f64],
) -> f64 {
    let mut e = 0.0;
    for v in contingency[r].iter().take(c).copied() {
        if v > 0 {
            let v_f = v as f64;
            e += v_f * byn * (-log_cached(v, logs) - mlogn);
        }
    }
    e
}

fn compute_mi_large(
    contingency: &[Vec<usize>], r: usize, c: usize, byn: f64, mlogn: f64,
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

fn lfac(i: usize, lfacs: &[f64]) -> f64 { if i <= 1 { 0.0 } else { lfacs[i] } }

fn compute_mi_full(
    contingency: &[Vec<usize>], r: usize, c: usize, n: usize, byn: f64, mlogn: f64,
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
                let tail = n.saturating_sub(ai + bj).saturating_add(start);
                let mut t2 = (lfac_ai + lfac(bj, &lfacs) + lfac_n_minus_ai + lfac(n - bj, &lfacs)
                    - lfac(n, &lfacs)
                    - lfac(start, &lfacs)
                    - lfac(ai - start, &lfacs)
                    - lfac(bj - start, &lfacs)
                    - lfac(tail, &lfacs))
                .exp();

                emi += start as f64 * byn * (t1 + log_cached(start, logs)) * t2;
                for nij in (start + 1)..=end {
                    let tail_term = (n.saturating_sub(ai + bj).saturating_add(nij)) as f64;
                    #[allow(clippy::cast_precision_loss)]
                    let term =
                        (ai - nij + 1) as f64 * (bj - nij + 1) as f64 / (nij as f64 * tail_term);
                    t2 *= term;
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
