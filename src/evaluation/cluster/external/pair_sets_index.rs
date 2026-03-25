#![allow(clippy::cast_precision_loss)]

use crate::evaluation::cluster::external::assignment::hungarian_min_cost_assignment;
use crate::evaluation::cluster::external::contingency_table::ClusterContingencyTable;

/// Pair‑sets index metric from cluster contingency tables.
#[derive(Debug, Clone, Copy)]
pub struct PairSetsIndex {
    pub simplified_psi: f64,
    pub psi: f64,
}

impl PairSetsIndex {
    pub fn new(table: &ClusterContingencyTable) -> PairSetsIndex {
        let rowlen = table.size1;
        let collen = table.size2;

        if rowlen == 1 && collen == 1 {
            return PairSetsIndex { simplified_psi: 1.0, psi: 1.0 };
        }

        let maxlen = rowlen.max(collen);
        if maxlen == 0 {
            return PairSetsIndex { simplified_psi: 0.0, psi: 0.0 };
        }

        // note: internal computations still use f64 because distance metric
        // requires floating point; we convert at the end.
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
        let simplified_psi =
            if s < 1.0 || denom_simple <= 0.0 { 0.0 } else { (s - 1.0) / denom_simple };

        let denom = maxlen as f64 - e;
        let psi = if s < e || denom <= 0.0 { 0.0 } else { (s - e) / denom };

        PairSetsIndex { simplified_psi, psi }
    }
}
