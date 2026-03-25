#![allow(clippy::cast_precision_loss)]

use crate::evaluation::cluster::external::contingency_table::ClusterContingencyTable;
use crate::evaluation::cluster::f1_measure;

/// Purity metrics based on set matching of contingency tables.
#[derive(Debug, Clone, Copy)]
pub struct SetMatchingPurity {
    pub purity: f64,
    pub inverse_purity: f64,
    pub f_first: f64,
    pub f_second: f64,
}

impl SetMatchingPurity {
    pub fn new(table: &ClusterContingencyTable) -> SetMatchingPurity {
        let r = table.size1;
        let c = table.size2;
        let mut agg_purity: f64 = 0.0;
        let mut agg_first: f64 = 0.0;

        for i in 0..r {
            let mut precision_max: f64 = 0.0;
            let mut f_max: f64 = 0.0;
            for j in 0..c {
                let x = table.contingency[i][j] as f64;
                precision_max = precision_max.max(x);
                let denom = (table.contingency[i][c] + table.contingency[r][j]) as f64;
                if denom > 0.0 {
                    f_max = f_max.max((2.0 * x) / denom);
                }
            }
            agg_purity += precision_max;
            agg_first += (table.contingency[i][c] as f64) * f_max;
        }

        let mut agg_inv_p: f64 = 0.0;
        let mut agg_second: f64 = 0.0;
        for i in 0..c {
            let mut recall_max: f64 = 0.0;
            let mut f_max: f64 = 0.0;
            for j in 0..r {
                let x = table.contingency[j][i] as f64;
                recall_max = recall_max.max(x);
                let denom = (table.contingency[j][c] + table.contingency[r][i]) as f64;
                if denom > 0.0 {
                    f_max = f_max.max((2.0 * x) / denom);
                }
            }
            agg_inv_p += recall_max;
            agg_second += (table.contingency[r][i] as f64) * f_max;
        }

        let n = table.contingency[r][c] as f64;
        if n == 0.0 {
            return SetMatchingPurity {
                purity: 0.0,
                inverse_purity: 0.0,
                f_first: 0.0,
                f_second: 0.0,
            };
        }

        SetMatchingPurity {
            purity: agg_purity / n,
            inverse_purity: agg_inv_p / n,
            f_first: agg_first / n,
            f_second: agg_second / n,
        }
    }

    pub fn f1_measure(&self) -> f64 { f1_measure(self.purity, self.inverse_purity) }
}
