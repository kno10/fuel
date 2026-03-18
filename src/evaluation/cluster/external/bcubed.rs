use super::contingency_table::ClusterContingencyTable;
use crate::evaluation::cluster::f1_measure;

/// BCubed precision/recall metrics derived from the contingency table.
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
