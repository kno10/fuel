use super::assignment::hungarian_min_cost_assignment;
use super::contingency_table::ClusterContingencyTable;

/// Accuracy obtained by solving the optimal matching between clusters.
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
