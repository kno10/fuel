#![allow(clippy::cast_precision_loss)]

use std::cmp::Ordering;

use super::helpers::{NeighborConsistencyStats, euc};

/// K‑nearest‑neighbour based neighbor consistency statistics.
#[must_use]
pub fn neighbor_consistency_knn(
    data: &[Vec<f64>], labels: &[isize], k: usize,
) -> NeighborConsistencyStats<f64> {
    assert_eq!(data.len(), labels.len());
    let n = data.len();
    if n == 0 || k == 0 {
        return NeighborConsistencyStats {
            average: 0.0,
            full: 0.0,
            per_element_average: vec![0.0; n],
            per_element_full: vec![0.0; n],
        };
    }

    let mut per_avg = vec![0.0; n];
    let mut per_full = vec![0.0; n];

    for i in 0..n {
        let mut ds = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i != j {
                ds.push((euc(&data[i], &data[j]), j));
            }
        }
        ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let kk = k.min(ds.len());

        let mut same = 0usize;
        for (_, j) in ds.iter().take(kk) {
            if labels[*j] == labels[i] {
                same += 1;
            }
        }

        let frac = same as f64 / kk.max(1) as f64;
        per_avg[i] = frac;
        per_full[i] = if same == kk { 1.0 } else { 0.0 };
    }

    NeighborConsistencyStats {
        average: per_avg.iter().sum::<f64>() / n as f64,
        full: per_full.iter().sum::<f64>() / n as f64,
        per_element_average: per_avg,
        per_element_full: per_full,
    }
}
