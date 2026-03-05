use crate::cluster::MergeHistory;

fn triangle_index(i: usize, j: usize) -> usize {
    (i * (i - 1)) / 2 + j
}

pub fn cophenetic_distances(history: &MergeHistory<f64>, n: usize) -> Vec<f64> {
    if n <= 1 {
        return Vec::new();
    }
    let mut distances = vec![0.0; n * (n - 1) / 2];
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    for merge in history.iter() {
        let a_members = &clusters[merge.idx1];
        let b_members = &clusters[merge.idx2];
        for &a in a_members {
            for &b in b_members {
                let (i, j) = if a > b { (a, b) } else { (b, a) };
                distances[triangle_index(i, j)] = merge.distance;
            }
        }
        let mut merged = a_members.clone();
        merged.extend(b_members.iter().copied());
        clusters.push(merged);
    }
    distances
}

pub fn cophenetic_correlation(
    base: &MergeHistory<f64>,
    other: &MergeHistory<f64>,
    n: usize,
) -> f64 {
    let baseline = cophenetic_distances(base, n);
    let counterpart = cophenetic_distances(other, n);
    pearson_correlation(&baseline, &counterpart)
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let len = x.len();
    if len == 0 {
        return 1.0;
    }
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|&a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|&b| b * b).sum();
    let n = len as f64;
    let numerator = n * sum_xy - sum_x * sum_y;
    let denom_left = n * sum_x2 - sum_x * sum_x;
    let denom_right = n * sum_y2 - sum_y * sum_y;
    let denom = denom_left * denom_right;
    if denom <= 0.0 {
        return 0.0;
    }
    numerator / denom.sqrt()
}
