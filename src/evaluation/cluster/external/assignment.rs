/// Helper for optimal assignment used by several metrics.
/// Hungarian algorithm for the minimum cost assignment problem.
///
/// The cost matrix is square (n×n). Rows correspond to one set of items and
/// columns to the other; the returned vector maps each row index to its chosen
/// column. If the matrix is empty, an empty vector is returned.
pub(crate) fn hungarian_min_cost_assignment(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }

    let m = cost[0].len();
    let maxm = n.max(m);

    // we work with 1-based indices internally to match pseudo-code
    let mut u = vec![0.0; maxm + 1];
    let mut v = vec![0.0; maxm + 1];
    let mut p = vec![0usize; maxm + 1];
    let mut way = vec![0usize; maxm + 1];

    for i in 1..=n {
        p[0] = i;
        let mut minv = vec![f64::INFINITY; maxm + 1];
        let mut used = vec![false; maxm + 1];
        let mut j0 = 0;

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;

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
