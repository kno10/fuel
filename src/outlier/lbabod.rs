use std::cmp::Ordering;

use super::abod::abod_kernel_score_for_neighbor_set;
use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float, VectorData};

fn dot_product<F: Float>(left: &[F], right: &[F]) -> F {
    left.iter().zip(right.iter()).map(|(&a, &b)| a * b).sum()
}

fn compute_lbabof_kernel<D, F, K>(data: &D, center: usize, k: usize, kernel: &K) -> F
where
    F: Float,
    D: DistanceData<F> + VectorData<F>,
    K: Fn(&[F], &[F]) -> F,
{
    let size = data.len();
    let xi = data.point(center);
    let sim_ii = kernel(xi, xi);

    let mut sumid = F::zero();
    let mut sumisqd = F::zero();
    let mut neighbors: Vec<(F, usize, F)> = Vec::with_capacity(size.saturating_sub(1));

    for j in 0..size {
        if j == center {
            continue;
        }
        let xj = data.point(j);
        let sim_jj = kernel(xj, xj);
        let sim_ij = kernel(xi, xj);
        let sqd_ab = sim_ii + sim_jj - sim_ij - sim_ij;

        if sqd_ab > F::zero() {
            let isqd = F::one() / sqd_ab;
            let sqrt_isqd = isqd.sqrt();
            sumid += &sqrt_isqd;
            sumisqd += &isqd;
        }
        neighbors.push((sqd_ab, j, sim_ij));
    }

    neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let k_effective = k.min(neighbors.len());
    if k_effective < 2 || sumid <= F::zero() {
        return F::infinity();
    }

    let mut nnsum = F::zero();
    let mut nnsumsq = F::zero();
    let mut nnsumisqd = F::zero();

    for a in 0..k_effective {
        let (sqd_ab, idx_ab, sim_ab) = neighbors[a];
        if sqd_ab.is_nan() || sqd_ab <= F::zero() {
            continue;
        }
        for (_b, &(sqd_ac, idx_ac, sim_ac)) in
            neighbors.iter().enumerate().skip(a + 1).take(k_effective - a - 1)
        {
            if sqd_ac.is_nan() || sqd_ac <= F::zero() {
                continue;
            }
            let x_ab = data.point(idx_ab);
            let x_ac = data.point(idx_ac);
            let sim_bc = kernel(x_ab, x_ac);
            let numerator = sim_bc - sim_ab - sim_ac + sim_ii;
            let sqweight = F::one() / (sqd_ab * sqd_ac);
            let weight = sqweight.sqrt();
            let value = numerator * sqweight;

            let value_weight = value * weight;
            let value_sq_weight = value * value * weight;
            nnsum += &value_weight;
            nnsumsq += &value_sq_weight;
            nnsumisqd += &sqweight;
        }
    }

    let denom = sumid * sumid;
    if denom.is_nan() || denom <= F::zero() {
        return F::infinity();
    }

    let r2 = sumisqd * sumisqd - F::two() * nnsumisqd;
    let tmp = (F::two() * nnsum + r2) / denom;
    F::two() * nnsumsq / denom - tmp * tmp
}

fn max_value<F: Float>(values: &[F]) -> Option<F> {
    values.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
}

/// LB-ABOD: lower-bound ABOD (approximate + refinement).
pub fn locality_based_abod<D, F>(data: &D, k: usize, l: usize) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + VectorData<F> + Sync,
{
    locality_based_abod_kernel(data, k, l, |x, y| dot_product(x, y))
}

/// LBABOD for kernel similarity variant.
pub fn locality_based_abod_kernel<D, F, K>(
    data: &D, k: usize, l: usize, kernel: K,
) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + VectorData<F> + Sync,
    K: Fn(&[F], &[F]) -> F,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "LBABOD",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let mut lbabof_scores = Vec::with_capacity(size);
    for i in 0..size {
        lbabof_scores.push(compute_lbabof_kernel(data, i, k, &kernel));
    }

    let mut scores = lbabof_scores.clone();
    let mut ranked: Vec<usize> = (0..size).collect();
    ranked.sort_by(|&a, &b| {
        lbabof_scores[a].partial_cmp(&lbabof_scores[b]).unwrap_or(Ordering::Equal)
    });

    let refine_count = l.min(size);
    let mut top_exact: Vec<F> = Vec::new();

    for idx in ranked {
        if refine_count == 0 {
            break;
        }

        if let Some(worst_exact) = max_value(&top_exact)
            && top_exact.len() >= refine_count
            && lbabof_scores[idx] > worst_exact
        {
            break;
        }

        let exact_score = abod_kernel_score_for_neighbor_set(
            data,
            idx,
            &(0..size).filter(|&j| j != idx).collect::<Vec<_>>(),
            &kernel,
        );
        scores[idx] = exact_score;

        if top_exact.len() < refine_count {
            top_exact.push(exact_score);
        } else if let Some(worst_exact) = max_value(&top_exact)
            && exact_score < worst_exact
            && let Some(pos) =
                top_exact.iter().position(|&x| x.partial_cmp(&worst_exact) == Some(Ordering::Equal))
        {
            top_exact[pos] = exact_score;
        }
    }

    make_outlier_result(scores, "LBABOD", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auroc;
    use crate::outlier::common::*;

    #[test]
    fn lbabod_10_poly2_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);

        let kernel = crate::kernel::polynomial::PolynomialKernel::new(2, 1.0, 0.0);
        let result = locality_based_abod_kernel(&data, 10, 10, |x, y| kernel.similarity(x, y));

        let reference = load_reference_scores();
        let expected = reference.get("LBABOD-10-poly2").expect("No reference for LBABOD-10-poly2");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "LBABOD-10-poly2",
            auroc(&result.scores, &labels),
            auroc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("LBABOD-10-poly2", &result.scores, expected, 1e-6);
    }
}
