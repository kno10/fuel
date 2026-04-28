use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float, VectorData};

/// Exact ABOD (angle-based outlier factor) using a kernel similarity function.
pub fn angle_based_outlier_detection<D, F, K>(data: &D, kernel: K) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + VectorData<F> + Sync,
    K: Fn(&[F], &[F]) -> F + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "ABOD", false, F::zero(), F::zero(), F::infinity());
    }

    let mut scores = Vec::with_capacity(size);
    for i in 0..size {
        let neighbors: Vec<usize> = (0..size).filter(|&j| j != i).collect();
        scores.push(abod_kernel_score_for_neighbor_set(data, i, &neighbors, &kernel));
    }

    make_outlier_result(scores, "ABOD", false, F::zero(), F::zero(), F::infinity())
}

/// Compute ABOF for center with kernel similarity.
pub fn abod_kernel_score_for_neighbor_set<D, F, K>(
    data: &D, center: usize, neighbors: &[usize], kernel: &K,
) -> F
where
    F: Float,
    D: DistanceData<F> + VectorData<F>,
    K: Fn(&[F], &[F]) -> F,
{
    let xi = data.point(center);
    let sim_aa = kernel(xi, xi);

    struct Neighbor<F> {
        index: usize,
        sim_bb: F,
        sim_ab: F,
        sqd_ab: F,
    }

    let mut valid_neighbors = Vec::with_capacity(neighbors.len());
    for &j in neighbors {
        if j == center {
            continue;
        }
        let xj = data.point(j);
        let sim_bb = kernel(xj, xj);
        let sim_ab = kernel(xi, xj);
        let sqd_ab = sim_aa + sim_bb - sim_ab - sim_ab;
        if sqd_ab > F::zero() {
            valid_neighbors.push(Neighbor { index: j, sim_bb, sim_ab, sqd_ab });
        }
    }

    let n = valid_neighbors.len();
    if n < 2 {
        return F::infinity();
    }

    let mut mean = F::zero();
    let mut m2 = F::zero();
    let mut total_weight = F::zero();

    for a in 0..n {
        for b in (a + 1)..n {
            let n_a = &valid_neighbors[a];
            let n_b = &valid_neighbors[b];
            let sim_ac = kernel(xi, data.point(n_b.index));
            let sim_bc = kernel(data.point(n_a.index), data.point(n_b.index));

            let numerator = sim_bc - n_a.sim_ab - sim_ac + sim_aa;
            let sqd_ac = sim_aa + n_b.sim_bb - sim_ac - sim_ac;

            if sqd_ac.is_nan() || sqd_ac <= F::zero() {
                continue;
            }

            let div = F::one() / (n_a.sqd_ab * sqd_ac);
            let value = numerator * div;
            let weight = div.sqrt();
            total_weight += weight;
            if total_weight.partial_cmp(&F::zero()) != Some(std::cmp::Ordering::Greater) {
                continue;
            }
            let delta = value - mean;
            let mult = weight / total_weight;
            mean += delta * mult;
            let delta2 = value - mean;
            m2 += weight * delta * delta2;
        }
    }

    if total_weight.partial_cmp(&F::zero()) != Some(std::cmp::Ordering::Greater) {
        return F::infinity();
    }

    let variance = m2 / total_weight;
    if variance <= F::zero() {
        return F::infinity();
    }

    variance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::outlier::common::*;

    #[test]
    fn abod_poly2_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);

        let kernel = crate::kernel::polynomial::PolynomialKernel::new(2, 1.0, 0.0);
        let result = angle_based_outlier_detection(&data, |x, y| kernel.similarity(x, y));

        let reference = load_reference_scores();
        let expected = reference.get("ABOD-poly2").expect("No reference for ABOD-poly2");
        let labels: Vec<u8> = label_from_reference(&reference);
        assert_outlier_auc_approx(
            "ABOD-poly2",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("ABOD-poly2", &result.scores, expected, 1e-6);
    }
}
