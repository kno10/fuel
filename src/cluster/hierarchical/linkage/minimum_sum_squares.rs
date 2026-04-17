use crate::cluster::hierarchical::linkage::ward::{cross_squared_sum, pairwise_squared_sum};
use crate::cluster::hierarchical::{GeometricLinkage, Linkage, SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Minimal sum-of-squares linkage (MNSSQ).
///
/// This global objective is defined by the sum of squared deviations of the
/// merged cluster around its centroid.
/// The recurrence is equivalent to updating the within-cluster squared error
/// for the merged cluster.
/// This method is supported by stored-matrix algorithms, geometric stored-data
/// approaches, and set-based implementations.
#[derive(Clone, Copy, Default, Debug)]
pub struct MinimumSumSquaresLinkage;

impl<F: Float> Linkage<F> for MinimumSumSquaresLinkage {
    fn initial(&self, d: F, issquare: bool) -> F { F::half() * if issquare { d } else { d * d } }

    fn restore(&self, d: F, issquare: bool) -> F {
        if issquare { F::two() * d } else { (F::two() * d).sqrt() }
    }

    fn combine(
        &self, sizex: usize, dx: F, sizey: usize, dy: F, sizej: usize, dxy: F, heightx: F,
        heighty: F, heightj: F,
    ) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        let sj = F::from(sizej).unwrap();
        ((sx + sj) * dx + (sy + sj) * dy + (sx + sy) * dxy
            - sx * heightx
            - sy * heighty
            - sj * heightj)
            / (sx + sy + sj)
    }
}

impl<F: Float> GeometricLinkage<F> for MinimumSumSquaresLinkage {
    fn linkage(&self, x: &[F], sizex: usize, y: &[F], sizey: usize, heightx: F, heighty: F) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        let mut sum = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let diff = *xi - *yi;
            sum += diff * diff;
        }
        heightx + heighty + sx * sy / (sx + sy) * sum
    }

    fn cutoff_factor(&self, size_a: usize) -> F {
        let sa = F::from(size_a).unwrap();
        F::one() + F::one() / sa
    }

    fn candidate_threshold(
        &self, min_link: F, size_a: usize, size_i: usize, height_a: F, height_i: F,
    ) -> F {
        let sa = F::from(size_a).unwrap();
        let si = F::from(size_i).unwrap();
        let numerator = min_link - height_a - height_i;
        if numerator <= F::zero() { F::infinity() } else { numerator * (sa + si) / (sa * si) }
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, F> for MinimumSumSquaresLinkage {
    fn summarize(data: &D, members: &[idsize]) -> F {
        if members.len() <= 1 {
            return F::zero();
        }
        let squared = data.is_squared_distance();
        pairwise_squared_sum(data, members, squared)
    }

    fn cluster_distance(
        data: &D, summary_a: &F, summary_b: &F, a: &[idsize], b: &[idsize],
    ) -> (F, F) {
        let squared = data.is_squared_distance();
        let cross = cross_squared_sum(data, a, b, squared);
        let na = F::from(a.len()).unwrap();
        let nb = F::from(b.len()).unwrap();
        let ssq = (*summary_a * na + *summary_b * nb + cross) / (na + nb);
        (ssq, ssq) // this is NOT Ward.
    }

    fn restore(d: F, issquare: bool) -> F {
        if issquare { F::two() * d } else { (F::two() * d).sqrt() }
    }
}
