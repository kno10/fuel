use crate::cluster::hierarchical::linkage::ward::{cross_squared_sum, pairwise_squared_sum};
use crate::cluster::hierarchical::{GeometricLinkage, Linkage, SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Minimum variance increase linkage (MIVAR).
///
/// This variant applies a corrected clustering objective and can produce
/// inversions.
/// It supports stored-matrix algorithms as well as geometric and set-based
/// approaches.
#[derive(Clone, Copy, Default, Debug)]
pub struct MinimumVarianceIncreaseLinkage;

impl<F: Float> Linkage<F> for MinimumVarianceIncreaseLinkage {
    fn can_produce_inversions(&self) -> bool { true }

    fn initial(&self, d: F, issquare: bool) -> F { F::quarter() * if issquare { d } else { d * d } }

    fn restore(&self, d: F, issquare: bool) -> F {
        if issquare { F::four() * d } else { (F::four() * d).sqrt() }
    }

    fn combine(
        &self, sizex: usize, dx: F, sizey: usize, dy: F, sizej: usize, dxy: F, _heightx: F,
        _heighty: F, _heightj: F,
    ) -> F {
        let xj = F::from(sizex + sizej).unwrap();
        let yj = F::from(sizey + sizej).unwrap();
        let n = F::from(sizex + sizey + sizej).unwrap();
        let cn = F::from(sizej).unwrap() * F::from(sizex + sizey).unwrap();
        (xj * xj * dx + yj * yj * dy - cn * dxy) / (n * n)
    }
}

impl<F: Float> GeometricLinkage<F> for MinimumVarianceIncreaseLinkage {
    fn linkage(&self, x: &[F], sizex: usize, y: &[F], sizey: usize, heightx: F, heighty: F) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        let tot = sx + sy;
        let mut sum = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let diff = *xi - *yi;
            sum += diff * diff;
        }
        (sx * sy * sum + sx * (sy - F::one()) * heightx + sy * (sx - F::one()) * heighty)
            / (tot * tot)
    }

    fn cutoff_factor(&self, size_a: usize) -> F {
        let sa = F::from(size_a).unwrap();
        let sa1 = F::from(size_a + 1).unwrap();
        F::four() * sa1 * sa1 / sa
    }

    fn candidate_threshold(
        &self, min_link: F, size_a: usize, size_i: usize, height_a: F, height_i: F,
    ) -> F {
        let sa = F::from(size_a).unwrap();
        let si = F::from(size_i).unwrap();
        let tot = sa + si;
        let numerator = min_link * tot * tot
            - sa * (si - F::one()) * height_a
            - si * (sa - F::one()) * height_i;
        if numerator <= F::zero() { F::infinity() } else { numerator / (sa * si) }
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, F> for MinimumVarianceIncreaseLinkage {
    fn can_produce_inversions(&self) -> bool { true }

    fn summarize(data: &D, members: &[idsize]) -> F {
        if members.len() <= 1 {
            return F::zero();
        }
        let n = F::from(members.len()).unwrap();
        pairwise_squared_sum(data, members, data.is_squared_distance()) / n
    }

    fn cluster_distance(
        data: &D, summary_a: &F, summary_b: &F, a: &[idsize], b: &[idsize],
    ) -> (F, F) {
        let na = F::from(a.len()).unwrap();
        let nb = F::from(b.len()).unwrap();
        let n = na + nb;
        let squared = data.is_squared_distance();
        let cross = cross_squared_sum(data, a, b, squared);
        let summary = (*summary_a * na + *summary_b * nb + cross) / n;
        let d = (summary - *summary_a - *summary_b) / n;
        (d, summary)
    }

    fn restore(d: F, issquare: bool) -> F {
        if issquare { F::four() * d } else { (F::four() * d).sqrt() }
    }
}
