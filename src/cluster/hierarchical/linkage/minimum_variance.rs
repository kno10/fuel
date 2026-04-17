use crate::cluster::hierarchical::linkage::ward::{cross_squared_sum, pairwise_squared_sum};
use crate::cluster::hierarchical::{GeometricLinkage, Linkage, SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Minimum variance linkage (MNVAR).
///
/// This global objective is defined by the average squared deviation of the
/// merged cluster around its centroid.
/// The resulting recurrence is a variance-based Lance-Williams update.
/// This method never produces inversions and supports stored-matrix,
/// geometric, and set-based implementations.
#[derive(Clone, Copy, Default, Debug)]
pub struct MinimumVarianceLinkage;

impl<F: Float> Linkage<F> for MinimumVarianceLinkage {
    fn initial(&self, d: F, issquare: bool) -> F { F::quarter() * if issquare { d } else { d * d } }

    fn restore(&self, d: F, issquare: bool) -> F {
        if issquare { F::four() * d } else { (F::four() * d).sqrt() }
    }

    fn combine(
        &self, sizex: usize, dx: F, sizey: usize, dy: F, sizej: usize, dxy: F, heightx: F,
        heighty: F, heightj: F,
    ) -> F {
        let nx = F::from(sizex).unwrap();
        let ny = F::from(sizey).unwrap();
        let nk = F::from(sizej).unwrap();
        let n = nx + ny + nk;
        let (alpha_x, alpha_y, beta) = (nk + nx, nk + ny, nx + ny);
        (alpha_x * alpha_x * dx + alpha_y * alpha_y * dy + beta * beta * dxy
            - nx * nx * heightx
            - ny * ny * heighty
            - nk * nk * heightj)
            / (n * n)
    }
}

impl<F: Float> GeometricLinkage<F> for MinimumVarianceLinkage {
    fn linkage(&self, x: &[F], sizex: usize, y: &[F], sizey: usize, heightx: F, heighty: F) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        let tot = sx + sy;
        let mut sum = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let diff = *xi - *yi;
            sum += diff * diff;
        }
        (heightx * sx + heighty * sy + sx * sy / tot * sum) / tot
    }

    fn cutoff_factor(&self, size_a: usize) -> F {
        let sa = F::from(size_a).unwrap();
        let sa1 = F::from(size_a + 1).unwrap();
        sa1 * sa1 / sa
    }

    fn candidate_threshold(
        &self, min_link: F, size_a: usize, size_i: usize, height_a: F, height_i: F,
    ) -> F {
        let sa = F::from(size_a).unwrap();
        let si = F::from(size_i).unwrap();
        let numerator = min_link * (sa + si) - height_a * sa - height_i * si;
        if numerator <= F::zero() { F::infinity() } else { numerator * (sa + si) / (sa * si) }
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, F> for MinimumVarianceLinkage {
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
        (summary / n, summary)
    }

    fn restore(d: F, issquare: bool) -> F {
        if issquare { F::four() * d } else { (F::four() * d).sqrt() }
    }
}
