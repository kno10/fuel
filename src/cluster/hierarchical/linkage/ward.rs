use crate::cluster::hierarchical::{GeometricLinkage, Linkage, SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Ward linkage (minimize increase of sum-of-squares).
///
/// This method is special in that it operates naturally on *squared*
/// distances; therefore both `initial` and `restore` are overridden.  It also
/// implements `GeometricLinkage` to provide centroid merging support.
#[derive(Clone, Copy, Default, Debug)]
pub struct WardLinkage;

impl<F: Float> Linkage<F> for WardLinkage {
    fn initial(&self, d: F, issquare: bool) -> F { F::half() * if issquare { d } else { d * d } }

    fn restore(&self, d: F, issquare: bool) -> F {
        if issquare { F::two() * d } else { (F::two() * d).sqrt() }
    }

    fn combine(
        &self, sizex: usize, dx: F, sizey: usize, dy: F, sizej: usize, dxy: F, _heightx: F,
        _heighty: F, _heightj: F,
    ) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        let sj = F::from(sizej).unwrap();
        ((sx + sj) * dx + (sy + sj) * dy - sj * dxy) / (sx + sy + sj)
    }
}

impl<F: Float> GeometricLinkage<F> for WardLinkage {
    fn linkage(&self, x: &[F], sizex: usize, y: &[F], sizey: usize, _heightx: F, _heighty: F) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        let mut sum = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let diff = *xi - *yi;
            sum += diff * diff;
        }
        sx * sy / (sx + sy) * sum
    }

    fn cutoff_factor(&self, size_a: usize) -> F {
        let sa = F::from(size_a).unwrap();
        F::four() * (F::one() + F::one() / sa)
    }

    fn candidate_threshold(
        &self, min_link: F, size_a: usize, size_i: usize, _height_a: F, height_i: F,
    ) -> F {
        let sa = F::from(size_a).unwrap();
        let si = F::from(size_i).unwrap();
        let d_ai = (sa + si) / (sa * si) * min_link;
        F::two() * (d_ai + height_i + F::two() * (d_ai * height_i).sqrt())
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, F> for WardLinkage {
    fn summarize(data: &D, members: &[idsize]) -> F {
        if members.len() <= 1 {
            return F::zero();
        }
        let squared = data.is_squared_distance();
        pairwise_squared_sum(data, members, squared) / F::from(members.len()).unwrap()
    }

    fn cluster_distance(
        data: &D, summary_a: &F, summary_b: &F, a: &[idsize], b: &[idsize],
    ) -> (F, F) {
        let squared = data.is_squared_distance();
        let cross = cross_squared_sum(data, a, b, squared);
        let na = F::from(a.len()).unwrap();
        let nb = F::from(b.len()).unwrap();
        let ssq = (*summary_a * na + *summary_b * nb + cross) / (na + nb);
        let d = ssq - *summary_a - *summary_b;
        (d, ssq)
    }

    fn restore(d: F, issquare: bool) -> F {
        if issquare { F::two() * d } else { (F::two() * d).sqrt() }
    }
}

pub(crate) fn pairwise_squared_sum<D, F>(data: &D, members: &[idsize], squared: bool) -> F
where
    D: DistanceData<F>,
    F: Float,
{
    let mut sum = F::zero();
    for (i, &a) in members.iter().enumerate() {
        for &b in &members[..i] {
            let d = data.distance(a as usize, b as usize);
            sum += if squared { d } else { d * d };
        }
    }
    sum
}

pub(crate) fn cross_squared_sum<D, F>(data: &D, a: &[idsize], b: &[idsize], squared: bool) -> F
where
    D: DistanceData<F>,
    F: Float,
{
    let mut sum = F::zero();
    for &i in a {
        for &j in b {
            let d = data.distance(i as usize, j as usize);
            sum += if squared { d } else { d * d };
        }
    }
    sum
}
