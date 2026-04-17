/// Group-average linkage (UPGMA).
///
/// The cluster distance is defined as the average pairwise distance between
/// clusters:
/// $\frac{n_x d_x + n_y d_y}{n_x + n_y}$.
/// This method never produces inversions.
/// It supports stored-matrix algorithms, geometric stored-data approaches,
/// and set-based implementations.
#[derive(Clone, Copy, Default, Debug)]
pub struct GroupAverageLinkage;

use crate::cluster::hierarchical::{GeometricLinkage, Linkage, SetLinkage, idsize};
use crate::{DistanceData, Float};

impl<F: Float> Linkage<F> for GroupAverageLinkage {
    fn combine(
        &self, sizex: usize, dx: F, sizey: usize, dy: F, _sizej: usize, _dxy: F, _heightx: F,
        _heighty: F, _heightj: F,
    ) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        (sx * dx + sy * dy) / F::from(sizex + sizey).unwrap()
    }
}

impl<F: Float> GeometricLinkage<F> for GroupAverageLinkage {
    fn linkage(&self, x: &[F], _sizex: usize, y: &[F], _sizey: usize, heightx: F, heighty: F) -> F {
        let mut total = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let d = *xi - *yi;
            total += d * d;
        }
        total + heightx + heighty
    }

    fn merge_height(
        &self, x: &[F], sizex: usize, y: &[F], sizey: usize, heightx: F, heighty: F,
    ) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        let tot = sx + sy;
        let link = self.linkage(x, sizex, y, sizey, heightx, heighty);
        (sx * sx * heightx + sy * sy * heighty + sx * sy * link) / (tot * tot)
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, ()> for GroupAverageLinkage {
    fn summarize(_data: &D, _members: &[idsize]) {}

    fn cluster_distance(
        data: &D, _summary_a: &(), _summary_b: &(), a: &[idsize], b: &[idsize],
    ) -> (F, ()) {
        let mut sum = F::zero();
        let mut count = 0usize;
        for &i in a {
            for &j in b {
                sum += F::from(data.distance(i as usize, j as usize)).unwrap();
                count += 1;
            }
        }
        (sum / F::from(count).unwrap(), ())
    }
}
