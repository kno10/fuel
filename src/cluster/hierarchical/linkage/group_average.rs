/// Group-average linkage (UPGMA).
#[derive(Clone, Copy, Default, Debug)]
pub struct GroupAverageLinkage;

use super::{GeometricLinkage, Linkage};
use crate::cluster::hierarchical::SetLinkage;
use crate::{DistanceData, Float};

impl<F: Float> Linkage<F> for GroupAverageLinkage {
    fn combine(&self, sizex: usize, dx: F, sizey: usize, dy: F, _sizej: usize, _dxy: F) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        (sx * dx + sy * dy) / F::from(sizex + sizey).unwrap()
    }
}

impl<F: Float> GeometricLinkage<F> for GroupAverageLinkage {
    fn merge(&self, x: &[F], sizex: usize, y: &[F], sizey: usize) -> Vec<F> {
        let tot = F::from(sizex + sizey).unwrap();
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        x.iter().zip(y.iter()).map(|(&xi, &yi)| (sx * xi + sy * yi) / tot).collect()
    }

    fn linkage(&self, x: &[F], _sizex: usize, y: &[F], _sizey: usize) -> F {
        let mut total = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let d = *xi - *yi;
            total = total + d * d;
        }
        total
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, ()> for GroupAverageLinkage {
    fn summarize(_data: &D, _members: &[usize]) {}

    fn cluster_distance(
        data: &D, _summary_a: &(), _summary_b: &(), a: &[usize], b: &[usize],
    ) -> (F, Option<usize>) {
        let mut sum = F::zero();
        let mut count = 0usize;
        for &i in a {
            for &j in b {
                sum = sum + F::from(data.distance(i, j)).unwrap();
                count += 1;
            }
        }
        (sum / F::from(count).unwrap(), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::agnes;

    #[test]
    fn group_average_basic() {
        let g = GroupAverageLinkage;
        assert_eq!(g.combine(1, 1.0, 1, 3.0, 0, 0.0), 2.0);
    }

    #[test]
    fn agnes_with_group_average_runs() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let history = agnes(&d, 4, GroupAverageLinkage, false);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().size, 4);
    }

    #[test]
    fn group_average_f32_compile() {
        let g = GroupAverageLinkage;
        let r: f32 = g.combine(1, 1.0_f32, 1, 3.0_f32, 0, 0.0_f32);
        assert_eq!(r, 2.0_f32);
    }
}
