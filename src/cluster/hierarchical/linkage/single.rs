/// Single-linkage criterion (minimum distance).
#[derive(Clone, Copy, Default, Debug)]
pub struct SingleLinkage;

use super::Linkage;
use crate::cluster::hierarchical::SetLinkage;
use crate::{DistanceData, Float};

impl<F: Float> Linkage<F> for SingleLinkage {
    fn combine(&self, _sizex: usize, dx: F, _sizey: usize, dy: F, _sizej: usize, _dxy: F) -> F {
        dx.min(dy)
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, ()> for SingleLinkage {
    fn summarize(_data: &D, _members: &[usize]) {}

    fn cluster_distance(
        data: &D, _summary_a: &(), _summary_b: &(), a: &[usize], b: &[usize],
    ) -> (F, Option<usize>) {
        let mut best = F::infinity();
        for &i in a {
            for &j in b {
                let d = F::from(data.distance(i, j)).unwrap();
                if d < best {
                    best = d;
                }
            }
        }
        (best, None) // prototype not tracked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::agnes;

    #[test]
    fn single_combine_behaviour() {
        let l = SingleLinkage;
        assert_eq!(l.combine(1, 2.0, 1, 3.0, 1, 4.0), 2.0);
        assert_eq!(l.combine(1, 5.0, 1, 1.0, 1, 0.0), 1.0);
    }

    #[test]
    fn agnes_with_single_runs() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let history = agnes(&d, 4, SingleLinkage, false);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().size, 4);
    }

    #[test]
    fn single_linkage_f32_compile() {
        let l = SingleLinkage;
        let r: f32 = l.combine(0, 1.0_f32, 0, 2.0_f32, 0, 3.0_f32);
        assert_eq!(r, 1.0_f32);
    }
}
