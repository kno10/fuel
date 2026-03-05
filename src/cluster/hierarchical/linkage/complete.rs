/// Complete-linkage criterion (maximum distance).
#[derive(Clone, Copy, Default, Debug)]
pub struct CompleteLinkage;

use num_traits::Float;
use super::Linkage;

impl<F: Float> Linkage<F> for CompleteLinkage {
    fn combine(
        &self,
        _sizex: usize,
        dx: F,
        _sizey: usize,
        dy: F,
        _sizej: usize,
        _dxy: F,
    ) -> F {
        dx.max(dy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::agnes;

    #[test]
    fn complete_combines_maximum() {
        let l = CompleteLinkage;
        assert_eq!(l.combine(0, 1.0, 0, 2.0, 0, 0.0), 2.0);
    }

    #[test]
    fn agnes_with_complete_runs() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let history = agnes(&d, 4, CompleteLinkage, false);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().size, 4);
    }

    #[test]
    fn complete_linkage_f32_compile() {
        let l = CompleteLinkage;
        let r: f32 = l.combine(0, 1.0_f32, 0, 2.0_f32, 0, 0.0_f32);
        assert_eq!(r, 2.0_f32);
    }
}
