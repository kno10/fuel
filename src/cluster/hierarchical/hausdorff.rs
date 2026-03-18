use crate::{DistanceData, cluster::hierarchical::PrototypeMergeHistory};
use num_traits::Float;

use super::linkage::HausdorffLinkage;
use crate::cluster::hierarchical::set_linkage;

/// Hierarchical clustering with Hausdorff linkage.
#[must_use]
pub fn hausdorff<D, F>(data: &D) -> PrototypeMergeHistory<F>
where
    D: DistanceData<F>,
    F: Float,
{
    set_linkage::<D, HausdorffLinkage, F, _>(data)
}

#[cfg(test)]
mod tests {
    use super::hausdorff;
    use crate::TableWithDistance;
    use crate::cluster::hierarchical::test_utils::ScalarDistance;

    #[test]
    fn hausdorff_produces_valid_history() {
        let data = TableWithDistance::with_distance(&[0.0, 1.0, 3.0, 10.0], ScalarDistance);
        let h = hausdorff(&data);
        assert_eq!(h.len(), 3);
        assert_eq!(h.last().expect("non-empty").size, 4);
        assert!(h.iter().all(|m| m.prototype < Some(4)));
    }
}
