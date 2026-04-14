use crate::cluster::hierarchical::{HausdorffLinkage, MergeHistory, set_agnes};
use crate::{DistanceData, Float};

/// Hierarchical clustering with Hausdorff linkage.
#[must_use]
pub fn hausdorff<D, F>(data: &D) -> MergeHistory<F>
where
    D: DistanceData<F>,
    F: Float,
{
    set_agnes::<D, HausdorffLinkage, F, _>(data)
}

#[cfg(test)]
mod tests {
    use super::hausdorff;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_table;

    #[test]
    fn hausdorff_regression() {
        test_clustering_table(
            "HAUSDORFF",
            "hausdorff",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = hausdorff(access);
                cut_dendrogram_by_number_of_clusters(&history, min_clusters)
            },
        );
    }
}
