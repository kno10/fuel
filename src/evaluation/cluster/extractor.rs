use num_traits::Float;

use crate::cluster::hierarchical::{
    Merge, cut_dendrogram_by_height, cut_dendrogram_by_number_of_clusters,
};

/// Port of ELKI's `CutDendrogramByHeightExtractor` evaluation helper.
#[must_use]
pub fn cut_dendrogram_by_height_extractor<F: Float>(
    history: &[Merge<F>],
    threshold: F,
) -> Vec<usize> {
    cut_dendrogram_by_height(history, threshold)
}

/// Port of ELKI's `CutDendrogramByNumberOfClustersExtractor` evaluation helper.
#[must_use]
pub fn cut_dendrogram_by_number_of_clusters_extractor<F: Float>(
    history: &[Merge<F>],
    min_clusters: usize,
) -> Vec<usize> {
    cut_dendrogram_by_number_of_clusters(history, min_clusters)
}

/// Apply a cut-by-height extraction to many merge histories.
#[must_use]
pub fn extract_all_by_height<F: Float>(
    histories: &[Vec<Merge<F>>],
    threshold: F,
) -> Vec<Vec<usize>> {
    histories
        .iter()
        .map(|h| cut_dendrogram_by_height(h, threshold))
        .collect()
}

/// Apply a cut-by-k extraction to many merge histories.
#[must_use]
pub fn extract_all_by_number_of_clusters<F: Float>(
    histories: &[Vec<Merge<F>>],
    min_clusters: usize,
) -> Vec<Vec<usize>> {
    histories
        .iter()
        .map(|h| cut_dendrogram_by_number_of_clusters(h, min_clusters))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::agnes;
    use crate::cluster::hierarchical::linkage::SingleLinkage;

    #[test]
    fn wrappers_match_core_extractors() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let h = agnes(&d, 4, SingleLinkage, false);

        assert_eq!(
            cut_dendrogram_by_height_extractor(&h, 1.1),
            cut_dendrogram_by_height(&h, 1.1)
        );
        assert_eq!(
            cut_dendrogram_by_number_of_clusters_extractor(&h, 2),
            cut_dendrogram_by_number_of_clusters(&h, 2)
        );
    }
}
