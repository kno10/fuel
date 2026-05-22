#[cfg(test)]
mod tests {
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::{MedoidLinkage, set_agnes};
    use crate::cluster::hierarchical::test::{ScalarDistance, test_clustering_table};

    #[test]
    fn medoid_linkage_produces_valid_history() {
        use crate::TableWithDistance;
        let points = [vec![0.0], vec![1.0], vec![3.0], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, ScalarDistance);
        let h = set_agnes::<_, MedoidLinkage, f64, _>(&data).unwrap();
        assert_eq!(h.len(), 3);
        assert_eq!(h.last().expect("non-empty").size, 4);
        assert!(h.iter().all(|m| m.prototype < 4));
    }

    #[test]
    fn medoid_regression() {
        test_clustering_table(
            "MedoidLinkage",
            "medoid",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, MedoidLinkage, f64, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }
}
