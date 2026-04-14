use crate::cluster::hierarchical::{Builder, GeometricLinkage, MergeHistory, idsize};
use crate::distance::SquaredEuclidean;
use crate::search::kdtree::{KdTree, KdTreePrioritySearcher, MaxVarianceSplit};
use crate::{CoordinateQuery, DistanceData, Float, PrioritySearcher, TableWithDistance};

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    cluster_id: Vec<idsize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
            cluster_id: (0..(n as idsize)).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut node = x;
        while self.parent[node] != root {
            let next = self.parent[node];
            self.parent[node] = root;
            node = next;
        }
        root
    }

    fn cluster_of(&mut self, x: usize) -> idsize {
        let root = self.find(x);
        self.cluster_id[root]
    }

    fn union(&mut self, a: usize, b: usize, new_id: idsize) -> usize {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return ra;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
        self.cluster_id[ra] = new_id;
        ra
    }

    fn find_root_excluding(&mut self, exclude: Option<usize>) -> usize {
        for i in 0..self.parent.len() {
            let root = self.find(i);
            if root == i && exclude != Some(root) {
                return root;
            }
        }
        panic!("no active cluster found");
    }
}

fn center<F: Float>(centers: &[Option<Vec<F>>], cid: usize) -> &[F] {
    centers[cid].as_ref().expect("cluster center must be initialized")
}

fn grow_chain<'a, F: Float, L: GeometricLinkage<F> + Copy>(
    data: &'a TableWithDistance<'a, F, Vec<F>, SquaredEuclidean, F>, linkage: L,
    builder: &Builder<F>, centers: &[Option<Vec<F>>], heights: &[F], uf: &mut UnionFind,
    chain: &mut Vec<usize>, visited: &mut [bool], searcher: &mut KdTreePrioritySearcher<'a, F, F>,
) -> bool {
    let mut query = data.query();
    let mut a_root;
    let mut b_root;

    if chain.len() < 2 {
        a_root = uf.find_root_excluding(None);
        b_root = uf.find_root_excluding(Some(a_root));
        chain.clear();
        chain.push(a_root);
    } else {
        a_root = uf.find(chain[chain.len() - 2]);
        b_root = uf.find(chain[chain.len() - 1]);
        if a_root == b_root {
            chain.truncate(chain.len().saturating_sub(2));
            return false;
        }
        chain.pop();
    }

    let mut a_cid = uf.cluster_of(a_root);
    let mut b_cid = uf.cluster_of(b_root);
    let mut min_link = {
        let size_a = builder.get_size(a_cid);
        let size_b = builder.get_size(b_cid);
        linkage.linkage(
            center(centers, a_cid as usize),
            size_a,
            center(centers, b_cid as usize),
            size_b,
            heights[a_cid as usize],
            heights[b_cid as usize],
        )
    };

    loop {
        visited.fill(false);
        let size_a = builder.get_size(a_cid);
        let a_center = center(centers, a_cid as usize);
        let cutoff_factor = linkage.cutoff_factor(size_a);

        searcher.reset_with_limits(cutoff_factor * min_link, F::zero());
        query.set_coordinates(a_center);

        let mut c_root = b_root;
        let mut c_cid = b_cid;

        while let Some(candidate) = searcher.next(&query) {
            let root = uf.find(candidate.index);
            let cid = uf.cluster_id[root];

            if cid == a_cid || cid == b_cid {
                let cid_index = cid as usize;
                if cid_index < visited.len() {
                    visited[cid_index] = true;
                }
                continue;
            }

            if visited.get(cid as usize).copied().unwrap_or(false) {
                continue;
            }

            let size_i = builder.get_size(cid);
            let threshold = linkage.candidate_threshold(
                min_link,
                size_a,
                size_i,
                heights[a_cid as usize],
                heights[cid as usize],
            );
            if candidate.distance > threshold {
                visited[cid as usize] = true;
                continue;
            }

            let link = linkage.linkage(
                a_center,
                size_a,
                center(centers, cid as usize),
                size_i,
                heights[a_cid as usize],
                heights[cid as usize],
            );
            if link < min_link {
                min_link = link;
                c_root = root;
                c_cid = cid;
                searcher.decrease_cutoff(cutoff_factor * link);
            }

            visited[cid as usize] = true;
        }

        b_root = a_root;
        b_cid = a_cid;
        a_root = c_root;
        a_cid = c_cid;
        chain.push(a_root);

        if chain.len() >= 3 && uf.find(chain[chain.len() - 3]) == a_root {
            break;
        }
    }

    true
}

/// Incremental nearest-neighbor chain clustering for vector data.
///
/// This variant uses incremental VP-tree search to accelerate nearest
/// neighbour discovery for geometric linkage criteria.
#[must_use]
pub fn incremental_nn_chain<F: Float, L: GeometricLinkage<F> + Copy + 'static, D>(
    data: &D, linkage: L,
) -> MergeHistory<F>
where
    D: crate::VectorData<F>,
{
    let n = data.nrows();
    assert!(n > 0, "number of points must be positive");

    let dim = data.dims();
    assert!(
        (0..n).all(|i| data.point(i).len() == dim),
        "all vectors must have equal dimensionality"
    );

    let points: Vec<Vec<F>> = (0..n).map(|i| data.point(i).to_vec()).collect();
    let table_data: TableWithDistance<'_, F, Vec<F>, SquaredEuclidean, F> =
        TableWithDistance::with_distance(&points, SquaredEuclidean);
    let tree = KdTree::new(&table_data, MaxVarianceSplit);

    let mut builder = Builder::<F>::new(n);
    let mut centers: Vec<Option<Vec<F>>> = vec![None; 2 * n - 1];
    for (i, v) in points.iter().enumerate() {
        centers[i] = Some(v.clone());
    }
    let mut heights = vec![F::zero(); 2 * n - 1];

    let squared = table_data.is_squared_distance();
    let mut uf = UnionFind::new(n);
    let mut chain: Vec<usize> = Vec::with_capacity(n.min(64));
    let mut visited = vec![false; 2 * n - 1];
    let mut searcher = KdTreePrioritySearcher::new(&tree);
    let mut merged = 0usize;

    while merged < n - 1 {
        if !grow_chain(
            &table_data,
            linkage,
            &builder,
            &centers,
            &heights,
            &mut uf,
            &mut chain,
            &mut visited,
            &mut searcher,
        ) {
            continue;
        }

        let a_root = uf.find(chain[chain.len() - 2]);
        let b_root = uf.find(chain[chain.len() - 1]);
        if a_root == b_root {
            chain.truncate(chain.len().saturating_sub(2));
            continue;
        }

        let a_cid = uf.cluster_of(a_root);
        let b_cid = uf.cluster_of(b_root);
        let size_a = builder.get_size(a_cid);
        let size_b = builder.get_size(b_cid);
        let a_center = center(&centers, a_cid as usize);
        let b_center = center(&centers, b_cid as usize);
        let min_link = linkage.linkage(
            a_center,
            size_a,
            b_center,
            size_b,
            heights[a_cid as usize],
            heights[b_cid as usize],
        );

        let (h1, h2) = if a_cid <= b_cid { (a_cid, b_cid) } else { (b_cid, a_cid) };
        let new_id = builder.add(h1, linkage.restore_linkage(min_link, squared), h2);

        let merged_center = linkage.merge(
            a_center,
            size_a,
            b_center,
            size_b,
            heights[a_cid as usize],
            heights[b_cid as usize],
        );
        let merged_height = linkage.merge_height(
            a_center,
            size_a,
            b_center,
            size_b,
            heights[a_cid as usize],
            heights[b_cid as usize],
        );

        centers[new_id as usize] = Some(merged_center);
        heights[new_id as usize] = merged_height;

        let new_root = uf.union(a_root, b_root, new_id);

        if chain.len() >= 3 {
            chain.truncate(chain.len() - 3);
        } else {
            chain.clear();
        }
        chain.push(new_root);

        merged += 1;
    }

    builder.optimize_order_in_place();
    builder.into_merges()
}

#[cfg(test)]
mod tests {
    use super::incremental_nn_chain;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_table;
    use crate::cluster::hierarchical::{
        CentroidLinkage, GroupAverageLinkage, MedianLinkage, MinimumSumSquaresLinkage,
        MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage, WardLinkage, nn_chain,
    };
    use crate::distance::SquaredEuclidean;
    use crate::{CondensedDistanceMatrix, TableWithDistance};

    #[test]
    fn incremental_nn_chain_average_regression() {
        // Only SquaredEuclidean is geometric!
        test_clustering_table(
            "IncrementalNNChain",
            "average",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = incremental_nn_chain(access, GroupAverageLinkage);
                cut_dendrogram_by_number_of_clusters(&history, min_clusters)
            },
        );
    }

    #[test]
    fn incremental_nn_chain_centroid_regression() {
        test_clustering_table(
            "IncrementalNNChain",
            "centroid",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = incremental_nn_chain(access, CentroidLinkage);
                cut_dendrogram_by_number_of_clusters(&history, min_clusters)
            },
        );
    }

    #[test]
    fn incremental_nn_chain_median_regression() {
        test_clustering_table(
            "IncrementalNNChain",
            "median",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = incremental_nn_chain(access, MedianLinkage);
                cut_dendrogram_by_number_of_clusters(&history, min_clusters)
            },
        );
    }

    #[test]
    fn incremental_nn_chain_ward_regression() {
        test_clustering_table(
            "IncrementalNNChain",
            "ward",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = incremental_nn_chain(access, WardLinkage);
                cut_dendrogram_by_number_of_clusters(&history, min_clusters)
            },
        );
    }

    #[test]
    fn incremental_nn_chain_minimum_sum_squares_regression() {
        test_clustering_table(
            "IncrementalNNChain",
            "mnssq",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = incremental_nn_chain(access, MinimumSumSquaresLinkage);
                cut_dendrogram_by_number_of_clusters(&history, min_clusters)
            },
        );
    }

    #[test]
    fn incremental_nn_chain_minimum_variance_regression() {
        test_clustering_table(
            "IncrementalNNChain",
            "mnvar",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = incremental_nn_chain(access, MinimumVarianceLinkage);
                cut_dendrogram_by_number_of_clusters(&history, min_clusters)
            },
        );
    }

    #[test]
    fn incremental_nn_chain_minimum_variance_increase_regression() {
        test_clustering_table(
            "IncrementalNNChain",
            "mivar",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = incremental_nn_chain(access, MinimumVarianceIncreaseLinkage);
                cut_dendrogram_by_number_of_clusters(&history, min_clusters)
            },
        );
    }
}
