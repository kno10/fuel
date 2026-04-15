use crate::api::{PrioritySearcher, PrioritySearcherFactory};
use crate::cluster::hierarchical::common::UnionFind;
use crate::cluster::hierarchical::{Builder, GeometricLinkage, MergeHistory, idsize};
use crate::{CoordinateQuery, Float};

fn center<F: Float>(centers: &[Option<Vec<F>>], cid: usize) -> &[F] {
    centers[cid].as_ref().expect("cluster center must be initialized")
}

/// Find starting elements (i.e., a root) for the nearest-neighbor chain
fn find_root_excluding(uf: &mut UnionFind<idsize>, exclude: Option<idsize>) -> idsize {
    for i in 0..uf.parent.len() {
        let node = i as idsize;
        let root = uf.find(node);
        if root == node && exclude != Some(root) {
            return root;
        }
    }
    panic!("no active cluster found");
}

const WITNESS_BIT: idsize = 1 << 31;

struct ClusterFilter<'a> {
    uf: &'a mut UnionFind<idsize>,
    cluster_id: &'a [idsize],
    visited: &'a [bool],
    node_cluster: &'a mut [idsize],
}

impl<'a> crate::SearchFilter for ClusterFilter<'a> {
    fn skip_node(&mut self, points: crate::NodePoints<'_>) -> bool {
        let vp = points.first_index();
        let cached = self.node_cluster[vp];

        if cached != idsize::MAX {
            if cached & WITNESS_BIT != 0 {
                let witness = (cached & !WITNESS_BIT) as usize;
                if self.uf.find(vp as idsize) != self.uf.find(witness as idsize) {
                    return false;
                }
            } else {
                let root = self.uf.find(cached);
                let cid = self.cluster_id[root as usize];
                return self.visited[cid as usize];
            }
        }

        let mut component = idsize::MAX;
        for index in points.indices() {
            let root = self.uf.find(index as idsize);
            if component == idsize::MAX {
                component = root;
            } else if root != component {
                self.node_cluster[vp] = WITNESS_BIT | (index as idsize);
                return false;
            }
        }

        self.node_cluster[vp] = component;
        let cid = self.cluster_id[component as usize];
        self.visited[cid as usize]
    }

    fn skip_point(&mut self, index: usize) -> bool {
        let root = self.uf.find(index as idsize);
        let cid = self.cluster_id[root as usize];
        self.visited[cid as usize]
    }
}

fn grow_chain<'a, D, F, L, S>(
    data: &'a D, linkage: L, builder: &Builder<F>, centers: &[Option<Vec<F>>], heights: &[F],
    uf: &mut UnionFind<idsize>, cluster_id: &[idsize], chain: &mut Vec<idsize>,
    visited: &mut [bool], node_cluster: &mut [idsize], searcher: &mut S,
) -> bool
where
    D: crate::DistanceData<F> + crate::VectorData<F> + ?Sized + 'a,
    D::Query<'a>: CoordinateQuery<F, F> + 'a,
    F: Float,
    L: GeometricLinkage<F> + Copy,
    S: PrioritySearcher<F, D::Query<'a>>,
{
    let mut query = data.query();
    let mut a_root;
    let mut b_root;

    if chain.len() < 2 {
        a_root = find_root_excluding(uf, None);
        b_root = find_root_excluding(uf, Some(a_root));
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

    let mut a_cid = cluster_id[uf.find(a_root) as usize];
    let mut b_cid = cluster_id[uf.find(b_root) as usize];
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
        visited[a_cid as usize] = true;
        visited[b_cid as usize] = true;

        let size_a = builder.get_size(a_cid);
        let a_center = center(centers, a_cid as usize);
        let cutoff_factor = linkage.cutoff_factor(size_a);

        searcher.reset_with_limits(cutoff_factor * min_link, F::zero());
        query.set_coordinates(a_center);

        let mut c_root = b_root;
        let mut c_cid = b_cid;

        while let Some(candidate) = {
            let mut filter = ClusterFilter { uf, cluster_id, visited, node_cluster };
            searcher.next_with_filter(&query, &mut filter)
        } {
            let root = uf.find(candidate.index as idsize);
            let cid = cluster_id[root as usize];

            let size_i = builder.get_size(cid);
            let threshold = linkage.candidate_threshold(
                min_link,
                size_a,
                size_i,
                heights[a_cid as usize],
                heights[cid as usize],
            );
            visited[cid as usize] = true;
            if candidate.distance > threshold {
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
/// This variant uses incremental priority search to accelerate nearest
/// neighbour discovery for geometric linkage criteria.
#[must_use]
pub fn incremental_nn_chain<'a, S, D, F, L>(tree: &'a S, data: &'a D, linkage: L) -> MergeHistory<F>
where
    F: Float + 'a,
    D: crate::DistanceData<F> + crate::VectorData<F> + ?Sized + 'a,
    D::Query<'a>: CoordinateQuery<F, F> + 'a,
    S: PrioritySearcherFactory<F, D::Query<'a>>,
    L: GeometricLinkage<F> + Copy + 'static,
{
    let n = data.nrows();
    assert!(n > 0, "number of points must be positive");

    let dim = data.dims();
    assert!(
        (0..n).all(|i| data.point(i).len() == dim),
        "all vectors must have equal dimensionality"
    );

    let points: Vec<Vec<F>> = (0..n).map(|i| data.point(i).to_vec()).collect();
    let mut builder = Builder::<F>::new(n);
    let mut centers: Vec<Option<Vec<F>>> = vec![None; 2 * n - 1];
    for (i, v) in points.iter().enumerate() {
        centers[i] = Some(v.clone());
    }
    let mut heights = vec![F::zero(); 2 * n - 1];

    let squared = data.is_squared_distance();
    let mut uf = UnionFind::<idsize>::new(n);
    let mut cluster_id: Vec<idsize> = (0..(n as idsize)).collect();
    let mut chain: Vec<idsize> = Vec::with_capacity(n.min(64));
    let mut visited = vec![false; 2 * n - 1];
    let mut node_cluster = vec![idsize::MAX; n];
    let mut searcher = tree.priority_searcher();
    let mut merged = 0usize;

    while merged < n - 1 {
        if !grow_chain(
            data,
            linkage,
            &builder,
            &centers,
            &heights,
            &mut uf,
            &cluster_id,
            &mut chain,
            &mut visited,
            &mut node_cluster,
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

        let a_cid = cluster_id[uf.find(a_root) as usize];
        let b_cid = cluster_id[uf.find(b_root) as usize];
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
        let new_id = builder.add(h1, linkage.restore(min_link, squared), h2);

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

        let (_, new_root) = uf.union(a_root, b_root);
        cluster_id[new_root as usize] = new_id;

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
        MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage, WardLinkage,
    };
    use crate::distance::SquaredEuclidean;
    use crate::search::kdtree::{KdTree, MaxVarianceSplit};

    #[test]
    fn incremental_nn_chain_average_regression() {
        // Only SquaredEuclidean is geometric!
        test_clustering_table(
            "IncrementalNNChain",
            "average",
            SquaredEuclidean,
            |access, min_clusters| {
                let tree = KdTree::new(access, MaxVarianceSplit);
                let history = incremental_nn_chain(&tree, access, GroupAverageLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
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
                let tree = KdTree::new(access, MaxVarianceSplit);
                let history = incremental_nn_chain(&tree, access, CentroidLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
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
                let tree = KdTree::new(access, MaxVarianceSplit);
                let history = incremental_nn_chain(&tree, access, MedianLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
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
                let tree = KdTree::new(access, MaxVarianceSplit);
                let history = incremental_nn_chain(&tree, access, WardLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
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
                let tree = KdTree::new(access, MaxVarianceSplit);
                let history = incremental_nn_chain(&tree, access, MinimumSumSquaresLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
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
                let tree = KdTree::new(access, MaxVarianceSplit);
                let history = incremental_nn_chain(&tree, access, MinimumVarianceLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
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
                let tree = KdTree::new(access, MaxVarianceSplit);
                let history = incremental_nn_chain(&tree, access, MinimumVarianceIncreaseLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }
}
