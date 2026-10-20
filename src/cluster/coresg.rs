//! CORE-SG (core-distance-ordered sparse graph) construction.
//!
//! The graph is the union of the `max_points`-nearest-neighbour graph and the
//! edges of a mutual-reachability MST that are not equal to an endpoint's
//! core distance. This follows the pruning rule in the original CORE-SG code.

use std::cmp::Ordering;

#[cfg(not(feature = "parallel"))]
use crate::cluster::hdbscan::hdbscan_common::compute_core_distances_tree;
use crate::cluster::hdbscan::hdbscan_common::{
    mutual_reachability_distance, HdbscanHierarchy, WeightedEdge,
};
use crate::cluster::hierarchical::{idsize, Builder};
use crate::{DistanceData, Float, IndexQuery, KnnSearch};

/// An undirected weighted edge in the CORE-SG graph (`a < b`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoreSgEdge<F: Float> {
    pub a: idsize,
    pub b: idsize,
    pub weight: F,
}

/// CORE-SG graph and the core distances used to construct it.
#[derive(Debug, Clone, PartialEq)]
pub struct CoreSgGraph<F: Float> {
    pub edges: Vec<CoreSgEdge<F>>,
    pub core_distances: Vec<F>,
}

/// Build CORE-SG using a kNN index. `max_points` is the largest supported
/// `min_points` setting and counts the query point itself, matching `fuel`'s
/// HDBSCAN core-distance convention and the Python code.
pub fn coresg<'a, S, D, F>(
    tree: &S, data: &'a D, max_points: usize,
) -> Result<CoreSgGraph<F>, String>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    assert!(max_points > 0, "max_points must be greater than 0");
    let n = data.len();
    assert!(n > 0, "number of points must be positive");
    if n > idsize::MAX as usize {
        return Err("dataset exceeds supported idsize range".into());
    }
    #[cfg(feature = "parallel")]
    let core_distances: Vec<F> = {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .map_init(
                || data.query(),
                |query, i| {
                    crate::poll_interrupted()?;
                    query.set_index(i);
                    let neighbors = tree.search_knn(query, max_points);
                    Ok(if neighbors.len() >= max_points {
                        neighbors[max_points - 1].distance
                    } else {
                        F::infinity()
                    })
                },
            )
            .collect::<Result<Vec<_>, String>>()?
    };
    #[cfg(not(feature = "parallel"))]
    let core_distances = compute_core_distances_tree(tree, data, max_points)?;

    // Keep all ties at the requested neighbour distance, as the Python
    // implementation augments its kNN graph with the tied neighbours.
    let neighbors_by_point: Vec<Vec<(idsize, F)>> = {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map_init(
                    || data.query(),
                    |query, i| {
                        query.set_index(i);
                        tree.search_knn(query, max_points)
                            .into_iter()
                            .filter(|p| p.index != i && p.index < n)
                            .map(|p| (p.index as idsize, p.distance))
                            .collect()
                    },
                )
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut query = data.query();
            (0..n)
                .map(|i| {
                    query.set_index(i);
                    tree.search_knn(&query, max_points)
                        .into_iter()
                        .filter(|p| p.index != i && p.index < n)
                        .map(|p| (p.index as idsize, p.distance))
                        .collect()
                })
                .collect()
        }
    };
    let mut graph = std::collections::BTreeMap::<(idsize, idsize), F>::new();
    for (i, neighbors) in neighbors_by_point.into_iter().enumerate() {
        crate::poll_interrupted()?;
        for (neighbor, distance) in neighbors {
            let i = i as idsize;
            let key = (i.min(neighbor), i.max(neighbor));
            graph.entry(key).and_modify(|w| *w = w.min(distance)).or_insert(distance);
        }
    }

    // Prim over mutual-reachability distances. CORE-SG's reference computes
    // this global MST and then unions it with the kNN graph.
    if n > 1 {
        let mut in_tree = vec![false; n];
        let mut best = vec![F::infinity(); n];
        let mut parent = vec![idsize::MAX; n];
        in_tree[0] = true;
        for b in 1..n {
            best[b] = mutual_reachability_distance(&core_distances, 0, b, data.distance(0, b));
            parent[b] = 0;
        }
        let mut edges = Vec::<WeightedEdge<F>>::with_capacity(n - 1);
        while edges.len() < n - 1 {
            crate::poll_interrupted()?;
            let mut next = usize::MAX;
            for b in 0..n {
                if !in_tree[b] && (next == usize::MAX || best[b] < best[next]) {
                    next = b;
                }
            }
            assert_ne!(next, usize::MAX, "complete distance graph must be connected");
            in_tree[next] = true;
            edges.push(WeightedEdge::new(parent[next] as usize, next, best[next]));
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                best.par_iter_mut().zip(parent.par_iter_mut()).enumerate().for_each(
                    |(b, (dist, from))| {
                        if !in_tree[b] {
                            let candidate = mutual_reachability_distance(
                                &core_distances,
                                next,
                                b,
                                data.distance(next, b),
                            );
                            if candidate < *dist || (candidate == *dist && (next as idsize) < *from)
                            {
                                *dist = candidate;
                                *from = next as idsize;
                            }
                        }
                    },
                );
            }
            #[cfg(not(feature = "parallel"))]
            for b in 0..n {
                if !in_tree[b] {
                    let candidate = mutual_reachability_distance(
                        &core_distances,
                        next,
                        b,
                        data.distance(next, b),
                    );
                    if candidate < best[b] || (candidate == best[b] && (next as idsize) < parent[b])
                    {
                        best[b] = candidate;
                        parent[b] = next as idsize;
                    }
                }
            }
        }
        for edge in edges {
            // This is the original CORE-SG pruning predicate.
            if edge.weight == core_distances[edge.a] || edge.weight == core_distances[edge.b] {
                continue;
            }
            let key = ((edge.a.min(edge.b)) as idsize, (edge.a.max(edge.b)) as idsize);
            // Store the underlying distance so this max-k graph can be
            // reweighted with core distances for any supported k <= max-k.
            let distance = data.distance(edge.a, edge.b);
            graph.entry(key).and_modify(|w| *w = w.min(distance)).or_insert(distance);
        }
    }

    let edges = graph.into_iter().map(|((a, b), weight)| CoreSgEdge { a, b, weight }).collect();
    Ok(CoreSgGraph { edges, core_distances })
}

impl<F: Float> CoreSgGraph<F> {
    /// Rebuild the HDBSCAN mutual-reachability MST from this sparse graph.
    /// This lets the usual HDBSCAN flat, simplified, and GLOSH extraction
    /// functions operate on CORE-SG output.
    pub fn hdbscan_hierarchy(&self) -> Result<HdbscanHierarchy<F>, String> {
        self.hdbscan_hierarchy_for_core_distances(&self.core_distances)
    }

    /// Build an HDBSCAN hierarchy for another core-distance vector using this
    /// CORE-SG graph. A graph built with `max_points` neighbors can be reused
    /// for supported settings up to that maximum.
    pub fn hdbscan_hierarchy_for_core_distances(
        &self, core_distances: &[F],
    ) -> Result<HdbscanHierarchy<F>, String> {
        let n = self.core_distances.len();
        assert!(n > 0, "number of points must be positive");
        assert_eq!(core_distances.len(), n, "core distance length must match graph");
        if n > idsize::MAX as usize {
            return Err("dataset exceeds supported idsize range".into());
        }
        let mut edges: Vec<_> = self
            .edges
            .iter()
            .map(|edge| {
                WeightedEdge::new(
                    edge.a as usize,
                    edge.b as usize,
                    core_distances[edge.a as usize]
                        .max(core_distances[edge.b as usize])
                        .max(edge.weight),
                )
            })
            .collect();
        edges.sort_by(|a, b| {
            a.weight
                .partial_cmp(&b.weight)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.a.cmp(&b.a))
                .then_with(|| a.b.cmp(&b.b))
        });

        let mut parent: Vec<idsize> = (0..n).map(|i| i as idsize).collect();
        let mut cluster_ids: Vec<idsize> = (0..n).map(|i| i as idsize).collect();
        let mut builder = Builder::new(n);
        for edge in edges {
            let mut a = edge.a;
            while parent[a] as usize != a {
                a = parent[a] as usize;
            }
            let mut b = edge.b;
            while parent[b] as usize != b {
                b = parent[b] as usize;
            }
            if a == b {
                continue;
            }
            let (root, child) = if a < b { (a, b) } else { (b, a) };
            parent[child] = root as idsize;
            let merged = builder.add(cluster_ids[root], edge.weight, cluster_ids[child]);
            cluster_ids[root] = merged;
        }
        debug_assert!(builder.check_monotone());
        let history = builder.into_merges();
        if history.len() != n.saturating_sub(1) {
            return Err("CORE-SG graph is disconnected".into());
        }
        Ok(HdbscanHierarchy::new(history, core_distances.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::coresg;
    use crate::cluster::hdbscan::extraction::extract_clusters_with_noise;
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;
    use crate::TableWithDistance;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn coresg_graph_produces_hdbscan_hierarchy() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(31);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let graph = coresg(&tree, &data, 2).unwrap();
        assert!(graph.edges.iter().all(|edge| edge.a < edge.b));
        let hierarchy = graph.hdbscan_hierarchy().unwrap();
        assert_eq!(hierarchy.merges.len(), points.len() - 1);
        let labels = extract_clusters_with_noise(&hierarchy.merges, 2, 2);
        assert_ne!(labels[0], labels[3]);
    }
}
