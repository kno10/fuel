//! CORE-SG (core-distance-ordered sparse graph) construction.
//!
//! The graph is the union of the `max_points`-nearest-neighbour graph and the
//! edges of a mutual-reachability MST that are not equal to an endpoint's
//! core distance. This follows the pruning rule in the original CORE-SG code.

use std::cmp::Ordering;

use crate::cluster::hdbscan::hdbscan_common::{
    HdbscanHierarchy, WeightedEdge, compute_core_distances_tree, mutual_reachability_distance,
};
use crate::cluster::hierarchical::{Builder, idsize};
use crate::{DistanceData, Float, IndexQuery, KnnSearch};

/// A target and weight in one source node's CSR row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoreSGEdge<F: Float> {
    pub target: idsize,
    pub weight: F,
}

/// CORE-SG graph and the core distances used to construct it.
#[derive(Debug, Clone, PartialEq)]
pub struct CoreSG<F: Float> {
    /// CSR offsets into `adjacency`; neighbors for node `i` are
    /// `adjacency[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<usize>,
    /// Each undirected edge occurs once in each endpoint's neighbor slice.
    pub adjacency: Vec<CoreSGEdge<F>>,
    /// Core distances for the largest `min_points` value used to build graph.
    pub core_distances: Vec<F>,
    /// Largest `min_points` value supported by this graph.
    pub max_points: usize,
}

/// Build CORE-SG using a kNN index. `max_points` is the largest supported
/// `min_points` setting and counts the query point itself, matching `fuel`'s
/// HDBSCAN core-distance convention and the Python code.
pub fn coresg<'a, S, D, F>(tree: &S, data: &'a D, max_points: usize) -> Result<CoreSG<F>, String>
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
            // The per-iteration update is a single distance call per element,
            // so parallelizing it (e.g. with rayon) costs more in scheduling
            // overhead than it saves. Keep it serial.
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

    let mut sourced_adjacency = Vec::with_capacity(graph.len().saturating_mul(2));
    for ((a, b), weight) in graph {
        sourced_adjacency.push((a, CoreSGEdge { target: b, weight }));
        sourced_adjacency.push((b, CoreSGEdge { target: a, weight }));
    }
    sourced_adjacency.sort_unstable_by_key(|(source, edge)| (*source, edge.target));
    let mut offsets = vec![0usize; n + 1];
    for (source, _) in &sourced_adjacency {
        offsets[*source as usize + 1] += 1;
    }
    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }
    let adjacency = sourced_adjacency.into_iter().map(|(_, edge)| edge).collect();
    Ok(CoreSG { offsets, adjacency, core_distances, max_points })
}

impl<F: Float> CoreSG<F> {
    /// Return the adjacency slice for a source node, or `None` if out of range.
    pub fn neighbors(&self, source: usize) -> Option<&[CoreSGEdge<F>]> {
        let start = *self.offsets.get(source)?;
        let end = *self.offsets.get(source + 1)?;
        Some(&self.adjacency[start..end])
    }

    /// Number of unique undirected edges.
    pub fn num_edges(&self) -> usize { self.adjacency.len() / 2 }

    /// Rebuild the HDBSCAN mutual-reachability MST from this sparse graph.
    /// This lets the usual HDBSCAN flat, simplified, and GLOSH extraction
    /// functions operate on CORE-SG output.
    pub fn hdbscan_hierarchy(&self) -> Result<HdbscanHierarchy<F>, String> {
        self.hdbscan_hierarchy_for_core_distances(&self.core_distances)
    }

    /// Compute an HDBSCAN hierarchy for `min_points` from this precomputed
    /// CORE-SG graph. The graph may be reused for any positive value up to
    /// the `max_points` supplied to [`coresg`].
    pub fn hdbscan_hierarchy_for_min_points<'a, S, D>(
        &self, tree: &S, data: &'a D, min_points: usize,
    ) -> Result<HdbscanHierarchy<F>, String>
    where
        D: DistanceData<F> + 'a,
        S: KnnSearch<F, D::Query<'a>>,
    {
        assert!(min_points > 0, "min_points must be greater than 0");
        assert!(min_points <= self.max_points, "min_points exceeds CORE-SG max_points");
        assert_eq!(data.len(), self.core_distances.len(), "dataset size must match CORE-SG graph");
        let core_distances = compute_core_distances_tree(tree, data, min_points)?;
        self.hdbscan_hierarchy_for_core_distances(&core_distances)
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
        let mut edges = Vec::with_capacity(self.num_edges());
        for source in 0..n {
            for edge in self.neighbors(source).expect("valid CSR row") {
                if source < edge.target as usize {
                    edges.push(WeightedEdge::new(
                        source,
                        edge.target as usize,
                        core_distances[source]
                            .max(core_distances[edge.target as usize])
                            .max(edge.weight),
                    ));
                }
            }
        }
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
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::coresg;
    use crate::TableWithDistance;
    use crate::cluster::hdbscan::extraction::extract_clusters_with_noise;
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;

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

        let graph = coresg(&tree, &data, 3).unwrap();
        assert!((0..graph.core_distances.len()).all(|source| {
            graph.neighbors(source).unwrap().iter().all(|edge| {
                graph.neighbors(edge.target as usize).unwrap().iter().any(|reverse| {
                    reverse.target as usize == source && reverse.weight == edge.weight
                })
            })
        }));
        let hierarchy = graph.hdbscan_hierarchy_for_min_points(&tree, &data, 2).unwrap();
        let hierarchy_max = graph.hdbscan_hierarchy().unwrap();
        assert_eq!(hierarchy.merges.len(), points.len() - 1);
        assert_eq!(hierarchy_max.merges.len(), points.len() - 1);
        assert_ne!(hierarchy.core_distances, hierarchy_max.core_distances);
        let labels = extract_clusters_with_noise(&hierarchy.merges, 2, 2);
        assert_ne!(labels[0], labels[3]);
    }
}
