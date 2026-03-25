use crate::Float;
use crate::api::{NodePoints, SearchFilter};
use crate::cluster::hierarchical::common::{Merge, MergeHistory};

/// Edge collected during search-based single-linkage construction.
struct Edge<F> {
    a: usize,
    b: usize,
    weight: F,
}

pub(crate) struct ClusterBuilder<F: Float> {
    n: usize,
    uf_parent: Vec<usize>,
    uf_size: Vec<usize>,
    merge_count: usize,
    /// Raw MST edges (original point pairs) collected during construction.
    edges: Vec<Edge<F>>,
}

impl<F: Float> ClusterBuilder<F> {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            n,
            uf_parent: (0..n).collect(),
            uf_size: vec![1; n],
            merge_count: 0,
            edges: Vec::with_capacity(n.saturating_sub(1)),
        }
    }

    pub(crate) fn merge_count(&self) -> usize { self.merge_count }

    pub(crate) fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.uf_parent[root] != root {
            root = self.uf_parent[root];
        }
        let mut node = x;
        while self.uf_parent[node] != root {
            let next = self.uf_parent[node];
            self.uf_parent[node] = root;
            node = next;
        }
        root
    }

    pub(crate) fn same_set(&mut self, a: usize, b: usize) -> bool { self.find(a) == self.find(b) }

    pub(crate) fn cluster_size_of_point(&mut self, a: usize) -> usize {
        let cida = self.find(a);
        self.uf_size[cida]
    }

    pub(crate) fn merge_points(&mut self, a: usize, b: usize, distance: F) -> Option<usize> {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return None;
        }

        self.edges.push(Edge { a, b, weight: distance });
        self.merge_count += 1;

        if self.uf_size[ra] < self.uf_size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.uf_parent[rb] = ra;
        self.uf_size[ra] += self.uf_size[rb];
        Some(ra)
    }

    /// Consume the builder and produce a sorted merge history.
    ///
    /// Sorts collected MST edges by weight and rebuilds the dendrogram
    /// with a fresh union-find, ensuring non-decreasing merge distances.
    pub(crate) fn into_history(mut self) -> MergeHistory<F> {
        let n = self.n;
        if n <= 1 {
            return Vec::new();
        }

        self.edges.sort_by(|l, r| {
            l.weight
                .partial_cmp(&r.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| l.a.min(l.b).cmp(&r.a.min(r.b)))
                .then_with(|| l.a.max(l.b).cmp(&r.a.max(r.b)))
        });

        let mut parent: Vec<usize> = (0..(2 * n - 1)).collect();
        let mut size = vec![1usize; 2 * n - 1];
        let mut merges = Vec::<Merge<F>>::with_capacity(n - 1);

        for edge in &self.edges {
            let s = uf_find(&mut parent, edge.a);
            let t = uf_find(&mut parent, edge.b);
            if s == t {
                continue;
            }

            let ss = size[s];
            let st = size[t];
            let (idx1, idx2) = if s <= t { (s, t) } else { (t, s) };
            merges.push(Merge { idx1, idx2, distance: edge.weight, size: ss + st });

            let new_id = n + merges.len() - 1;
            parent[s] = new_id;
            parent[t] = new_id;
            parent[new_id] = new_id;
            size[new_id] = ss + st;

            if merges.len() == n - 1 {
                break;
            }
        }

        debug_assert_eq!(merges.len(), n - 1, "edge set did not connect all points");
        merges
    }
}

fn uf_find(parent: &mut [usize], x: usize) -> usize {
    let mut p = x;
    while parent[p] != p {
        p = parent[p];
    }
    let mut i = x;
    while parent[i] != p {
        let next = parent[i];
        parent[i] = p;
        i = next;
    }
    p
}

pub(crate) struct SameClusterFilter<'a, F: Float> {
    pub(crate) builder: &'a mut ClusterBuilder<F>,
    pub(crate) query_component: usize,
    /// Per-node cache indexed by vantage-point dataset index.
    /// Encoding (assuming n < 2^31):
    /// - `u32::MAX`            = unknown (initial state)
    /// - bit 31 clear          = uniform, value is cluster representative
    /// - bit 31 set, != MAX    = non-uniform witness (lower 31 bits = point index
    ///   of a point in a different cluster than the VP)
    pub(crate) node_cluster: &'a mut [u32],
}

const WITNESS_BIT: u32 = 1 << 31;

impl<'a, F: Float> SearchFilter for SameClusterFilter<'a, F> {
    fn skip_node(&mut self, points: NodePoints<'_>) -> bool {
        let vp = points.first_index();
        let cached = self.node_cluster[vp];

        // Fast path: cached witness for non-uniformity
        if cached != u32::MAX && (cached & WITNESS_BIT) != 0 {
            let witness = (cached & !WITNESS_BIT) as usize;
            if self.builder.find(vp) != self.builder.find(witness) {
                // Still non-uniform, cannot skip
                return false;
            }
            // Witness merged into VP's cluster, fall through to re-scan
        } else if cached != u32::MAX {
            // Fast path: cached uniform cluster
            return self.builder.find(cached as usize) == self.query_component;
        }

        // Unknown or invalidated witness: scan all points
        let mut component = u32::MAX;
        for i in points.indices() {
            let c = self.builder.find(i) as u32;
            if component == u32::MAX {
                component = c;
            } else if c != component {
                // Non-uniform: cache witness
                self.node_cluster[vp] = WITNESS_BIT | (i as u32);
                return false;
            }
        }
        // All points share the same cluster - cache it.
        if component != u32::MAX {
            self.node_cluster[vp] = component;
        }
        component != u32::MAX && (component as usize) == self.query_component
    }

    fn skip_point(&mut self, index: usize) -> bool {
        self.builder.find(index) == self.query_component
    }
}
