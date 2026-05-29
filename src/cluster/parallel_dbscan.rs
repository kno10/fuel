use std::sync::Mutex;

use rayon::prelude::*;

use crate::api::{DistanceData, RangeSearch};
use crate::cluster::dbscan::{DbscanResult, NOISE};
use crate::{Float, IndexQuery};

/// Run DBSCAN in parallel, querying each point's neighbors exactly once.
///
/// Follows the ELKI `ParallelGeneralizedDBSCAN` design: each worker thread
/// computes the eps-neighborhood of its assigned point without any
/// synchronization.  If the point is a core, it acquires a single shared
/// lock to merge clusters and label unassigned neighbors as border in the
/// same step.  A short sequential finalization resolves the union-find and
/// renumbers the cluster labels.
///
/// Reference: ELKI `ParallelGeneralizedDBSCAN` (Schubert et al.).
/// Related: Patwary et al., SC 2012 (<https://doi.org/10.1109/SC.2012.9>).
pub fn parallel_dbscan<'a, S, D, F>(
    tree: &S, data: &'a D, eps: F, min_points: usize,
) -> Result<DbscanResult, String>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    D::Query<'a>: Send,
    S: RangeSearch<F, D::Query<'a>> + Sync,
{
    assert!(eps >= F::zero(), "eps must be non-negative");
    assert!(min_points > 0, "min_points must be greater than 0");

    let size = data.len();
    if size == 0 {
        return Ok(DbscanResult { labels: Vec::new(), core: Vec::new() });
    }

    // Shared mutable state protected by a single mutex.
    // The mutex is only held for the labeling step (O(eps-neighbors)),
    // not for the neighbor query (the dominant cost).
    struct Shared {
        /// -1 = unassigned, >= 0 = cluster ID (before union-find resolution).
        assignments: Vec<isize>,
        is_core: Vec<bool>,
        /// Min-label union-find over cluster IDs (like ELKI's Core.num array).
        cluster_parent: Vec<usize>,
        next_cluster: usize,
    }

    impl Shared {
        fn alloc_cluster(&mut self) -> usize {
            let id = self.next_cluster;
            self.next_cluster += 1;
            self.cluster_parent.push(id);
            id
        }

        /// Find root with path-halving compression.
        fn find(&mut self, mut c: usize) -> usize {
            while self.cluster_parent[c] != c {
                self.cluster_parent[c] = self.cluster_parent[self.cluster_parent[c]];
                c = self.cluster_parent[c];
            }
            c
        }

        /// Union by min-label: the smaller ID becomes the representative.
        fn union_clusters(&mut self, a: usize, b: usize) {
            let ra = self.find(a);
            let rb = self.find(b);
            if ra != rb {
                let (small, large) = if ra < rb { (ra, rb) } else { (rb, ra) };
                self.cluster_parent[large] = small;
            }
        }
    }

    let shared = Mutex::new(Shared {
        assignments: vec![-1; size],
        is_core: vec![false; size],
        cluster_parent: Vec::new(),
        next_cluster: 0,
    });

    // One parallel pass: each point's neighbors are queried exactly once.
    (0..size).into_par_iter().for_each_init(
        || data.query(),
        |query, idx| {
            query.set_index(idx);
            // Collect neighbors outside the lock (the expensive part).
            let neighbors: Vec<usize> =
                tree.search_range(query, eps).into_iter().map(|p| p.index).collect();
            if neighbors.len() < min_points {
                return; // Not a core; may be labeled as border later.
            }

            // Core point: acquire the lock for the labeling step (like ELKI's
            // synchronized block in processNeighbors).
            let mut st = shared.lock().unwrap();

            // Find and merge all clusters that already-processed core neighbors
            // belong to.  After this loop `cluster` is a union-find root.
            let mut cluster: Option<usize> = None;
            for &nb in &neighbors {
                if !st.is_core[nb] {
                    continue;
                }
                let nb_root = {
                    let nb_assign = st.assignments[nb] as usize;
                    st.find(nb_assign)
                };
                cluster = Some(match cluster {
                    None => nb_root,
                    Some(c) => {
                        // c is already a root (maintained by prior iterations).
                        st.union_clusters(c, nb_root);
                        c.min(nb_root)
                    }
                });
            }

            // Resolve the final representative (or create a new cluster).
            let cid = match cluster {
                Some(c) => st.find(c),
                None => st.alloc_cluster(),
            };

            // Mark this point as a core and assign to the cluster.
            st.is_core[idx] = true;
            st.assignments[idx] = cid as isize;

            // Label every unassigned neighbor as a border of this cluster.
            // Already-assigned borders keep their first assignment (ELKI's
            // behavior); already-processed cores were merged above.
            for &nb in &neighbors {
                if st.assignments[nb] < 0 {
                    st.assignments[nb] = cid as isize;
                }
            }
        },
    );
    crate::poll_interrupted()?;

    // Sequential finalization: resolve union-find and remap to dense labels.
    let mut st = shared.into_inner().unwrap();
    let mut label_remap: Vec<isize> = vec![-1; st.next_cluster];
    let mut next_label: isize = 0;

    let labels: Vec<isize> = (0..size)
        .map(|i| {
            let a = st.assignments[i];
            if a < 0 {
                return NOISE;
            }
            let root = st.find(a as usize);
            if label_remap[root] < 0 {
                label_remap[root] = next_label;
                next_label += 1;
            }
            label_remap[root]
        })
        .collect();

    Ok(DbscanResult { labels, core: st.is_core })
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::cluster::dbscan::dbscan;
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;

    fn build_tree(
        points: &[Vec<f64>],
    ) -> (TableWithDistance<'_, f64, Vec<f64>, Euclidean, f64>, VPTree<f64>) {
        let data = TableWithDistance::with_distance(points, Euclidean);
        let mut rng = StdRng::seed_from_u64(7);
        let tree = VPTree::new(&data, 2, &mut rng);
        (data, tree)
    }

    #[test]
    fn matches_sequential_two_clusters() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![5.0, 5.0],
        ];
        let (data, tree) = build_tree(&points);
        let parallel_labels = parallel_dbscan(&tree, &data, 0.25, 3);
        let sequential_labels = dbscan(&tree, &data, 0.25, 3);
        assert_eq!(parallel_labels.unwrap().labels, sequential_labels.unwrap().labels);
    }

    #[test]
    fn matches_expected_noise_cases() {
        let points =
            vec![vec![0.0], vec![2.0], vec![3.0], vec![4.0], vec![6.0], vec![8.0], vec![10.0]];
        let (data, tree) = build_tree(&points);
        let expected_cases = [
            (1, vec![0, 1, 1, 1, 2, 3, 4]),
            (2, vec![NOISE, 0, 0, 0, NOISE, NOISE, NOISE]),
            (3, vec![NOISE, 0, 0, 0, NOISE, NOISE, NOISE]),
            (4, vec![NOISE, NOISE, NOISE, NOISE, NOISE, NOISE, NOISE]),
        ];
        for (min_points, expected) in expected_cases {
            let parallel_labels = parallel_dbscan(&tree, &data, 1.0, min_points).unwrap();
            assert_eq!(parallel_labels.labels, expected);
        }
    }
}
