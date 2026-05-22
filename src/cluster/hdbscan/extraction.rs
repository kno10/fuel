use std::collections::{BTreeMap, HashMap};

use crate::Float;
use crate::cluster::dbscan::NOISE;
use crate::cluster::hdbscan::HdbscanHierarchy;
use crate::cluster::hierarchical::MergeHistory;

/// One cluster node in an extracted hierarchy.
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchyNode<F: Float> {
    /// Merge height / distance of this node.
    pub distance: F,
    /// Members directly attached to this node.
    pub members: Vec<usize>,
    /// Child node indices into [`ExtractedHierarchy::nodes`].
    pub children: Vec<usize>,
}

/// Extracted hierarchy represented as nodes plus root indices.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedHierarchy<F: Float> {
    pub nodes: Vec<HierarchyNode<F>>,
    pub roots: Vec<usize>,
}

impl<F: Float> ExtractedHierarchy<F> {
    #[must_use]
    pub fn new() -> Self { Self { nodes: Vec::new(), roots: Vec::new() } }

    fn push_node(&mut self, distance: F, members: Vec<usize>, children: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(HierarchyNode { distance, members, children });
        id
    }
}

impl<F: Float> Default for ExtractedHierarchy<F> {
    fn default() -> Self { Self::new() }
}

/// Result of the ELKI-style HDBSCAN hierarchy extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct HdbscanHierarchyExtractionResult<F: Float> {
    pub hierarchy: ExtractedHierarchy<F>,
    /// Per-point GLOSH outlier scores.
    pub glosh: Vec<F>,
}

/// Port of ELKI's `ClustersWithNoiseExtraction` for merge histories.
///
/// Returns one label per point:
/// - `NOISE` for points assigned to noise
/// - `0..k-1` for extracted clusters
#[must_use]
pub fn extract_clusters_with_noise<F: Float>(
    history: &MergeHistory<F>, num_clusters: usize, min_cluster_size: usize,
) -> Vec<isize> {
    assert!(num_clusters > 0, "num_clusters must be positive");
    assert!(min_cluster_size > 0, "min_cluster_size must be positive");

    let n = history.len() + 1;
    if n == 1 {
        return vec![0];
    }

    let mut cur_good: isize = 0;
    let mut best_cl: isize = n as isize;
    let mut best_off: isize = -1;

    for (i, merge) in history.iter().enumerate() {
        let sa = cluster_size(history, n, merge.idx1);
        let sb = cluster_size(history, n, merge.idx2);
        let sc = merge.size;

        cur_good += -((sa >= min_cluster_size) as isize) - ((sb >= min_cluster_size) as isize)
            + (sc >= min_cluster_size) as isize;

        if cur_good == num_clusters as isize
            || (cur_good - num_clusters as isize).abs() < (best_cl - num_clusters as isize).abs()
        {
            best_cl = cur_good;
            best_off = i as isize;
        }
    }

    let mut leaf_map: HashMap<usize, usize> = HashMap::with_capacity(n);
    let mut cluster_members: Vec<Vec<usize>> = Vec::with_capacity(n);
    cluster_members.push(Vec::new()); // noise bucket at index 0

    for i in (0..=best_off).rev() {
        let i = i as usize;
        let merge = history.get(i).unwrap();

        let mut c = leaf_map.remove(&(i + n));
        if c.is_none() && merge.size < min_cluster_size {
            c = Some(0);
        }
        let c = c.unwrap_or_else(|| {
            let next = cluster_members.len();
            cluster_members.push(Vec::new());
            next
        });

        if merge.idx2 < n {
            cluster_members[c].push(merge.idx2);
        } else {
            leaf_map.insert(merge.idx2, c);
        }

        if merge.idx1 < n {
            cluster_members[c].push(merge.idx1);
        } else {
            leaf_map.insert(merge.idx1, c);
        }
    }

    let mut labels = vec![NOISE; n];
    for (cluster_idx, members) in cluster_members.iter().enumerate().skip(1) {
        let label = (cluster_idx - 1) as isize;
        for &p in members {
            labels[p] = label;
        }
    }
    labels
}

/// Port of ELKI's `SimplifiedHierarchyExtraction`.
#[must_use]
pub fn extract_simplified_hierarchy<F: Float>(
    history: &MergeHistory<F>, core_distances: Option<&[F]>, min_cluster_size: usize,
) -> ExtractedHierarchy<F> {
    assert!(min_cluster_size > 0, "min_cluster_size must be positive");

    let n = history.len() + 1;
    if let Some(core) = core_distances {
        assert_eq!(core.len(), n, "core_distances length must match points");
    }

    let mut cluster_map: BTreeMap<usize, SimplifiedTempCluster<F>> = BTreeMap::new();
    let mut hierarchy = ExtractedHierarchy::new();
    let mut leaves_cache = vec![None; n + history.len()];

    for (i, merge) in history.iter().enumerate() {
        let dist = merge.distance;
        let a = merge.idx1;
        let b = merge.idx2;

        let mut aclus = cluster_map.remove(&a);
        let a_is_core = core_distances.is_none_or(|core| a >= n || dist >= core[a]);
        let a_not_spurious = aclus.as_ref().is_some_and(|c| c.is_not_spurious(min_cluster_size))
            || (aclus.is_none() && min_cluster_size <= 1 && a_is_core);

        let mut bclus = cluster_map.remove(&b);
        let b_is_core = core_distances.is_none_or(|core| b >= n || dist <= core[b]);
        let b_not_spurious = bclus.as_ref().is_some_and(|c| c.is_not_spurious(min_cluster_size))
            || (bclus.is_none() && min_cluster_size <= 1 && b_is_core);

        let mut nclus =
            if let (Some(mut aclus_val), Some(mut bclus_val)) = (aclus.take(), bclus.take()) {
                if a_not_spurious && b_not_spurious {
                    let bnode = simplified_to_node(&mut bclus_val, &mut hierarchy);
                    let anode = simplified_to_node(&mut aclus_val, &mut hierarchy);
                    bclus_val.children.push(bnode);
                    bclus_val.children.push(anode);
                    bclus_val.depth = dist;
                    bclus_val
                } else if a_not_spurious {
                    aclus_val.newids.extend(bclus_val.newids);
                    aclus_val.depth = dist;
                    aclus_val
                } else {
                    bclus_val.newids.extend(aclus_val.newids);
                    bclus_val.depth = dist;
                    bclus_val
                }
            } else if let Some(mut aclus_val) = aclus.take() {
                if a_not_spurious && b_not_spurious {
                    let anode = simplified_to_node(&mut aclus_val, &mut hierarchy);
                    aclus_val.children.push(anode);
                }
                simplified_add_id(
                    &mut aclus_val,
                    b,
                    dist,
                    b_not_spurious,
                    &mut hierarchy,
                    history,
                    n,
                    &mut leaves_cache,
                );
                aclus_val
            } else if let Some(mut bclus_val) = bclus.take() {
                if a_not_spurious && b_not_spurious {
                    let bnode = simplified_to_node(&mut bclus_val, &mut hierarchy);
                    bclus_val.children.push(bnode);
                }
                simplified_add_id(
                    &mut bclus_val,
                    a,
                    dist,
                    a_not_spurious,
                    &mut hierarchy,
                    history,
                    n,
                    &mut leaves_cache,
                );
                bclus_val
            } else {
                let mut tmp = SimplifiedTempCluster::new(dist);
                simplified_add_id(
                    &mut tmp,
                    a,
                    dist,
                    a_not_spurious,
                    &mut hierarchy,
                    history,
                    n,
                    &mut leaves_cache,
                );
                simplified_add_id(
                    &mut tmp,
                    b,
                    dist,
                    b_not_spurious,
                    &mut hierarchy,
                    history,
                    n,
                    &mut leaves_cache,
                );
                tmp
            };

        nclus.depth = dist;
        cluster_map.insert(i + n, nclus);
    }

    for (_, mut clus) in cluster_map {
        let root = simplified_to_node(&mut clus, &mut hierarchy);
        hierarchy.roots.push(root);
    }

    hierarchy
}

/// Convenience wrapper for simplified extraction on `HdbscanHierarchy`.
#[must_use]
pub fn extract_simplified_hierarchy_hdbscan<F: Float>(
    hierarchy: &HdbscanHierarchy<F>, min_cluster_size: usize,
) -> ExtractedHierarchy<F> {
    extract_simplified_hierarchy(
        &hierarchy.merges,
        Some(&hierarchy.core_distances),
        min_cluster_size,
    )
}

/// Port of ELKI's `HDBSCANHierarchyExtraction`, including GLOSH scores.
#[must_use]
pub fn extract_hdbscan_hierarchy<F: Float>(
    history: &MergeHistory<F>, core_distances: Option<&[F]>, min_cluster_size: usize,
    hierarchical: bool,
) -> HdbscanHierarchyExtractionResult<F> {
    assert!(min_cluster_size > 0, "min_cluster_size must be positive");

    let n = history.len() + 1;
    if let Some(core) = core_distances {
        assert_eq!(core.len(), n, "core_distances length must match points");
    }

    let mut cluster_map: BTreeMap<usize, HdbscanTempCluster<F>> = BTreeMap::new();

    for (i, merge) in history.iter().enumerate() {
        let dist = merge.distance;
        let a = merge.idx1;
        let b = merge.idx2;

        let cclus = cluster_map.remove(&a);
        let cdist = core_distances.map_or(dist, |core| if a < n { core[a] } else { dist });
        let c_spurious = hdbscan_is_spurious(cclus.as_ref(), min_cluster_size, cdist <= dist);

        let oclus = cluster_map.remove(&b);
        let odist = core_distances.map_or(dist, |core| if b < n { core[b] } else { dist });
        let o_spurious = hdbscan_is_spurious(oclus.as_ref(), min_cluster_size, odist <= dist);

        let nclus = if !c_spurious && !o_spurious {
            let c = cclus.unwrap_or_else(|| HdbscanTempCluster::new_leaf(cdist, a));
            let o = oclus.unwrap_or_else(|| HdbscanTempCluster::new_leaf(odist, b));
            HdbscanTempCluster::new_parent(dist, o, c)
        } else if !o_spurious {
            if let Some(oc) = oclus {
                if a < n {
                    oc.grow_point(dist, a)
                } else {
                    oc.grow_cluster(dist, cclus.expect("cluster must exist"))
                }
            } else {
                HdbscanTempCluster::new_leaf(dist, a)
            }
        } else if !c_spurious {
            if let Some(cc) = cclus {
                if b < n {
                    cc.grow_point(dist, b)
                } else {
                    cc.grow_cluster(dist, oclus.expect("cluster must exist"))
                }
            } else {
                HdbscanTempCluster::new_leaf(dist, b)
            }
        } else if let Some(oc) = oclus {
            if a < n {
                oc.grow_point(dist, a).reset_aggregate()
            } else {
                oc.grow_cluster(dist, cclus.expect("cluster must exist")).reset_aggregate()
            }
        } else if let Some(cc) = cclus {
            if b < n {
                cc.grow_point(dist, b).reset_aggregate()
            } else {
                cc.grow_cluster(dist, oclus.expect("cluster must exist")).reset_aggregate()
            }
        } else {
            let mut tmp = HdbscanTempCluster::new_leaf(dist, a);
            tmp.members.push(b);
            tmp
        };

        cluster_map.insert(i + n, nclus);
    }

    let mut out = HdbscanHierarchyExtractionResult {
        hierarchy: ExtractedHierarchy::new(),
        glosh: vec![F::zero(); n],
    };

    for (_, clus) in cluster_map {
        finalize_hdbscan_cluster(clus, &mut out, core_distances, None, false, hierarchical);
    }

    out
}

/// Convenience wrapper for ELKI-style HDBSCAN extraction on `HdbscanHierarchy`.
#[must_use]
pub fn extract_hdbscan_hierarchy_hdbscan<F: Float>(
    hierarchy: &HdbscanHierarchy<F>, min_cluster_size: usize, hierarchical: bool,
) -> HdbscanHierarchyExtractionResult<F> {
    extract_hdbscan_hierarchy(
        &hierarchy.merges,
        Some(&hierarchy.core_distances),
        min_cluster_size,
        hierarchical,
    )
}

#[inline]
fn cluster_size<F: Float>(history: &MergeHistory<F>, n: usize, id: usize) -> usize {
    if id < n { 1 } else { history.get(id - n).unwrap().size }
}

fn collect_leaf_points<F: Float>(
    history: &MergeHistory<F>, n: usize, id: usize, cache: &mut [Option<Vec<usize>>],
) -> Vec<usize> {
    if let Some(v) = &cache[id] {
        return v.clone();
    }

    let leaves = if id < n {
        vec![id]
    } else {
        let merge = history.get(id - n).unwrap();
        let mut left = collect_leaf_points(history, n, merge.idx1, cache);
        let mut right = collect_leaf_points(history, n, merge.idx2, cache);
        left.append(&mut right);
        left
    };
    cache[id] = Some(leaves.clone());
    leaves
}

#[derive(Debug, Clone)]
struct SimplifiedTempCluster<F: Float> {
    newids: Vec<usize>,
    depth: F,
    children: Vec<usize>,
}

impl<F: Float> SimplifiedTempCluster<F> {
    fn new(depth: F) -> Self { Self { newids: Vec::new(), depth, children: Vec::new() } }

    fn is_not_spurious(&self, min_cluster_size: usize) -> bool {
        !self.children.is_empty() || self.newids.len() >= min_cluster_size
    }
}

#[allow(clippy::too_many_arguments)]
fn simplified_add_id<F: Float>(
    clus: &mut SimplifiedTempCluster<F>, id: usize, dist: F, as_cluster: bool,
    hierarchy: &mut ExtractedHierarchy<F>, history: &MergeHistory<F>, n: usize,
    leaves_cache: &mut [Option<Vec<usize>>],
) {
    let members = collect_leaf_points(history, n, id, leaves_cache);
    if as_cluster {
        let child = hierarchy.push_node(dist, members, Vec::new());
        clus.children.push(child);
    } else {
        clus.newids.extend(members);
    }
    clus.depth = dist;
}

fn simplified_to_node<F: Float>(
    temp: &mut SimplifiedTempCluster<F>, hierarchy: &mut ExtractedHierarchy<F>,
) -> usize {
    let members = std::mem::take(&mut temp.newids);
    let children = std::mem::take(&mut temp.children);
    hierarchy.push_node(temp.depth, members, children)
}

#[derive(Debug, Clone)]
struct HdbscanTempCluster<F: Float> {
    members: Vec<usize>,
    dist: F,
    dmin: F,
    aggregate: F,
    children_total: usize,
    children: Vec<HdbscanTempCluster<F>>,
}

impl<F: Float> HdbscanTempCluster<F> {
    fn new_leaf(dist: F, point: usize) -> Self {
        Self {
            members: vec![point],
            dist,
            dmin: dist,
            aggregate: F::one() / dist,
            children_total: 0,
            children: Vec::new(),
        }
    }

    fn new_parent(dist: F, a: Self, b: Self) -> Self {
        let children_total = a.total_elements() + b.total_elements();
        Self {
            members: Vec::new(),
            dist,
            dmin: a.dmin.min(b.dmin),
            aggregate: F::from_usize(children_total).unwrap() / dist,
            children_total,
            children: vec![a, b],
        }
    }

    fn grow_cluster(mut self, dist: F, other: Self) -> Self {
        debug_assert!(other.children.is_empty());
        self.dist = dist;
        self.dmin = self.dmin.min(other.dmin);
        self.aggregate += F::from_usize(other.members.len()).unwrap() / dist;
        self.members.extend(other.members);
        self
    }

    fn grow_point(mut self, dist: F, point: usize) -> Self {
        self.dist = dist;
        self.dmin = dist;
        self.aggregate += F::one() / dist;
        self.members.push(point);
        self
    }

    fn reset_aggregate(mut self) -> Self {
        self.aggregate = F::from_usize(self.total_elements()).unwrap() / self.dist;
        self.dmin = self.dist;
        self
    }

    fn total_elements(&self) -> usize { self.children_total + self.members.len() }

    fn excess_of_mass(&self) -> F {
        self.aggregate - F::from_usize(self.total_elements()).unwrap() / self.dist
    }

    fn total_stability(&self) -> F {
        let stability = self.excess_of_mass();
        let cstab = self.children.iter().map(HdbscanTempCluster::total_stability).map(F::abs).sum();
        if stability > cstab { stability } else { -cstab }
    }

    fn is_spurious(&self, min_cluster_size: usize) -> bool {
        self.children.is_empty() && self.members.len() < min_cluster_size
    }
}

#[inline]
fn hdbscan_is_spurious<F: Float>(
    clus: Option<&HdbscanTempCluster<F>>, min_cluster_size: usize, is_core: bool,
) -> bool {
    match clus {
        Some(c) => c.is_spurious(min_cluster_size),
        None => min_cluster_size > 1 || !is_core,
    }
}

fn finalize_hdbscan_cluster<F: Float>(
    mut temp: HdbscanTempCluster<F>, out: &mut HdbscanHierarchyExtractionResult<F>,
    core_distances: Option<&[F]>, parent: Option<usize>, flatten: bool, hierarchical: bool,
) -> F {
    let node_id = out.hierarchy.push_node(temp.dist, std::mem::take(&mut temp.members), Vec::new());

    if hierarchical {
        if let Some(parent_id) = parent {
            out.hierarchy.nodes[parent_id].children.push(node_id);
        } else {
            out.hierarchy.roots.push(node_id);
        }
    } else {
        out.hierarchy.roots.push(node_id);
    }

    let mut dmin = temp.dmin;
    for child in temp.children {
        let cdmin = if flatten || child.total_stability() < F::zero() {
            collect_hdbscan_children(child, out, core_distances, node_id, flatten, hierarchical)
        } else {
            finalize_hdbscan_cluster(child, out, core_distances, Some(node_id), true, hierarchical)
        };
        dmin = dmin.min(cdmin);
    }

    for &point in &out.hierarchy.nodes[node_id].members {
        let mut cdist = core_distances.map_or(temp.dist, |core| core[point]);
        if cdist < dmin {
            cdist = dmin;
        }
        out.glosh[point] = if cdist > F::zero() { F::one() - dmin / cdist } else { F::zero() };
    }

    dmin
}

fn collect_hdbscan_children<F: Float>(
    cur: HdbscanTempCluster<F>, out: &mut HdbscanHierarchyExtractionResult<F>,
    core_distances: Option<&[F]>, parent: usize, flatten: bool, hierarchical: bool,
) -> F {
    let mut dmin = cur.dmin;
    out.hierarchy.nodes[parent].members.extend(cur.members);

    for child in cur.children {
        let cdmin = if flatten || child.total_stability() < F::zero() {
            collect_hdbscan_children(child, out, core_distances, parent, flatten, hierarchical)
        } else {
            finalize_hdbscan_cluster(child, out, core_distances, Some(parent), true, hierarchical)
        };
        dmin = dmin.min(cdmin);
    }

    dmin
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        extract_clusters_with_noise, extract_hdbscan_hierarchy, extract_hdbscan_hierarchy_hdbscan,
        extract_simplified_hierarchy, extract_simplified_hierarchy_hdbscan,
    };
    use crate::cluster::dbscan::NOISE;
    use crate::cluster::hdbscan::HdbscanHierarchy;
    use crate::cluster::hierarchical::{Merge, MergeHistory};

    fn all_members(
        extracted_roots: &[usize], nodes: &[super::HierarchyNode<f64>],
    ) -> BTreeSet<usize> {
        fn visit(node: usize, nodes: &[super::HierarchyNode<f64>], out: &mut BTreeSet<usize>) {
            for &p in &nodes[node].members {
                out.insert(p);
            }
            for &child in &nodes[node].children {
                visit(child, nodes, out);
            }
        }

        let mut members = BTreeSet::new();
        for &root in extracted_roots {
            visit(root, nodes, &mut members);
        }
        members
    }

    #[test]
    fn clusters_with_noise_relaxes_k_and_emits_noise() {
        let history: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 2, idx2: 3, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 6, idx2: 4, distance: 2.0, size: 3, prototype: usize::MAX },
            Merge { idx1: 7, idx2: 5, distance: 2.0, size: 3, prototype: usize::MAX },
            Merge { idx1: 8, idx2: 9, distance: 10.0, size: 6, prototype: usize::MAX },
        ]
        .into();

        let labels = extract_clusters_with_noise(&history, 3, 2);
        assert_eq!(labels[4], NOISE);
        assert_eq!(labels[5], NOISE);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn clusters_with_noise_can_hit_exact_k_without_noise() {
        let history: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 2, idx2: 3, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 4, idx2: 5, distance: 2.0, size: 4, prototype: usize::MAX },
        ]
        .into();

        let labels = extract_clusters_with_noise(&history, 2, 2);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
        assert!(labels.iter().all(|&x| x >= 0));
    }

    #[test]
    fn clusters_with_noise_quality_regression() {
        use crate::cluster::hierarchical::test::{
            DATASETS, evaluate_clustering_isize, expected_quality, load_dataset,
        };
        use crate::cluster::hierarchical::{GroupAverageLinkage, WardLinkage, agnes};
        use crate::distance::Euclidean;
        use crate::{CondensedDistanceMatrix, TableWithDistance};

        for dataset in DATASETS {
            let (features, truth) = load_dataset(dataset.name);
            let access = TableWithDistance::with_distance(&features, Euclidean);
            let condensed: CondensedDistanceMatrix<f64> =
                CondensedDistanceMatrix::new_from_data(&access);
            let history = match dataset.name {
                "nested_clusters" => agnes(&condensed, WardLinkage),
                _ => agnes(&condensed, GroupAverageLinkage),
            }
            .unwrap();
            let labels = extract_clusters_with_noise(&history, dataset.clusters, 2);
            let (ari, nmi) =
                evaluate_clustering_isize(&labels, &truth, Some(crate::cluster::dbscan::NOISE));
            let (ref_ari, ref_nmi, _) = expected_quality("HDBSCAN", "hdbscan", dataset.name);
            let tolerance = 1e-6;
            assert!(
                (ari - ref_ari).abs() <= tolerance,
                "{} ARI differs: {:.12} vs {:.12}",
                dataset.name,
                ari,
                ref_ari
            );
            assert!(
                (nmi - ref_nmi).abs() <= tolerance,
                "{} NMI differs: {:.12} vs {:.12}",
                dataset.name,
                nmi,
                ref_nmi
            );
        }
    }

    #[test]
    fn simplified_extraction_keeps_non_spurious_children() {
        let history: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 2, idx2: 3, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 4, idx2: 5, distance: 2.0, size: 4, prototype: usize::MAX },
        ]
        .into();
        let core = vec![0.5, 0.5, 0.5, 0.5];

        let extracted = extract_simplified_hierarchy(&history, Some(&core), 2);
        assert_eq!(extracted.roots.len(), 1);

        let root = &extracted.nodes[extracted.roots[0]];
        assert_eq!(root.children.len(), 2);
        let c0 = &extracted.nodes[root.children[0]];
        let c1 = &extracted.nodes[root.children[1]];

        assert_eq!(c0.members.len(), 2);
        assert_eq!(c1.members.len(), 2);
    }

    #[test]
    fn simplified_hierarchy_wrapper_matches_direct_call() {
        let history: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 2, idx2: 3, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 4, idx2: 5, distance: 2.0, size: 4, prototype: usize::MAX },
        ]
        .into();
        let core = vec![1.0, 1.0, 1.0, 1.0];
        let hdb = HdbscanHierarchy { merges: history.clone(), core_distances: core.clone() };

        let direct = extract_simplified_hierarchy(&history, Some(&core), 2);
        let wrapped = extract_simplified_hierarchy_hdbscan(&hdb, 2);
        assert_eq!(direct, wrapped);
    }

    #[test]
    fn hdbscan_extraction_returns_glosh_for_all_points() {
        let history: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 2, idx2: 3, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 4, idx2: 5, distance: 2.0, size: 4, prototype: usize::MAX },
        ]
        .into();
        let core = vec![1.0, 1.0, 1.0, 1.0];

        let extracted = extract_hdbscan_hierarchy::<f64>(&history, Some(&core), 2, false);
        assert_eq!(extracted.glosh.len(), 4);
        assert!(extracted.glosh.iter().all(|v| v.is_finite()));
        assert!(!extracted.hierarchy.roots.is_empty());
    }

    #[test]
    fn hdbscan_hierarchical_mode_changes_tree_shape_not_membership() {
        let history: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 2, idx2: 3, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 4, idx2: 5, distance: 2.0, size: 4, prototype: usize::MAX },
        ]
        .into();
        let core = vec![1.0, 1.0, 1.0, 1.0];
        let hdb = HdbscanHierarchy { merges: history.clone(), core_distances: core.clone() };

        let hierarchical = extract_hdbscan_hierarchy_hdbscan(&hdb, 2, true);
        let flat = extract_hdbscan_hierarchy(&history, Some(&core), 2, false);

        let m_h = all_members(&hierarchical.hierarchy.roots, &hierarchical.hierarchy.nodes);
        let m_f = all_members(&flat.hierarchy.roots, &flat.hierarchy.nodes);

        assert_eq!(m_h, m_f);
        assert_eq!(m_h.len(), 4);
        assert!(hierarchical.hierarchy.roots.len() <= flat.hierarchy.roots.len());
    }
}
