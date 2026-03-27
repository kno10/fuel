use std::cmp::Ordering;

use rand::Rng;

use crate::{DistanceData, Float};

#[derive(Debug)]
pub(crate) struct CoverTreeNode<F>
where
    F: Float,
{
    pub(crate) center: usize,
    pub(crate) max_dist: F,
    pub(crate) parent_dist: F,
    pub(crate) children: Vec<Box<CoverTreeNode<F>>>,
    pub(crate) singletons: Vec<(usize, F)>,
}

impl<F: Float> CoverTreeNode<F> {
    fn new(center: usize, max_dist: F, parent_dist: F) -> Box<Self> {
        Box::new(Self {
            center,
            max_dist,
            parent_dist,
            children: Vec::new(),
            singletons: Vec::new(),
        })
    }

    fn leaf(center: usize, max_dist: F, parent_dist: F, candidates: Vec<(usize, F)>) -> Box<Self> {
        let mut node = Self::new(center, max_dist, parent_dist);
        node.singletons.extend(candidates);
        node
    }
}

#[derive(Debug)]
pub struct CoverTree<F>
where
    F: Float,
{
    pub(crate) root: Option<Box<CoverTreeNode<F>>>,
    expansion: F,
    scale_bottom: i32,
    truncate: usize,
}

impl<F: Float> CoverTree<F> {
    /// Create a new cover tree instance from the supplied data.
    ///
    /// `expansion` is the cover-tree expansion rate (e.g., 1.3 in ELKI).
    /// `truncate` controls leaf size (roughly).
    pub fn new<D: DistanceData<F>, R: Rng>(
        data: &D, expansion: F, truncate: usize, rng: &mut R,
    ) -> Self {
        let size = data.size();
        assert!(size > 0, "Data set must contain at least one point.");
        assert!(expansion > F::one(), "Expansion must be > 1");

        let scale_bottom =
            (f64::MIN_POSITIVE.ln() / expansion.to_f64().unwrap().ln()).ceil() as i32;

        let root_idx = if size == 1 { 0 } else { rng.gen_range(0..size) };

        // Build candidate list with distances to root candidate.
        let mut candidates: Vec<(usize, F)> = Vec::new();
        for idx in 0..size {
            if idx == root_idx {
                continue;
            }
            candidates.push((idx, data.distance(root_idx, idx)));
        }

        let mut tree = Self { root: None, expansion, scale_bottom, truncate };

        let root_node = if candidates.is_empty() {
            CoverTreeNode::new(root_idx, F::zero(), F::zero())
        } else {
            tree.bulk_construct(data, root_idx, i32::MAX, F::zero(), &mut candidates)
        };

        tree.root = Some(root_node);
        tree
    }

    /// Create an incremental priority searcher for a query.
    pub fn priority_searcher(
        &self,
    ) -> crate::covertree::priority::CoverTreePrioritySearcher<'_, F> {
        crate::covertree::priority::CoverTreePrioritySearcher::new(self)
    }

    fn scale_to_dist(&self, scale: i32) -> F {
        let base = self.expansion.to_f64().unwrap();
        // exponent may be negative.
        let dist = base.powi(scale);
        F::from_f64(dist).unwrap_or(F::infinity())
    }

    fn dist_to_scale(&self, dist: F) -> i32 {
        let d = dist.to_f64().unwrap_or(0.0);
        if d <= 0.0 {
            self.scale_bottom
        } else {
            (d.ln() / self.expansion.to_f64().unwrap().ln()).ceil() as i32
        }
    }

    fn max_distance(candidates: &[(usize, F)]) -> F {
        candidates
            .iter()
            .map(|(_, d)| *d)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or_else(F::zero)
    }

    fn exclude_not_covered(
        candidates: &mut Vec<(usize, F)>, fmax: F, collect: &mut Vec<(usize, F)>,
    ) {
        let mut i = 0;
        while i < candidates.len() {
            if candidates[i].1 > fmax {
                let entry = candidates.swap_remove(i);
                collect.push(entry);
            } else {
                i += 1;
            }
        }
    }

    fn collect_by_cover<D: DistanceData<F>>(
        data: &D, cur: usize, candidates: &mut Vec<(usize, F)>, fmax: F,
        collect: &mut Vec<(usize, F)>,
    ) {
        let mut i = 0;
        while i < candidates.len() {
            let (idx, _) = candidates[i];
            let d = data.distance(cur, idx);
            if d <= fmax {
                let _ = candidates.swap_remove(i);
                collect.push((idx, d));
            } else {
                i += 1;
            }
        }
    }

    fn bulk_construct<D: DistanceData<F>>(
        &self, data: &D, cur: usize, max_scale: i32, parent_dist: F,
        candidates: &mut Vec<(usize, F)>,
    ) -> Box<CoverTreeNode<F>> {
        let max = Self::max_distance(candidates);
        let mut scale = self.dist_to_scale(max) - 1;
        if scale > max_scale {
            scale = max_scale;
        }
        let next_scale = scale - 1;

        if max <= F::zero() || scale <= self.scale_bottom || candidates.len() < self.truncate {
            return CoverTreeNode::leaf(cur, max, parent_dist, std::mem::take(candidates));
        }

        let mut cover_candidates = Vec::new();
        Self::exclude_not_covered(candidates, self.scale_to_dist(scale), &mut cover_candidates);

        if cover_candidates.is_empty() {
            return self.bulk_construct(data, cur, next_scale, parent_dist, candidates);
        }

        let mut node = CoverTreeNode::new(cur, max, parent_dist);

        if !candidates.is_empty() {
            let child = self.bulk_construct(data, cur, next_scale, F::zero(), candidates);
            node.children.push(child);
        }

        let fmax = self.scale_to_dist(next_scale);

        while let Some((candidate_idx, candidate_parent_dist)) = cover_candidates.pop() {
            let mut elems = Vec::new();
            Self::collect_by_cover(data, candidate_idx, &mut cover_candidates, fmax, &mut elems);

            if elems.is_empty() {
                node.singletons.push((candidate_idx, candidate_parent_dist));
            } else {
                let child = self.bulk_construct(
                    data,
                    candidate_idx,
                    next_scale,
                    candidate_parent_dist,
                    &mut elems,
                );
                node.children.push(child);
            }
        }

        node
    }
}
