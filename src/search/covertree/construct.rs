use std::cmp::Ordering;

use crate::{DistPair, DistanceData, Float};

#[derive(Debug)]
pub(crate) struct CoverTreeNode<F>
where
    F: Float,
{
    pub(crate) center: usize,
    pub(crate) max_dist: F,
    pub(crate) parent_dist: F,
    pub(crate) children: Vec<u32>,
    pub(crate) singletons: Vec<DistPair<F>>,
}

impl<F: Float> CoverTreeNode<F> {
    fn new(center: usize, max_dist: F, parent_dist: F) -> Self {
        Self { center, max_dist, parent_dist, children: Vec::new(), singletons: Vec::new() }
    }
}

#[derive(Debug)]
pub struct CoverTree<F>
where
    F: Float,
{
    pub(crate) nodes: Vec<CoverTreeNode<F>>,
}

#[derive(Debug)]
struct CoverTreeBuilder<F>
where
    F: Float,
{
    nodes: Vec<CoverTreeNode<F>>,
    expansion: f64,
    scale_bottom: i32,
}

impl<F: Float> CoverTreeBuilder<F> {
    // Enable experimental nearest-center reassignment during cover completion.
    // This will assign cover candidates to their best parent center within fmax,
    // avoiding first-come-first-served assignment.
    const BEST_CENTER_REASSIGN: bool = true;

    fn new(expansion: f64) -> Self {
        let scale_bottom = (f64::MIN_POSITIVE.ln() / expansion.ln()).ceil() as i32;
        Self { nodes: Vec::new(), expansion, scale_bottom }
    }

    fn new_node(&mut self, center: usize, max_dist: F, parent_dist: F) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(CoverTreeNode::new(center, max_dist, parent_dist));
        idx
    }

    fn leaf(
        &mut self, center: usize, max_dist: F, parent_dist: F, candidates: Vec<DistPair<F>>,
    ) -> u32 {
        let idx = self.new_node(center, max_dist, parent_dist);
        self.nodes[idx as usize].singletons = candidates;
        idx
    }

    fn scale_to_dist(&self, scale: i32) -> F {
        let dist = self.expansion.powi(scale);
        F::from_f64(dist).unwrap_or(F::infinity())
    }

    fn dist_to_scale(&self, dist: F) -> i32 {
        let d = dist.to_f64().unwrap_or(0.0);
        if d <= 0.0 {
            return self.scale_bottom;
        }
        (d.ln() / self.expansion.ln()).ceil() as i32
    }

    fn max_distance(candidates: &[DistPair<F>]) -> F {
        candidates
            .iter()
            .map(|pair| pair.distance)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or_else(F::zero)
    }

    fn exclude_not_covered(
        candidates: &mut Vec<DistPair<F>>, fmax: F, collect: &mut Vec<DistPair<F>>,
    ) {
        let mut i = 0;
        while i < candidates.len() {
            if candidates[i].distance > fmax {
                collect.push(candidates.swap_remove(i));
            } else {
                i += 1;
            }
        }
    }

    fn collect_by_cover<D: DistanceData<F>>(
        data: &D, cur: usize, candidates: &mut Vec<DistPair<F>>, fmax: F,
        collect: &mut Vec<DistPair<F>>,
    ) {
        let mut i = 0;
        while i < candidates.len() {
            let candidate = candidates[i];
            let d = data.distance(cur, candidate.index);
            if d <= fmax {
                let _ = candidates.swap_remove(i);
                collect.push(DistPair::new(d, candidate.index));
            } else {
                i += 1;
            }
        }
    }

    fn bulk_construct<D: DistanceData<F>>(
        &mut self, data: &D, cur: usize, max_scale: i32, parent_dist: F,
        candidates: &mut Vec<DistPair<F>>,
    ) -> u32 {
        let max = Self::max_distance(candidates);
        let mut scale = self.dist_to_scale(max) - 1;
        if scale > max_scale {
            scale = max_scale;
        }
        let next_scale = scale - 1;

        if max <= F::zero() || scale <= self.scale_bottom {
            return self.leaf(cur, max, parent_dist, std::mem::take(candidates));
        }

        let mut cover_candidates = Vec::new();
        Self::exclude_not_covered(candidates, self.scale_to_dist(scale), &mut cover_candidates);

        if cover_candidates.is_empty() {
            return self.bulk_construct(data, cur, next_scale, parent_dist, candidates);
        }

        let node_index = self.new_node(cur, max, parent_dist);

        if !candidates.is_empty() {
            let child_idx = self.bulk_construct(data, cur, next_scale, F::zero(), candidates);
            self.nodes[node_index as usize].children.push(child_idx);
        }

        let fmax = self.scale_to_dist(next_scale);

        let mut buckets: Vec<(DistPair<F>, Vec<DistPair<F>>)> = Vec::new();
        while let Some(candidate) = cover_candidates.pop() {
            let mut elems = Vec::new();
            Self::collect_by_cover(data, candidate.index, &mut cover_candidates, fmax, &mut elems);
            buckets.push((candidate, elems));
        }

        if Self::BEST_CENTER_REASSIGN {
            let fmax_half = fmax / F::from_f64(2.0).unwrap_or(F::one() + F::one());

            let center_indices: Vec<usize> = buckets.iter().map(|(c, _)| c.index).collect();
            let bucket_count = buckets.len();
            let mut moves: Vec<Vec<(usize, F)>> = vec![Vec::new(); bucket_count];

            for (i, bucket_elems) in buckets.iter_mut().enumerate().take(bucket_count - 1) {
                for j in (0..bucket_elems.1.len()).rev() {
                    let point = bucket_elems.1[j];
                    if point.distance <= fmax_half {
                        continue;
                    }

                    let mut best_center = i;
                    let mut best_dist = point.distance;

                    for (k, &other_center_idx) in
                        center_indices.iter().enumerate().take(bucket_count)
                    {
                        if k == i {
                            continue;
                        }
                        let d = data.distance(other_center_idx, point.index);
                        if d <= fmax && d < best_dist {
                            best_dist = d;
                            best_center = k;
                        }
                    }

                    if best_center != i {
                        let _ = bucket_elems.1.swap_remove(j);
                        moves[best_center].push((point.index, best_dist));
                    }
                }
            }

            for (target_idx, entries) in moves.into_iter().enumerate() {
                let target = &mut buckets[target_idx].1;
                for (idx, dist) in entries {
                    target.push(DistPair::new(dist, idx));
                }
            }
        }

        for (center_pair, mut elems) in buckets {
            if elems.is_empty() {
                self.nodes[node_index as usize]
                    .singletons
                    .push(DistPair::new(center_pair.distance, center_pair.index));
            } else {
                let child_idx = self.bulk_construct(
                    data,
                    center_pair.index,
                    next_scale,
                    center_pair.distance,
                    &mut elems,
                );
                self.nodes[node_index as usize].children.push(child_idx);
            }
        }

        node_index
    }

    fn build<D: DistanceData<F>>(data: &D, root_idx: usize, expansion: f64) -> CoverTree<F> {
        let size = data.len();
        let mut candidates: Vec<DistPair<F>> = Vec::with_capacity(size.saturating_sub(1));
        for idx in 0..size {
            if idx == root_idx {
                continue;
            }
            candidates.push(DistPair::new(data.distance(root_idx, idx), idx));
        }

        let mut builder = CoverTreeBuilder::new(expansion);
        if candidates.is_empty() {
            builder.leaf(root_idx, F::zero(), F::zero(), Vec::new());
        } else {
            builder.bulk_construct(data, root_idx, i32::MAX, F::zero(), &mut candidates);
        }
        CoverTree { nodes: builder.nodes }
    }
}

impl<F: Float> CoverTree<F> {
    pub fn new<D: DistanceData<F>>(data: &D, expansion: f64, root_idx: usize) -> Self {
        let size = data.len();
        assert!(size > 0, "Data set must contain at least one point.");
        assert!(expansion > 1.0, "Expansion must be > 1");
        assert!(root_idx < size, "root_idx must be in [0, size)");

        CoverTreeBuilder::build(data, root_idx, expansion)
    }

    pub fn new_with_sampling<D: DistanceData<F>, R: rand::Rng + ?Sized>(
        data: &D, expansion: f64, sample_size: usize, rng: &mut R,
    ) -> Self {
        let root_idx = Self::choose_initial_center(data, sample_size, rng);
        Self::new(data, expansion, root_idx)
    }

    pub fn choose_initial_center<D: DistanceData<F>, R: rand::Rng + ?Sized>(
        data: &D, sample_size: usize, rng: &mut R,
    ) -> usize {
        let size = data.len();
        assert!(size > 0, "Data set must contain at least one point.");

        let sampled: Vec<usize> =
            rand::seq::index::sample(rng, size, sample_size.min(size)).into_vec();

        let mut best = (0, F::infinity());
        for candidate in 0..size {
            let sum = sampled.iter().map(|&i| data.distance(candidate, i)).sum::<F>();
            if sum < best.1 {
                best = (candidate, sum);
            }
        }

        best.0
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use crate::TableWithDistance;
    use crate::data::Data;
    use crate::distance::SquaredEuclidean;
    use crate::search::covertree::{CoverTree, expansion_heuristic_from_id};

    #[test]
    fn covertree_helpers_are_accessible_at_module_level() {
        let e = expansion_heuristic_from_id(10.0);
        assert!(e > 1.0 && e <= 2.0);

        let points =
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let center = CoverTree::<f64>::choose_initial_center(&data, 3, &mut rng);
        assert!(center < data.len());
    }

    #[test]
    fn expansion_heuristic_is_between_one_and_two() {
        for dim in &[1.0, 2.0, 10.0, 100.0, 1000.0] {
            let expansion = expansion_heuristic_from_id(*dim);
            assert!(
                expansion > 1.0 && expansion <= 2.0,
                "expansion {} for dim {} not in (1,2]",
                expansion,
                dim
            );
        }
    }
}
