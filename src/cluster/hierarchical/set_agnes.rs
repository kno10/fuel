use super::linkage::{MinimaxLinkage, SetLinkage};
use crate::{DistanceData, Float};

/// General agglomerative clustering (AGNES) using MiniMax linkage.
#[must_use]
pub fn set_agnes<D, F>(data: &D) -> PrototypeMergeHistory<F>
where
    D: DistanceData<F>,
    F: Float,
{
    set_linkage::<D, MinimaxLinkage, F, _>(data)
}

// ----------------------------------------------------------------------
// Generic set‑linkage implementation previously in `set_linkage.rs`.
// This routine drives the clustering logic for any type implementing the
// `SetLinkage` trait now defined in `linkage/mod.rs`.  Keeping the algorithm
// here makes the overall `set` workflows easier to follow.

use super::common::{
    PrototypeBuilder, PrototypeMergeHistory, initialize_set_clusters, set_find_best_active_pair,
    shrink_active_end, triangle_index, update_set_entry,
};

/// Generic agglomerative clustering routine parameterised by a `SetLinkage`
/// implementation.
#[must_use]
pub fn set_linkage<D, L, F, S>(data: &D) -> PrototypeMergeHistory<F>
where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    if n == 1 {
        return Vec::new();
    }

    let (mut members, mut summaries, mut distances, mut prototypes, mut clustermap) =
        initialize_set_clusters::<D, L, F, S>(data);

    let mut builder = PrototypeBuilder::<F>::new(n);
    let mut end = n;

    for _ in 1..n {
        let (x, y, mindist) =
            set_find_best_active_pair::<D, F, L, S>(&distances, &clustermap, &summaries, end);
        let offset = triangle_index(x, y);
        let proto = prototypes[offset];

        let xx = clustermap[x].expect("x must be active");
        let yy = clustermap[y].expect("y must be active");
        let new_id = builder.add(xx, mindist, yy, proto);
        clustermap[y] = Some(new_id);
        clustermap[x] = None;

        let cx = std::mem::take(&mut members[x]);
        members[y].extend(cx);

        let summary_x = summaries[x].take().expect("summary missing for x");
        let summary_y = summaries[y].as_mut().expect("summary missing for y while merging");
        L::merge_summary(summary_y, summary_x, proto, distances[offset]);

        update_matrices::<D, L, F, S>(
            data,
            &mut distances,
            &mut prototypes,
            &clustermap,
            &members,
            &summaries,
            y,
            end,
        );
        if x == end - 1 {
            shrink_active_end(&clustermap, &mut end);
        }
    }

    builder.into_merges()
}

fn update_matrices<D, L, F, S>(
    data: &D, distances: &mut [F], prototypes: &mut [Option<usize>], clustermap: &[Option<usize>],
    members: &[Vec<usize>], summaries: &[Option<S>], c: usize, end: usize,
) where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    for (j, opt) in clustermap.iter().enumerate().take(c) {
        if opt.is_none() {
            continue;
        }
        update_set_entry::<D, L, F, S>(data, distances, prototypes, members, summaries, c, j);
    }
    for (x, opt) in clustermap.iter().enumerate().skip(c + 1).take(end - (c + 1)) {
        if opt.is_none() {
            continue;
        }
        update_set_entry::<D, L, F, S>(data, distances, prototypes, members, summaries, x, c);
    }
}

#[cfg(test)]
mod tests {
    use super::set_agnes;
    use crate::TableWithDistance;
    use crate::cluster::hierarchical::test_utils::ScalarDistance;

    #[test]
    fn set_agnes_produces_valid_hierarchy() {
        let points = [vec![0.0], vec![1.0], vec![3.0], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, ScalarDistance);
        let h = set_agnes(&data);
        assert_eq!(h.len(), 3);
        assert_eq!(h.last().expect("non-empty").size, 4);
        assert!(h.iter().all(|m| m.prototype < Some(4)));
    }
}
