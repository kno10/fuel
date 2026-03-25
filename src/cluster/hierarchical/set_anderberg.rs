use std::cell::RefCell;
use std::rc::Rc;

use super::common::{
    PrototypeBuilder, PrototypeMergeHistory, initialize_set_clusters, run_anderberg_nn_cache,
    set_find_best, set_update_cache, triangle_index, update_set_entry,
};
// SetLinkage trait now lives in `linkage/mod.rs`; import via the public
// re-export on the parent hierarchical module.
use crate::cluster::hierarchical::SetLinkage;
use crate::{DistanceData, Float};

/// Shared implementation of the Anderberg/HACAM set‑linkage heuristic.  This
/// used to be duplicated in both `set_anderberg` and `hacam`, so the two
/// public entrypoints now delegate to a single helper.  The `recompute_y`
/// predicate determines whether the best candidate for cluster `y` is
/// re‑evaluated after a merge.  *set_anderberg* simply recomputes whenever
/// `y>0`; HACAM only recomputes if `y`'s previous nearest neighbor was the
/// cluster that just disappeared, giving a small performance win without
/// changing results.
#[must_use]
pub(super) fn set_anderberg_common<D, L, F, S, C>(
    data: &D, mut recompute_y: C,
) -> PrototypeMergeHistory<F>
where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
    C: FnMut(usize, usize, &[usize]) -> bool,
{
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    if n == 1 {
        return Vec::new();
    }

    let (members, summaries, distances, mut prototypes, _) =
        initialize_set_clusters::<D, L, F, S>(data);
    let members = Rc::new(RefCell::new(members));
    let summaries = Rc::new(RefCell::new(summaries));
    let summaries_for_update = summaries.clone();
    let summaries_for_recompute = summaries.clone();

    run_anderberg_nn_cache::<F, PrototypeBuilder<F>, _, _, _>(
        distances,
        n,
        move |distances,
              clustermap,
              _builder,
              bestd,
              besti,
              x,
              y,
              mindist,
              end,
              _size_x,
              _size_y,
              _offset,
              prototypes,
              proto| {
            let mut members_ref = members.borrow_mut();
            let mut summaries_ref = summaries_for_update.borrow_mut();

            let cx = std::mem::take(&mut members_ref[x]);
            members_ref[y].extend(cx);

            let summary_x = summaries_ref[x].take().expect("summary missing for x");
            let summary_y = summaries_ref[y].as_mut().expect("summary missing for y while merging");
            L::merge_summary(summary_y, summary_x, proto, mindist);

            update_matrices::<D, L, F, S>(
                data,
                distances,
                prototypes,
                clustermap,
                &members_ref,
                &summaries_ref,
                bestd,
                besti,
                x,
                y,
                end,
            );
        },
        move |y, x, clustermap, distances, bestd, besti| {
            let summaries_ref = summaries_for_recompute.borrow();
            let should_recompute = recompute_y(y, x, &*besti);
            if should_recompute {
                set_find_best::<D, F, L, S>(distances, clustermap, &summaries_ref, bestd, besti, y);
            }
        },
        move |mindist, _x, _y, offset, prototypes| (mindist, prototypes[offset]),
        false,
        &mut prototypes,
    )
}

/// Public wrapper that implements the traditional Anderberg heuristic (which
/// always recomputes the nearest neighbour for cluster `y` after a merge).
#[must_use]
pub fn set_anderberg<D, L, F, S>(data: &D) -> PrototypeMergeHistory<F>
where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    set_anderberg_common::<D, L, F, S, _>(data, |y, _x, _besti| y > 0)
}

#[allow(clippy::too_many_arguments)]
fn update_matrices<D, L, F, S>(
    data: &D, distances: &mut [F], prototypes: &mut [Option<usize>], clustermap: &[Option<usize>],
    members: &[Vec<usize>], summaries: &[Option<S>], bestd: &mut [F], besti: &mut [usize],
    x: usize, y: usize, end: usize,
) where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    if y > 0 {
        let yoffset = triangle_index(y, 0);
        for b in 0..y {
            if clustermap[b].is_none() {
                continue;
            }
            update_set_entry::<D, L, F, S>(data, distances, prototypes, members, summaries, y, b);
            let _ = set_update_cache::<D, F, L, S>(
                distances,
                clustermap,
                summaries,
                bestd,
                besti,
                x,
                y,
                b,
                distances[yoffset + b],
            );
        }
    }

    for a in (y + 1)..end {
        if clustermap[a].is_none() {
            continue;
        }
        update_set_entry::<D, L, F, S>(data, distances, prototypes, members, summaries, a, y);
        let d = distances[triangle_index(a, y)];
        let _ = set_update_cache::<D, F, L, S>(
            distances, clustermap, summaries, bestd, besti, x, y, a, d,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
mod tests {
    use super::set_anderberg;
    use crate::TableWithDistance;
    use crate::cluster::hierarchical::linkage::MinimaxLinkage;
    use crate::cluster::hierarchical::set_agnes::set_agnes;
    use crate::cluster::hierarchical::test_utils::ScalarDistance;

    #[test]
    fn set_anderberg_matches_set_agnes() {
        let points = [vec![0.0], vec![0.8], vec![2.0], vec![5.0], vec![9.0]];
        let data = TableWithDistance::with_distance(&points, ScalarDistance);
        let a = set_agnes(&data);
        let b = set_anderberg::<_, MinimaxLinkage, f64, ()>(&data);
        assert_eq!(a, b);
    }
}
