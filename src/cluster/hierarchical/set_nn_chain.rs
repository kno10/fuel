use super::common::{
    PrototypeBuilder, PrototypeMergeHistory, condensed_get, find_active, initialize_set_clusters,
    shrink_active_end, triangle_index, update_set_entry,
};
use crate::cluster::hierarchical::SetLinkage;
use crate::{DistanceData, Float};

/// Nearest‑neighbor chain heuristic agglomeration using an arbitrary set‑based
/// linkage criterion.
#[must_use]
pub fn set_nn_chain<D, L, F, S>(data: &D) -> PrototypeMergeHistory<F>
where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");
    if n == 1 {
        return Vec::new();
    }

    let (mut members, mut summaries, mut distances, mut prototypes, mut clustermap) =
        initialize_set_clusters::<D, L, F, S>(data);
    let mut builder = PrototypeBuilder::<F>::new(n);
    let mut chain: Vec<usize> = Vec::with_capacity((n / 4).max(2));
    let mut end = n;
    let mut merged = 0usize;

    while merged < n - 1 {
        let mut a;
        let mut b;
        if chain.len() < 2 {
            a = find_active(0, end, &clustermap).expect("at least one active cluster");
            b = find_active(a + 1, end, &clustermap).expect("at least two active clusters");
            chain.clear();
            chain.push(a);
        } else {
            a = chain[chain.len() - 2];
            b = chain[chain.len() - 1];
            if clustermap[a].is_none() || clustermap[b].is_none() {
                chain.clear();
                continue;
            }
            chain.pop();
        }

        let mut min_dist = condensed_get(&distances, a, b);
        loop {
            let mut c = b;
            for (i, opt) in clustermap.iter().enumerate().take(end) {
                if i == a || i == b || opt.is_none() {
                    continue;
                }
                let d = condensed_get(&distances, a, i);
                if d < min_dist {
                    min_dist = d;
                    c = i;
                }
            }
            b = a;
            a = c;
            chain.push(a);
            if chain.len() >= 3 && a == chain[chain.len() - 3] {
                break;
            }
        }

        let (x, y) = if a > b { (a, b) } else { (b, a) };
        let offset = triangle_index(x, y);
        let proto = prototypes[offset];

        let xx = clustermap[x].expect("x must be active");
        let yy = clustermap[y].expect("y must be active");
        let new_id = builder.add(xx, min_dist, yy, proto);
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

        if chain.len() >= 3 {
            chain.truncate(chain.len() - 3);
        } else {
            chain.clear();
        }
        chain.push(y);
        merged += 1;
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
    for (y, opt) in clustermap.iter().enumerate().take(c) {
        if opt.is_none() {
            continue;
        }
        update_set_entry::<D, L, F, S>(data, distances, prototypes, members, summaries, c, y);
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
    use super::set_nn_chain;
    use crate::TableWithDistance;
    use crate::cluster::hierarchical::linkage::MinimaxLinkage;
    use crate::cluster::hierarchical::set_agnes::set_agnes;
    use crate::cluster::hierarchical::test_utils::ScalarDistance;

    #[test]
    fn set_nn_chain_matches_set_agnes() {
        let points = [vec![0.0], vec![0.8], vec![2.0], vec![5.0], vec![9.0]];
        let data = TableWithDistance::with_distance(&points, ScalarDistance);
        let a = set_agnes(&data);
        let b = set_nn_chain::<_, MinimaxLinkage, f64, ()>(&data);
        assert_eq!(a, b);
    }
}
