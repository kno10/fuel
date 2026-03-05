use crate::DataAccess;

use super::common::{
    PrototypeBuilder, PrototypeMergeHistory, initialize_distances_and_prototypes,
    shrink_active_end, triangle_index,
};
use super::nn_cache::{find_best, find_merge_scan, initialize_nn_cache, update_cache};

/// HACAM objective variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HacamVariant {
    /// Minimize the sum-of-distances objective of the merged cluster medoid.
    MinimumSum,
    /// Minimize increase of the above objective.
    MinimumSumIncrease,
}

/// Hierarchical Agglomerative Clustering Around Medoids (HACAM).
#[must_use]
pub fn hacam<D: DataAccess>(data: &D, variant: HacamVariant) -> PrototypeMergeHistory<f64> {
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    if n == 1 {
        return Vec::new();
    }

    let (mut distances, mut prototypes) = initialize_distances_and_prototypes(data);
    let mut builder = PrototypeBuilder::new(n);
    let mut clustermap: Vec<Option<usize>> = (0..n).map(Some).collect();
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut tds = vec![0.0; n];
    let mut end = n;

    let mut bestd = vec![f64::INFINITY; n];
    let mut besti = vec![usize::MAX; n];
    initialize_nn_cache(&distances, &clustermap, &mut bestd, &mut besti);

    for _ in 1..n {
        let (mindist, x, y) = find_merge_scan(&bestd, &besti, &clustermap, end);
        let offset = triangle_index(x, y);
        let proto = prototypes[offset];

        if variant == HacamVariant::MinimumSumIncrease {
            tds[y] = distances[offset] + tds[x] + tds[y];
        }

        let xx = clustermap[x].expect("x must be active");
        let yy = clustermap[y].expect("y must be active");
        let new_id = builder.add(xx, mindist, yy, proto);
        clustermap[y] = Some(new_id);
        clustermap[x] = None;
        besti[x] = usize::MAX;
        bestd[x] = f64::INFINITY;

        let cx = std::mem::take(&mut members[x]);
        members[y].extend(cx);

        update_matrices(
            data,
            &mut distances,
            &mut prototypes,
            &clustermap,
            &members,
            &tds,
            variant,
            &mut bestd,
            &mut besti,
            x,
            y,
            end,
        );
        if besti[y] == x {
            find_best(&distances, &clustermap, &mut bestd, &mut besti, y);
        }

        if x == end - 1 {
            shrink_active_end(&clustermap, &mut end);
        }
    }

    builder.into_merges()
}

#[allow(clippy::too_many_arguments)]
fn update_matrices<D: DataAccess>(
    data: &D,
    distances: &mut [f64],
    prototypes: &mut [usize],
    clustermap: &[Option<usize>],
    members: &[Vec<usize>],
    tds: &[f64],
    variant: HacamVariant,
    bestd: &mut [f64],
    besti: &mut [usize],
    x: usize,
    y: usize,
    end: usize,
) {
    let yoffset = if y > 0 { triangle_index(y, 0) } else { 0 };
    for b in 0..y {
        if clustermap[b].is_none() {
            continue;
        }
        update_entry(data, distances, prototypes, members, tds, variant, y, b);
        update_cache(
            distances,
            clustermap,
            bestd,
            besti,
            x,
            y,
            b,
            distances[yoffset + b],
        );
    }
    for a in (y + 1)..end {
        if clustermap[a].is_none() {
            continue;
        }
        update_entry(data, distances, prototypes, members, tds, variant, a, y);
        let d = distances[triangle_index(a, y)];
        update_cache(distances, clustermap, bestd, besti, x, y, a, d);
    }
}

fn update_entry<D: DataAccess>(
    data: &D,
    distances: &mut [f64],
    prototypes: &mut [usize],
    members: &[Vec<usize>],
    tds: &[f64],
    variant: HacamVariant,
    x: usize,
    y: usize,
) {
    let (mut dist, proto) = minimum_sum_candidate(data, &members[x], &members[y]);
    if variant == HacamVariant::MinimumSumIncrease {
        dist -= tds[x] + tds[y];
    }
    let offset = triangle_index(x, y);
    distances[offset] = dist;
    prototypes[offset] = proto;
}

fn minimum_sum_candidate<D: DataAccess>(data: &D, cx: &[usize], cy: &[usize]) -> (f64, usize) {
    let mut best_sum = f64::INFINITY;
    let mut best_proto = cx[0];
    for &cand in cx.iter().chain(cy.iter()) {
        let mut sum = 0.0;
        for &p in cx {
            sum += data.distance(cand, p);
            if sum >= best_sum {
                break;
            }
        }
        if sum >= best_sum {
            continue;
        }
        for &p in cy {
            sum += data.distance(cand, p);
            if sum >= best_sum {
                break;
            }
        }
        if sum < best_sum {
            best_sum = sum;
            best_proto = cand;
        }
    }
    (best_sum, best_proto)
}

#[cfg(test)]
mod tests {
    use crate::DataAccess;

    use super::{HacamVariant, hacam};

    struct LineData(Vec<f64>);
    impl DataAccess for LineData {
        fn distance(&self, a: usize, b: usize) -> f64 {
            (self.0[a] - self.0[b]).abs()
        }
        fn query_distance(&self, _b: usize) -> f64 {
            unreachable!()
        }
        fn size(&self) -> usize {
            self.0.len()
        }
    }

    #[test]
    fn hacam_variants_return_valid_histories() {
        let data = LineData(vec![0.0, 0.5, 2.0, 3.0, 8.0]);
        let a = hacam(&data, HacamVariant::MinimumSum);
        let b = hacam(&data, HacamVariant::MinimumSumIncrease);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 4);
        assert_eq!(a.last().expect("non-empty").size, 5);
        assert_eq!(b.last().expect("non-empty").size, 5);
    }
}
