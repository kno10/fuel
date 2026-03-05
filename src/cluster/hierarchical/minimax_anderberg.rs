use crate::DataAccess;

use super::common::{
    PrototypeBuilder, PrototypeMergeHistory, initialize_distances_and_prototypes,
    minimax_candidate, shrink_active_end, triangle_index,
};

/// MiniMax linkage with Anderberg nearest-neighbor cache acceleration.
#[must_use]
pub fn minimax_anderberg<D: DataAccess>(data: &D) -> PrototypeMergeHistory<f64> {
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    if n == 1 {
        return Vec::new();
    }

    let (mut distances, mut prototypes) = initialize_distances_and_prototypes(data);
    let mut builder = PrototypeBuilder::new(n);
    let mut clustermap: Vec<Option<usize>> = (0..n).map(Some).collect();
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut end = n;

    let mut bestd = vec![f64::INFINITY; n];
    let mut besti = vec![usize::MAX; n];
    initialize_nn_cache(&distances, &clustermap, &mut bestd, &mut besti);

    for _ in 1..n {
        let (mindist, x, y) = find_merge(&bestd, &besti, &clustermap, end);
        let offset = triangle_index(x, y);
        let proto = prototypes[offset];

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
            &mut bestd,
            &mut besti,
            x,
            y,
            end,
        );
        if y > 0 {
            find_best(&distances, &clustermap, &mut bestd, &mut besti, y);
        }

        if x == end - 1 {
            shrink_active_end(&clustermap, &mut end);
        }
    }

    builder.into_merges()
}

fn initialize_nn_cache(
    distances: &[f64],
    clustermap: &[Option<usize>],
    bestd: &mut [f64],
    besti: &mut [usize],
) {
    for x in 1..bestd.len() {
        if clustermap[x].is_none() {
            continue;
        }
        find_best(distances, clustermap, bestd, besti, x);
    }
}

fn find_merge(
    bestd: &[f64],
    besti: &[usize],
    clustermap: &[Option<usize>],
    end: usize,
) -> (f64, usize, usize) {
    let mut mindist = f64::INFINITY;
    let mut x = usize::MAX;
    let mut y = usize::MAX;
    for cx in 1..end {
        if clustermap[cx].is_none() {
            continue;
        }
        let cy = besti[cx];
        if cy == usize::MAX {
            continue;
        }
        let d = bestd[cx];
        if d <= mindist {
            mindist = d;
            x = cx;
            y = cy;
        }
    }
    assert!(
        x != usize::MAX && y != usize::MAX,
        "no merge candidate found"
    );
    if y < x {
        (mindist, x, y)
    } else {
        (mindist, y, x)
    }
}

#[allow(clippy::too_many_arguments)]
fn update_matrices<D: DataAccess>(
    data: &D,
    distances: &mut [f64],
    prototypes: &mut [usize],
    clustermap: &[Option<usize>],
    members: &[Vec<usize>],
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
        update_entry(data, distances, prototypes, members, y, b);
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
        update_entry(data, distances, prototypes, members, a, y);
        let d = distances[triangle_index(a, y)];
        update_cache(distances, clustermap, bestd, besti, x, y, a, d);
    }
}

fn update_cache(
    distances: &[f64],
    clustermap: &[Option<usize>],
    bestd: &mut [f64],
    besti: &mut [usize],
    x: usize,
    y: usize,
    j: usize,
    d: f64,
) {
    if y < j && d <= bestd[j] {
        bestd[j] = d;
        besti[j] = y;
        return;
    }
    if besti[j] == x || besti[j] == y {
        find_best(distances, clustermap, bestd, besti, j);
    }
}

fn find_best(
    distances: &[f64],
    clustermap: &[Option<usize>],
    bestd: &mut [f64],
    besti: &mut [usize],
    j: usize,
) {
    let mut best_dist = f64::INFINITY;
    let mut best_idx = usize::MAX;
    for i in 0..j {
        if clustermap[i].is_none() {
            continue;
        }
        let d = distances[triangle_index(j, i)];
        if d <= best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    bestd[j] = best_dist;
    besti[j] = best_idx;
}

fn update_entry<D: DataAccess>(
    data: &D,
    distances: &mut [f64],
    prototypes: &mut [usize],
    members: &[Vec<usize>],
    x: usize,
    y: usize,
) {
    let (dist, proto) = minimax_candidate(data, &members[x], &members[y]);
    let offset = triangle_index(x, y);
    distances[offset] = dist;
    prototypes[offset] = proto;
}

#[cfg(test)]
mod tests {
    use crate::DataAccess;

    use super::minimax_anderberg;
    use crate::cluster::hierarchical::minimax;

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
    fn minimax_anderberg_matches_minimax() {
        let data = LineData(vec![0.0, 0.8, 2.0, 5.0, 9.0]);
        let a = minimax(&data);
        let b = minimax_anderberg(&data);
        assert_eq!(a, b);
    }
}
