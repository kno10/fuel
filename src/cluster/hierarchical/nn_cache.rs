use num_traits::Float;

use super::common::triangle_index;

pub(crate) fn initialize_nn_cache<F: Float>(
    distances: &[F],
    clustermap: &[Option<usize>],
    bestd: &mut [F],
    besti: &mut [usize],
) {
    let size = bestd.len();
    for x in 1..size {
        if clustermap[x].is_none() {
            continue;
        }
        find_best(distances, clustermap, bestd, besti, x);
    }
}

pub(crate) fn find_merge_scan<F: Float>(
    bestd: &[F],
    besti: &[usize],
    clustermap: &[Option<usize>],
    end: usize,
) -> (F, usize, usize) {
    let mut mindist = F::infinity();
    let mut x = usize::MAX;
    let mut y = usize::MAX;

    for cx in 1..end {
        if clustermap[cx].is_none() || besti[cx] == usize::MAX {
            continue;
        }
        let d = bestd[cx];
        if d <= mindist {
            mindist = d;
            x = cx;
            y = besti[cx];
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

pub(crate) fn update_cache<F: Float>(
    distances: &[F],
    clustermap: &[Option<usize>],
    bestd: &mut [F],
    besti: &mut [usize],
    x: usize,
    y: usize,
    j: usize,
    d: F,
) -> bool {
    if y < j && d <= bestd[j] {
        bestd[j] = d;
        besti[j] = y;
        return true;
    }

    if besti[j] == x || besti[j] == y {
        let oldd = bestd[j];
        let oldi = besti[j];
        find_best(distances, clustermap, bestd, besti, j);
        return besti[j] != oldi || bestd[j] != oldd;
    }

    false
}

pub(crate) fn find_best<F: Float>(
    distances: &[F],
    clustermap: &[Option<usize>],
    bestd: &mut [F],
    besti: &mut [usize],
    j: usize,
) {
    let mut best_dist = F::infinity();
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
