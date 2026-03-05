use crate::DataAccess;

use super::common::{
    PrototypeBuilder, PrototypeMergeHistory, condensed_get, find_active,
    initialize_distances_and_prototypes, minimax_candidate, shrink_active_end, triangle_index,
};

/// MiniMax linkage with the nearest-neighbor chain heuristic.
#[must_use]
pub fn minimax_nn_chain<D: DataAccess>(data: &D) -> PrototypeMergeHistory<f64> {
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    if n == 1 {
        return Vec::new();
    }

    let (mut distances, mut prototypes) = initialize_distances_and_prototypes(data);
    let mut builder = PrototypeBuilder::new(n);
    let mut clustermap: Vec<Option<usize>> = (0..n).map(Some).collect();
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
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
            for i in 0..end {
                if i == a || i == b || clustermap[i].is_none() {
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
        update_matrices(
            data,
            &mut distances,
            &mut prototypes,
            &clustermap,
            &members,
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

fn update_matrices<D: DataAccess>(
    data: &D,
    distances: &mut [f64],
    prototypes: &mut [usize],
    clustermap: &[Option<usize>],
    members: &[Vec<usize>],
    c: usize,
    end: usize,
) {
    for y in 0..c {
        if clustermap[y].is_none() {
            continue;
        }
        update_entry(data, distances, prototypes, members, c, y);
    }
    for x in (c + 1)..end {
        if clustermap[x].is_none() {
            continue;
        }
        update_entry(data, distances, prototypes, members, x, c);
    }
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

    use super::minimax_nn_chain;
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
    fn minimax_nn_chain_matches_minimax() {
        let data = LineData(vec![0.0, 0.8, 2.0, 5.0, 9.0]);
        let a = minimax(&data);
        let b = minimax_nn_chain(&data);
        assert_eq!(a, b);
    }
}
