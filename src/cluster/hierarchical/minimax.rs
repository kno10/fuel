use crate::DataAccess;

use super::common::{
    PrototypeBuilder, PrototypeMergeHistory, find_best_active_pair,
    initialize_distances_and_prototypes, minimax_candidate, shrink_active_end, triangle_index,
};

/// MiniMax linkage hierarchical clustering.
///
/// This is a direct Rust port of the ELKI `MiniMax` behavior working with an
/// index-based `DataAccess` distance oracle.
#[must_use]
pub fn minimax<D: DataAccess>(data: &D) -> PrototypeMergeHistory<f64> {
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

    for _ in 1..n {
        let (x, y, mindist) = find_best_active_pair(&distances, &clustermap, end);
        let offset = triangle_index(x, y);
        let proto = prototypes[offset];

        let xx = clustermap[x].expect("x must be active");
        let yy = clustermap[y].expect("y must be active");
        let new_id = builder.add(xx, mindist, yy, proto);
        clustermap[y] = Some(new_id);
        clustermap[x] = None;

        // Keep y and append x to preserve deterministic candidate order.
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

    use super::minimax;

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
    fn minimax_produces_valid_hierarchy() {
        let data = LineData(vec![0.0, 1.0, 3.0, 10.0]);
        let h = minimax(&data);
        assert_eq!(h.len(), 3);
        assert_eq!(h.last().expect("non-empty").size, 4);
        assert!(h.iter().all(|m| m.prototype < 4));
    }
}
