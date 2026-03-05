use crate::DataAccess;

use super::common::{
    PrototypeBuilder, PrototypeMergeHistory, find_best_active_pair, initialize_distances,
    shrink_active_end, triangle_index,
};

/// Hierarchical clustering with medoid-linkage criterion.
///
/// Cluster distance is defined as the distance between current cluster medoids.
#[must_use]
pub fn medoid_linkage<D: DataAccess>(data: &D) -> PrototypeMergeHistory<f64> {
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    if n == 1 {
        return Vec::new();
    }

    let mut distances = initialize_distances(data);
    let mut builder = PrototypeBuilder::new(n);
    let mut clustermap: Vec<Option<usize>> = (0..n).map(Some).collect();
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut medoids: Vec<usize> = (0..n).collect();
    let mut end = n;

    for _ in 1..n {
        let (x, y, mindist) = find_best_active_pair(&distances, &clustermap, end);

        let cx = std::mem::take(&mut members[x]);
        members[y].extend(cx);
        medoids[y] = find_medoid(data, &members[y]);

        let xx = clustermap[x].expect("x must be active");
        let yy = clustermap[y].expect("y must be active");
        let new_id = builder.add(xx, mindist, yy, medoids[y]);
        clustermap[y] = Some(new_id);
        clustermap[x] = None;

        update_matrix(data, &mut distances, &clustermap, &medoids, x, y, end);
        if x == end - 1 {
            shrink_active_end(&clustermap, &mut end);
        }
    }

    builder.into_merges()
}

fn find_medoid<D: DataAccess>(data: &D, cluster: &[usize]) -> usize {
    let mut best = cluster[0];
    let mut min_sum = f64::INFINITY;

    for &cand in cluster {
        let mut sum = 0.0;
        for &other in cluster {
            if cand != other {
                sum += data.distance(cand, other);
                if sum >= min_sum {
                    break;
                }
            }
        }
        if sum < min_sum {
            min_sum = sum;
            best = cand;
        }
    }
    best
}

fn update_matrix<D: DataAccess>(
    data: &D,
    distances: &mut [f64],
    clustermap: &[Option<usize>],
    medoids: &[usize],
    x: usize,
    y: usize,
    end: usize,
) {
    for j in 0..y {
        if clustermap[j].is_none() {
            continue;
        }
        distances[triangle_index(y, j)] = data.distance(medoids[y], medoids[j]);
    }
    for j in (y + 1)..x {
        if clustermap[j].is_none() {
            continue;
        }
        distances[triangle_index(j, y)] = data.distance(medoids[y], medoids[j]);
    }
    for j in (x + 1)..end {
        if clustermap[j].is_none() {
            continue;
        }
        distances[triangle_index(j, y)] = data.distance(medoids[y], medoids[j]);
    }
}

#[cfg(test)]
mod tests {
    use crate::DataAccess;

    use super::medoid_linkage;

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
    fn medoid_linkage_produces_valid_history() {
        let data = LineData(vec![0.0, 1.0, 3.0, 10.0]);
        let h = medoid_linkage(&data);
        assert_eq!(h.len(), 3);
        assert_eq!(h.last().expect("non-empty").size, 4);
        assert!(h.iter().all(|m| m.prototype < 4));
    }
}
