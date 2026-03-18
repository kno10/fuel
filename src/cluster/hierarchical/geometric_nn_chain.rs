use num_traits::Float;

use super::common::{Builder, MergeHistory, find_active, shrink_active_end};
use super::linkage::GeometricLinkage;

/// Perform NN-Chain clustering on vector data with geometric linkage.
///
/// This variant avoids the `O(n^2)` condensed matrix and instead keeps only
/// current cluster representatives. Runtime is still `O(n^3)` in the worst
/// case, but memory usage is `O(n * dim)`.
#[must_use]
pub fn geometric_nn_chain<F: Float, L: GeometricLinkage<F> + Copy>(
    vectors: &[Vec<F>], // FIXME: use the data API
    linkage: L,
    is_squared: bool,
) -> MergeHistory<F> {
    let n = vectors.len();
    assert!(n > 0, "number of points must be positive");

    let dim = vectors[0].len();
    assert!(
        vectors.iter().all(|v| v.len() == dim),
        "all vectors must have equal dimensionality"
    );

    let mut builder = Builder::<F>::new(n);
    let mut clustermap: Vec<Option<usize>> = (0..n).map(Some).collect();
    let mut clusters: Vec<Option<Vec<F>>> = vectors.iter().cloned().map(Some).collect();
    let mut end = n;
    let mut chain: Vec<usize> = Vec::with_capacity(n.min(64));
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
            if clustermap[a].is_none() {
                chain.truncate(chain.len() - 2);
                continue;
            }
            chain.pop();
        }

        let mut min_dist = {
            let aa = clusters[a].as_ref().expect("a center must exist");
            let bb = clusters[b].as_ref().expect("b center must exist");
            let size_a = builder.get_size(clustermap[a].expect("a id must exist"));
            let size_b = builder.get_size(clustermap[b].expect("b id must exist"));
            linkage.linkage(aa, size_a, bb, size_b)
        };

        loop {
            let mut c = b;
            let a_center = clusters[a].as_ref().expect("a center must exist");
            let size_a = builder.get_size(clustermap[a].expect("a id must exist"));
            for i in 0..end {
                if i == a || i == b || clustermap[i].is_none() {
                    continue;
                }
                let i_center = clusters[i].as_ref().expect("i center must exist");
                let size_i = builder.get_size(clustermap[i].expect("i id must exist"));
                let d = linkage.linkage(a_center, size_a, i_center, size_i);
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
        let cid_x = clustermap[x].expect("x must be active");
        let cid_y = clustermap[y].expect("y must be active");
        let size_x = builder.get_size(cid_x);
        let size_y = builder.get_size(cid_y);

        let (h1, h2) = if cid_y <= cid_x {
            (cid_y, cid_x)
        } else {
            (cid_x, cid_y)
        };
        let new_id = builder.add(h1, linkage.restore_linkage(min_dist, is_squared), h2);

        let merged_center = {
            let cx = clusters[x].as_ref().expect("x center must exist");
            let cy = clusters[y].as_ref().expect("y center must exist");
            linkage.merge(cx, size_x, cy, size_y)
        };

        clustermap[y] = Some(new_id);
        clustermap[x] = None;
        clusters[y] = Some(merged_center);
        clusters[x] = None;

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

#[cfg(test)]
mod tests {
    use crate::cluster::hierarchical::CentroidLinkage;
    use crate::cluster::hierarchical::agnes;

    use super::geometric_nn_chain;

    fn condensed_squared_euclidean(points: &[Vec<f64>]) -> Vec<f64> {
        let n = points.len();
        let mut out = Vec::with_capacity(n * (n - 1) / 2);
        for i in 1..n {
            for j in 0..i {
                let mut s = 0.0;
                for (a, b) in points[i].iter().zip(points[j].iter()) {
                    let d = a - b;
                    s += d * d;
                }
                out.push(s);
            }
        }
        out
    }

    #[test]
    fn geometric_nn_chain_matches_agnes_for_centroid_linkage() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![3.0, 3.0],
            vec![3.1, 3.2],
            vec![10.0, 10.0],
        ];

        let condensed = condensed_squared_euclidean(&points);
        let a = agnes(&condensed, points.len(), CentroidLinkage, true);
        let b = geometric_nn_chain(&points, CentroidLinkage, true);
        assert_eq!(a.len(), b.len());
        for (ma, mb) in a.iter().zip(b.iter()) {
            assert_eq!(ma.idx1, mb.idx1);
            assert_eq!(ma.idx2, mb.idx2);
            assert_eq!(ma.size, mb.size);
            assert!((ma.distance - mb.distance).abs() < 1e-12);
        }
    }
}
