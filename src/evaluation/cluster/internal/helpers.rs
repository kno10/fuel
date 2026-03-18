//! Common types and utility functions shared by the various internal
//! clustering quality measures.

use num_traits::Float;

use crate::distance::squared_euclidean_distance;

/// Strategy for handling points tagged as noise while computing a measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseHandling {
    MergeNoise,
    TreatNoiseAsSingletons,
    IgnoreNoise,
}

/// Internal representation of a cluster built from a label vector.
#[derive(Clone, Debug)]
pub struct Cluster {
    pub members: Vec<usize>,
    pub is_noise: bool,
}

/// Result returned by [`squared_errors`](super::squared_errors).
#[derive(Debug, Clone, Copy)]
pub struct BasicDistanceStats<F: Float> {
    pub mean: F,
    pub sum_of_squares: F,
    pub rmsd: F,
}

/// Return value for the silhouette-related functions.
#[derive(Debug, Clone)]
pub struct SilhouetteStats<F: Float> {
    pub mean: F,
    pub stddev: F,
    pub values: Vec<F>,
}

/// Radius stats returned by [`super::cluster_radius`].
#[derive(Debug, Clone, Copy)]
pub struct RadiusStats<F: Float> {
    pub weighted: F,
    pub unweighted: F,
}

/// Concordance statistics for pairwise gamma/tau.
#[derive(Debug, Clone, Copy)]
pub struct ConcordanceStats<F: Float> {
    pub gamma: F,
    pub tau: F,
}

/// Neighbor consistency results produced by
/// [`super::neighbor_consistency_knn`].
#[derive(Debug, Clone)]
pub struct NeighborConsistencyStats<F: Float> {
    pub average: F,
    pub full: F,
    pub per_element_average: Vec<F>,
    pub per_element_full: Vec<F>,
}

/// Basic squared‑and‑euclidean helpers operating on `&[f64]` slices.
#[inline]
pub fn sq_euc(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean_distance::<f64, f64>(a, b)
}

#[inline]
pub fn euc(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean_distance::<f64, f64>(a, b).sqrt()
}

pub fn mean_std(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    if xs.len() <= 1 {
        return (mean, 0.0);
    }
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

pub fn centroid(data: &[Vec<f64>], members: &[usize]) -> Vec<f64> {
    let dim = data[0].len();
    let mut c = vec![0.0; dim];
    if members.is_empty() {
        return c;
    }
    for &idx in members {
        for (d, val) in data[idx].iter().enumerate() {
            c[d] += *val;
        }
    }
    let inv = 1.0 / members.len() as f64;
    for x in &mut c {
        *x *= inv;
    }
    c
}

pub fn build_clusters(
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
) -> (Vec<Cluster>, usize) {
    let mut map: std::collections::BTreeMap<isize, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        map.entry(l).or_default().push(i);
    }

    let mut clusters = Vec::new();
    let mut ignored = 0usize;

    for (label, members) in map {
        let is_noise = noise_label == Some(label);
        if is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => {
                    ignored += members.len();
                    continue;
                }
                NoiseHandling::TreatNoiseAsSingletons => {
                    for idx in members {
                        clusters.push(Cluster {
                            members: vec![idx],
                            is_noise: true,
                        });
                    }
                    continue;
                }
                NoiseHandling::MergeNoise => {}
            }
        }
        clusters.push(Cluster { members, is_noise });
    }

    (clusters, ignored)
}

pub fn cluster_centroids(
    data: &[Vec<f64>],
    clusters: &[Cluster],
    nh: NoiseHandling,
) -> Vec<Option<Vec<f64>>> {
    clusters
        .iter()
        .map(|cl| {
            if cl.members.len() <= 1 || cl.is_noise {
                match nh {
                    NoiseHandling::IgnoreNoise => None,
                    NoiseHandling::TreatNoiseAsSingletons | NoiseHandling::MergeNoise => None,
                }
            } else {
                Some(centroid(data, &cl.members))
            }
        })
        .collect()
}

pub fn lower_bound(arr: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len();
    while lo < hi {
        let mid = usize::midpoint(lo, hi);
        if arr[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

pub fn upper_bound(arr: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len();
    while lo < hi {
        let mid = usize::midpoint(lo, hi);
        if arr[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

pub fn prim_mst_dense(matrix: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let n = matrix.len();
    if n <= 1 {
        return Vec::new();
    }
    let mut in_mst = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    let mut parent = vec![usize::MAX; n];

    best[0] = 0.0;
    for _ in 0..n {
        let mut u = usize::MAX;
        let mut bu = f64::INFINITY;
        for i in 0..n {
            if !in_mst[i] && best[i] < bu {
                bu = best[i];
                u = i;
            }
        }
        if u == usize::MAX {
            break;
        }
        in_mst[u] = true;
        for v in 0..n {
            if !in_mst[v] && matrix[u][v] < best[v] {
                best[v] = matrix[u][v];
                parent[v] = u;
            }
        }
    }

    let mut edges = Vec::with_capacity(n - 1);
    for (v, &p) in parent.iter().enumerate().skip(1) {
        if p != usize::MAX {
            edges.push((v, p));
        }
    }
    edges
}
