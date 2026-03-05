use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseHandling {
    MergeNoise,
    TreatNoiseAsSingletons,
    IgnoreNoise,
}

#[derive(Clone, Debug)]
struct Cluster {
    members: Vec<usize>,
    is_noise: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct BasicDistanceStats {
    pub mean: f64,
    pub sum_of_squares: f64,
    pub rmsd: f64,
}

#[derive(Debug, Clone)]
pub struct SilhouetteStats {
    pub mean: f64,
    pub stddev: f64,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct RadiusStats {
    pub weighted: f64,
    pub unweighted: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ConcordanceStats {
    pub gamma: f64,
    pub tau: f64,
}

#[derive(Debug, Clone)]
pub struct NeighborConsistencyStats {
    pub average: f64,
    pub full: f64,
    pub per_element_average: Vec<f64>,
    pub per_element_full: Vec<f64>,
}

fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean(a, b).sqrt()
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
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

fn centroid(data: &[Vec<f64>], members: &[usize]) -> Vec<f64> {
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

fn build_clusters(
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
) -> (Vec<Cluster>, usize) {
    let mut map: BTreeMap<isize, Vec<usize>> = BTreeMap::new();
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

pub fn squared_errors(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
) -> BasicDistanceStats {
    assert_eq!(data.len(), labels.len());
    if data.is_empty() {
        return BasicDistanceStats {
            mean: 0.0,
            sum_of_squares: 0.0,
            rmsd: 0.0,
        };
    }

    let (clusters, ignored) = build_clusters(labels, noise_label, nh);
    let mut sum = 0.0;
    let mut ssq = 0.0;

    for cluster in clusters {
        if cluster.members.len() <= 1 || cluster.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => continue,
                NoiseHandling::MergeNoise => {}
            }
        }
        let c = centroid(data, &cluster.members);
        for &i in &cluster.members {
            let d = euclidean(&c, &data[i]);
            sum += d;
            ssq += d * d;
        }
    }

    let div = (data.len() - ignored).max(1) as f64;
    BasicDistanceStats {
        mean: sum / div,
        sum_of_squares: ssq,
        rmsd: (ssq / div).sqrt(),
    }
}

fn cluster_centroids(
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
                    NoiseHandling::TreatNoiseAsSingletons => None,
                    NoiseHandling::MergeNoise => Some(centroid(data, &cl.members)),
                }
            } else {
                Some(centroid(data, &cl.members))
            }
        })
        .collect()
}

pub fn simplified_silhouette(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
    penalize: bool,
) -> SilhouetteStats {
    assert_eq!(data.len(), labels.len());
    let n = data.len();
    if n == 0 {
        return SilhouetteStats {
            mean: 0.0,
            stddev: 0.0,
            values: Vec::new(),
        };
    }

    let (clusters, ignored) = build_clusters(labels, noise_label, nh);
    let centroids = cluster_centroids(data, &clusters, nh);
    let mut values = vec![0.0; n];
    let mut included = Vec::new();

    for (i, cluster) in clusters.iter().enumerate() {
        if cluster.members.len() <= 1 {
            continue;
        }
        if cluster.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => continue,
                NoiseHandling::MergeNoise => {}
            }
        }

        let center = centroids[i].as_ref().expect("centroid required");
        for &p in &cluster.members {
            let a = euclidean(center, &data[p]);
            let mut b = f64::INFINITY;

            for (j, other) in clusters.iter().enumerate() {
                if i == j {
                    continue;
                }
                match &centroids[j] {
                    Some(cj) => {
                        b = b.min(euclidean(cj, &data[p]));
                    }
                    None => {
                        if nh == NoiseHandling::IgnoreNoise {
                            continue;
                        }
                        for &q in &other.members {
                            b = b.min(euclidean(&data[p], &data[q]));
                        }
                    }
                }
            }

            if !b.is_finite() {
                b = a;
            }
            let s = if a == 0.0 && b == 0.0 {
                0.0
            } else {
                (b - a) / a.max(b)
            };
            values[p] = s;
            included.push(s);
        }
    }

    let mut penalty = 1.0;
    if penalize && nh == NoiseHandling::IgnoreNoise && ignored > 0 {
        penalty = (n - ignored) as f64 / n as f64;
    }
    let (mean, stddev) = mean_std(&included);
    SilhouetteStats {
        mean: penalty * mean,
        stddev: penalty * stddev,
        values,
    }
}

pub fn silhouette(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
    penalize: bool,
) -> SilhouetteStats {
    assert_eq!(data.len(), labels.len());
    let n = data.len();
    if n == 0 {
        return SilhouetteStats {
            mean: 0.0,
            stddev: 0.0,
            values: Vec::new(),
        };
    }

    let (clusters, ignored) = build_clusters(labels, noise_label, nh);
    let mut values = vec![0.0; n];
    let mut included = Vec::new();

    if clusters.len() <= 1 {
        return SilhouetteStats {
            mean: 0.0,
            stddev: 0.0,
            values,
        };
    }

    for (i, cluster) in clusters.iter().enumerate() {
        if cluster.members.len() <= 1 || cluster.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => continue,
                NoiseHandling::MergeNoise => {}
            }
        }

        for &p in &cluster.members {
            let a = if cluster.members.len() <= 1 {
                0.0
            } else {
                let mut sum = 0.0;
                for &q in &cluster.members {
                    if p != q {
                        sum += euclidean(&data[p], &data[q]);
                    }
                }
                sum / (cluster.members.len() as f64 - 1.0)
            };

            let mut b = f64::INFINITY;
            for (j, other) in clusters.iter().enumerate() {
                if i == j {
                    continue;
                }
                if (other.members.len() <= 1 || other.is_noise) && nh == NoiseHandling::IgnoreNoise
                {
                    continue;
                }
                if (other.members.len() <= 1 || other.is_noise)
                    && nh == NoiseHandling::TreatNoiseAsSingletons
                {
                    for &q in &other.members {
                        b = b.min(euclidean(&data[p], &data[q]));
                    }
                    continue;
                }
                let avg = other
                    .members
                    .iter()
                    .map(|&q| euclidean(&data[p], &data[q]))
                    .sum::<f64>()
                    / other.members.len() as f64;
                b = b.min(avg);
            }

            let s = if b.is_finite() && (a > 0.0 || b > 0.0) {
                (b - a) / a.max(b)
            } else {
                0.0
            };
            values[p] = s;
            included.push(s);
        }
    }

    let mut penalty = 1.0;
    if penalize && nh == NoiseHandling::IgnoreNoise && ignored > 0 {
        penalty = (n - ignored) as f64 / n as f64;
    }
    let (mean, stddev) = mean_std(&included);
    SilhouetteStats {
        mean: penalty * mean,
        stddev: penalty * stddev,
        values,
    }
}

pub fn variance_ratio_criterion(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
    penalize: bool,
) -> f64 {
    assert_eq!(data.len(), labels.len());
    let n = data.len();
    if n <= 1 {
        return 0.0;
    }
    let (clusters, ignored) = build_clusters(labels, noise_label, nh);
    if clusters.len() <= 1 {
        return 0.0;
    }
    let centroids = cluster_centroids(data, &clusters, nh);

    let dim = data[0].len();
    let mut overall = vec![0.0; dim];
    let mut overall_count = 0usize;
    let mut clustercount = 0usize;

    for (i, cl) in clusters.iter().enumerate() {
        if cl.members.len() <= 1 || cl.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => {
                    clustercount += cl.members.len();
                    for &p in &cl.members {
                        for (d, val) in data[p].iter().enumerate() {
                            overall[d] += *val;
                        }
                    }
                    overall_count += cl.members.len();
                    continue;
                }
                NoiseHandling::MergeNoise => {}
            }
        }
        if let Some(c) = &centroids[i] {
            clustercount += 1;
            for d in 0..dim {
                overall[d] += c[d] * cl.members.len() as f64;
            }
            overall_count += cl.members.len();
        }
    }

    if overall_count == 0 || clustercount <= 1 {
        return 0.0;
    }
    for x in &mut overall {
        *x /= overall_count as f64;
    }

    let mut a = 0.0;
    let mut b = 0.0;

    for (i, cl) in clusters.iter().enumerate() {
        if cl.members.len() <= 1 || cl.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => {
                    for &p in &cl.members {
                        b += squared_euclidean(&overall, &data[p]);
                    }
                    continue;
                }
                NoiseHandling::MergeNoise => {}
            }
        }
        let c = centroids[i].as_ref().expect("centroid required");
        for &p in &cl.members {
            a += squared_euclidean(c, &data[p]);
            b += squared_euclidean(&overall, &data[p]);
        }
    }

    if a == 0.0 {
        return 0.0;
    }
    let mut vrc = ((b - a) / a) * ((n as f64 - clustercount as f64) / (clustercount as f64 - 1.0));
    if penalize && nh == NoiseHandling::IgnoreNoise && ignored > 0 {
        vrc *= (n - ignored) as f64 / n as f64;
    }
    vrc
}

pub fn davies_bouldin_index(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
    p: f64,
) -> f64 {
    assert_eq!(data.len(), labels.len());
    let (clusters, _) = build_clusters(labels, noise_label, nh);
    if clusters.is_empty() {
        return 0.0;
    }
    let centroids = cluster_centroids(data, &clusters, nh);

    let mut within = vec![0.0; clusters.len()];
    for (i, cl) in clusters.iter().enumerate() {
        if let Some(c) = &centroids[i] {
            let mut w = 0.0;
            for &m in &cl.members {
                let dist = euclidean(c, &data[m]);
                w += if p != 1.0 { dist.powf(p) } else { dist };
            }
            w /= cl.members.len() as f64;
            within[i] = if p != 1.0 { w.powf(1.0 / p) } else { w };
        }
    }

    let mut vals = Vec::new();
    for i in 0..clusters.len() {
        let mut maxd = 0.0f64;
        for j in 0..clusters.len() {
            if i == j {
                continue;
            }
            match (&centroids[i], &centroids[j]) {
                (Some(ci), Some(cj)) => {
                    let bd = euclidean(ci, cj);
                    if bd > 0.0 {
                        maxd = maxd.max((within[i] + within[j]) / bd);
                    }
                }
                (Some(ci), None) if nh != NoiseHandling::IgnoreNoise => {
                    let mut d = f64::INFINITY;
                    for &m in &clusters[j].members {
                        d = d.min(euclidean(ci, &data[m]));
                    }
                    if d.is_finite() && d > 0.0 {
                        maxd = maxd.max(within[i] / d);
                    }
                }
                (None, Some(cj)) if nh != NoiseHandling::IgnoreNoise => {
                    let mut d = f64::INFINITY;
                    for &m in &clusters[i].members {
                        d = d.min(euclidean(&data[m], cj));
                    }
                    if d.is_finite() && d > 0.0 {
                        maxd = maxd.max(within[j] / d);
                    }
                }
                _ => {}
            }
        }
        vals.push(maxd);
    }

    if vals.len() > 1 {
        vals.iter().sum::<f64>() / vals.len() as f64
    } else {
        2.0
    }
}

pub fn cluster_radius(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
) -> RadiusStats {
    assert_eq!(data.len(), labels.len());
    let (clusters, _) = build_clusters(labels, noise_label, nh);
    if clusters.is_empty() {
        return RadiusStats {
            weighted: 0.0,
            unweighted: 0.0,
        };
    }

    let mut weighted = 0.0;
    let mut unweighted = 0.0;
    let mut cnum = 0usize;

    for cl in &clusters {
        if cl.members.len() <= 1 || cl.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => {}
                NoiseHandling::MergeNoise => {}
            }
        }
        let c = centroid(data, &cl.members);
        let mut maxd = 0.0f64;
        for &m in &cl.members {
            maxd = maxd.max(euclidean(&c, &data[m]));
        }
        cnum += 1;
        weighted += maxd * cl.members.len() as f64;
        unweighted += maxd;
    }

    let n = data.len().max(1) as f64;
    RadiusStats {
        weighted: weighted / n,
        unweighted: if cnum > 0 {
            unweighted / cnum as f64
        } else {
            0.0
        },
    }
}

pub fn c_index(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
) -> f64 {
    assert_eq!(data.len(), labels.len());
    let (clusters, _) = build_clusters(labels, noise_label, nh);

    let mut w = 0usize;
    let mut theta = 0.0;
    for cl in &clusters {
        if (cl.members.len() <= 1 || cl.is_noise) && nh == NoiseHandling::IgnoreNoise {
            continue;
        }
        w += cl.members.len() * cl.members.len().saturating_sub(1) / 2;
        for i in 0..cl.members.len() {
            for j in (i + 1)..cl.members.len() {
                theta += euclidean(&data[cl.members[i]], &data[cl.members[j]]);
            }
        }
    }
    if w == 0 {
        return 1.0;
    }

    let mut considered = Vec::new();
    for cl in &clusters {
        if (cl.members.len() <= 1 || cl.is_noise) && nh == NoiseHandling::IgnoreNoise {
            continue;
        }
        considered.extend_from_slice(&cl.members);
    }
    considered.sort_unstable();
    considered.dedup();

    let mut dists = Vec::new();
    for i in 0..considered.len() {
        for j in (i + 1)..considered.len() {
            dists.push(euclidean(&data[considered[i]], &data[considered[j]]));
        }
    }
    if dists.is_empty() {
        return 1.0;
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let min = dists.iter().take(w).sum::<f64>();
    let max = dists.iter().rev().take(w).sum::<f64>();
    if max > min {
        (theta - min) / (max - min)
    } else {
        1.0
    }
}

fn lower_bound(arr: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if arr[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn upper_bound(arr: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if arr[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

pub fn concordant_pairs_gamma_tau(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
) -> ConcordanceStats {
    assert_eq!(data.len(), labels.len());
    let (clusters, ignored) = build_clusters(labels, noise_label, nh);

    let mut within = Vec::new();
    for cl in &clusters {
        if cl.members.len() <= 1 || cl.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => continue,
                NoiseHandling::MergeNoise => {}
            }
        }
        for i in 0..cl.members.len() {
            for j in (i + 1)..cl.members.len() {
                within.push(euclidean(&data[cl.members[i]], &data[cl.members[j]]));
            }
        }
    }
    within.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let mut concordant_pairs = 0u64;
    let mut discordant_pairs = 0u64;
    let mut between_pairs = 0u64;

    for i in 0..clusters.len() {
        let c1 = &clusters[i];
        if (c1.members.len() <= 1 || c1.is_noise) && nh == NoiseHandling::IgnoreNoise {
            continue;
        }
        for c2 in clusters.iter().skip(i + 1) {
            if (c2.members.len() <= 1 || c2.is_noise) && nh == NoiseHandling::IgnoreNoise {
                continue;
            }
            between_pairs += (c1.members.len() * c2.members.len()) as u64;
            for &p in &c1.members {
                for &q in &c2.members {
                    let d = euclidean(&data[p], &data[q]);
                    let lb = lower_bound(&within, d);
                    let ub = upper_bound(&within, d);
                    concordant_pairs += lb as u64;
                    discordant_pairs += (within.len() - ub) as u64;
                }
            }
        }
    }

    let denom = concordant_pairs + discordant_pairs;
    let mut gamma = if denom > 0 {
        (concordant_pairs as f64 - discordant_pairs as f64) / denom as f64
    } else {
        0.0
    };

    let n_eff = data.len().saturating_sub(ignored) as u64;
    let t = (n_eff * n_eff.saturating_sub(1)) >> 1;
    let m = (t * t.saturating_sub(1)) as f64 / 2.0;
    let wd = within.len() as u64;
    let bd = between_pairs;
    let tie = ((wd * wd.saturating_sub(1)) + (bd * bd.saturating_sub(1))) as f64 / 2.0;
    let mut tau = if m > 0.0 && (m - tie) > 0.0 {
        (concordant_pairs as f64 - discordant_pairs as f64) / ((m - tie) * m).sqrt()
    } else {
        0.0
    };

    if !gamma.is_finite() || gamma < 0.0 {
        gamma = 0.0;
    }
    if !tau.is_finite() || tau < 0.0 {
        tau = 0.0;
    }

    ConcordanceStats { gamma, tau }
}

pub fn pbm_index(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
) -> f64 {
    assert_eq!(data.len(), labels.len());
    if data.is_empty() {
        return 0.0;
    }
    let (clusters, _) = build_clusters(labels, noise_label, nh);
    if clusters.is_empty() {
        return 0.0;
    }
    let centroids = cluster_centroids(data, &clusters, nh);

    let dim = data[0].len();
    let mut overall = vec![0.0; dim];
    for p in data {
        for d in 0..dim {
            overall[d] += p[d];
        }
    }
    for x in &mut overall {
        *x /= data.len() as f64;
    }

    let mut maxd = 0.0f64;
    for i in 0..clusters.len() {
        if centroids[i].is_none() && nh != NoiseHandling::TreatNoiseAsSingletons {
            continue;
        }
        for j in (i + 1)..clusters.len() {
            if centroids[j].is_none() && nh != NoiseHandling::TreatNoiseAsSingletons {
                continue;
            }
            match (&centroids[i], &centroids[j]) {
                (Some(ci), Some(cj)) => maxd = maxd.max(euclidean(ci, cj)),
                (None, Some(cj)) => {
                    for &m in &clusters[i].members {
                        maxd = maxd.max(euclidean(&data[m], cj));
                    }
                }
                (Some(ci), None) => {
                    for &m in &clusters[j].members {
                        maxd = maxd.max(euclidean(ci, &data[m]));
                    }
                }
                (None, None) => {
                    for &a in &clusters[i].members {
                        for &b in &clusters[j].members {
                            maxd = maxd.max(euclidean(&data[a], &data[b]));
                        }
                    }
                }
            }
        }
    }

    let mut a = 0.0;
    let mut b = 0.0;
    let mut n_cl = clusters.len() as f64;

    for (i, cl) in clusters.iter().enumerate() {
        if cl.members.len() <= 1 || cl.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => {
                    n_cl -= 1.0;
                    continue;
                }
                NoiseHandling::TreatNoiseAsSingletons => {
                    for &m in &cl.members {
                        b += euclidean(&overall, &data[m]);
                    }
                    n_cl += cl.members.len() as f64 - 1.0;
                    continue;
                }
                NoiseHandling::MergeNoise => {}
            }
        }
        if let Some(c) = &centroids[i] {
            for &m in &cl.members {
                a += euclidean(c, &data[m]);
                b += euclidean(&overall, &data[m]);
            }
        }
    }

    if a <= 0.0 || n_cl <= 0.0 {
        0.0
    } else {
        ((1.0 / n_cl) * (b / a) * maxd).powi(2)
    }
}

fn prim_mst_dense(matrix: &[Vec<f64>]) -> Vec<(usize, usize)> {
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

pub fn dbcv(data: &[Vec<f64>], labels: &[isize], noise_label: Option<isize>) -> f64 {
    assert_eq!(data.len(), labels.len());
    if data.is_empty() {
        return 0.0;
    }

    let (clusters, _) = build_clusters(labels, noise_label, NoiseHandling::MergeNoise);
    let dim = data[0].len() as f64;
    let numc = clusters.len();

    let mut core_dists: Vec<Option<Vec<f64>>> = vec![None; numc];
    for (c, cl) in clusters.iter().enumerate() {
        if cl.is_noise || cl.members.len() < 2 {
            continue;
        }
        let mut d = vec![0.0; cl.members.len()];
        for (i, &p) in cl.members.iter().enumerate() {
            let mut sum = 0.0;
            let mut neighbors = 0usize;
            for &q in &cl.members {
                if p == q {
                    continue;
                }
                let dist = euclidean(&data[p], &data[q]);
                if dist > 0.0 {
                    sum += (1.0 / dist).powf(dim);
                    neighbors += 1;
                }
            }
            d[i] = if neighbors > 0 {
                (sum / neighbors as f64).powf(-1.0 / dim)
            } else {
                f64::INFINITY
            };
        }
        core_dists[c] = Some(d);
    }

    let mut cluster_degrees: Vec<Option<Vec<usize>>> = vec![None; numc];
    let mut cluster_dsc_max = vec![f64::NAN; numc];
    let mut internal_edges = vec![false; numc];

    for c in 0..numc {
        let cl = &clusters[c];
        if cl.is_noise || cl.members.len() < 2 {
            continue;
        }
        let core = core_dists[c].as_ref().expect("core distances present");
        let mut matrix = vec![vec![0.0; cl.members.len()]; cl.members.len()];

        for i in 0..cl.members.len() {
            for j in (i + 1)..cl.members.len() {
                let mr = core[i]
                    .max(core[j])
                    .max(euclidean(&data[cl.members[i]], &data[cl.members[j]]));
                matrix[i][j] = mr;
                matrix[j][i] = mr;
            }
        }

        let edges = prim_mst_dense(&matrix);
        let mut deg = vec![0usize; cl.members.len()];
        for &(a, b) in &edges {
            deg[a] += 1;
            deg[b] += 1;
        }
        for &(a, b) in &edges {
            if deg[a] > 1 && deg[b] > 1 {
                internal_edges[c] = true;
            }
        }

        let mut dsc = 0.0f64;
        for &(a, b) in &edges {
            if !internal_edges[c] || (deg[a] > 1 && deg[b] > 1) {
                dsc = dsc.max(matrix[a][b]);
            }
        }
        cluster_degrees[c] = Some(deg);
        cluster_dsc_max[c] = dsc;
    }

    let mut dbcv_sum = 0.0;
    for c in 0..numc {
        let cl = &clusters[c];
        if cl.is_noise || cl.members.len() < 2 {
            continue;
        }
        let current_dsc = cluster_dsc_max[c];
        let core = core_dists[c].as_ref().expect("core distances present");
        let deg = cluster_degrees[c].as_ref().expect("degrees present");

        let mut dspc_min = f64::INFINITY;
        for (i, &p) in cl.members.iter().enumerate() {
            if deg[i] < 2 && cl.members.len() > 2 {
                continue;
            }
            for oc in 0..numc {
                if oc == c {
                    continue;
                }
                let ocl = &clusters[oc];
                if ocl.is_noise || ocl.members.len() < 2 {
                    continue;
                }
                let ocore = core_dists[oc].as_ref().expect("core distances present");
                let odeg = cluster_degrees[oc].as_ref().expect("degrees present");

                for (j, &q) in ocl.members.iter().enumerate() {
                    if odeg[j] < 2 && cl.members.len() > 2 {
                        continue;
                    }
                    let mr = core[i].max(ocore[j]).max(euclidean(&data[p], &data[q]));
                    dspc_min = dspc_min.min(mr);
                }
            }
        }

        let vc = (dspc_min - current_dsc) / dspc_min.max(current_dsc);
        dbcv_sum += cl.members.len() as f64 / data.len() as f64 * vc;
    }

    dbcv_sum
}

pub fn neighbor_consistency_knn(
    data: &[Vec<f64>],
    labels: &[isize],
    k: usize,
) -> NeighborConsistencyStats {
    assert_eq!(data.len(), labels.len());
    let n = data.len();
    if n == 0 || k == 0 {
        return NeighborConsistencyStats {
            average: 0.0,
            full: 0.0,
            per_element_average: vec![0.0; n],
            per_element_full: vec![0.0; n],
        };
    }

    let mut per_avg = vec![0.0; n];
    let mut per_full = vec![0.0; n];

    for i in 0..n {
        let mut ds = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i != j {
                ds.push((euclidean(&data[i], &data[j]), j));
            }
        }
        ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let kk = k.min(ds.len());

        let mut same = 0usize;
        for (_, j) in ds.iter().take(kk) {
            if labels[*j] == labels[i] {
                same += 1;
            }
        }

        let frac = same as f64 / kk.max(1) as f64;
        per_avg[i] = frac;
        per_full[i] = if same == kk { 1.0 } else { 0.0 };
    }

    NeighborConsistencyStats {
        average: per_avg.iter().sum::<f64>() / n as f64,
        full: per_full.iter().sum::<f64>() / n as f64,
        per_element_average: per_avg,
        per_element_full: per_full,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silhouette_and_sse_compile_and_return_ranges() {
        let x = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.1],
            vec![3.0, 3.0],
            vec![3.1, 3.0],
        ];
        let labels = vec![0, 0, 1, 1];

        let s = silhouette(&x, &labels, None, NoiseHandling::MergeNoise, true);
        assert!(s.mean <= 1.0 && s.mean >= -1.0);

        let ssq = squared_errors(&x, &labels, None, NoiseHandling::MergeNoise);
        assert!(ssq.sum_of_squares >= 0.0);

        let db = davies_bouldin_index(&x, &labels, None, NoiseHandling::MergeNoise, 1.0);
        assert!(db >= 0.0);

        let nc = neighbor_consistency_knn(&x, &labels, 1);
        assert!(nc.average >= 0.0 && nc.average <= 1.0);
        assert!(nc.full >= 0.0 && nc.full <= 1.0);
    }

    #[test]
    fn dbcv_runs() {
        let x = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.1],
            vec![3.0, 3.0],
            vec![3.1, 3.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let val = dbcv(&x, &labels, None);
        assert!(val.is_finite());
    }
}
