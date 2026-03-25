#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use super::helpers::{NoiseHandling, build_clusters, euc, prim_mst_dense};

/// Density-based cluster validity (DBCV) index.
#[must_use]
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
                let dist = euc(&data[p], &data[q]);
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
                let mr = core[i].max(core[j]).max(euc(&data[cl.members[i]], &data[cl.members[j]]));
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
                    let mr = core[i].max(ocore[j]).max(euc(&data[p], &data[q]));
                    dspc_min = dspc_min.min(mr);
                }
            }
        }

        let vc = (dspc_min - current_dsc) / dspc_min.max(current_dsc);
        dbcv_sum += cl.members.len() as f64 / data.len() as f64 * vc;
    }

    dbcv_sum
}
