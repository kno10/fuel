use crate::cluster::hierarchical::{SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Minimax linkage selects a prototype minimizing the maximum distance to
/// all points in the merged cluster.
///
/// The merged distance is
/// $\min_{z\in X\cup Y}\max(\max_{x\in X} d(z,x),\ \max_{y\in Y} d(z,y))$.
/// This method is only implemented as a set-based linkage.
pub struct MinimaxLinkage;
impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, idsize> for MinimaxLinkage {
    fn summarize(data: &D, members: &[idsize]) -> idsize {
        if members.len() == 1 {
            members[0]
        } else {
            let (_, proto) = minimax_candidate(data, members, &[]);
            proto
        }
    }

    fn cluster_distance(
        data: &D, _summary_a: &idsize, _summary_b: &idsize, a: &[idsize], b: &[idsize],
    ) -> (F, idsize) {
        minimax_candidate::<D, F>(data, a, b)
    }

    fn merged_prototype(summary: &idsize) -> usize { *summary as usize }
}

fn minimax_candidate<D: DistanceData<F>, F: Float>(
    data: &D, cx: &[idsize], cy: &[idsize],
) -> (F, idsize) {
    let mut best_dist = F::infinity();
    let mut best_proto = cx[0];

    for &cand in cx.iter().chain(cy.iter()) {
        let mut max_dist = F::zero();
        for &p in cx {
            let d = F::from(data.distance(cand as usize, p as usize)).unwrap();
            if d > max_dist {
                max_dist = d;
                if max_dist >= best_dist {
                    break;
                }
            }
        }
        if max_dist >= best_dist {
            continue;
        }
        for &p in cy {
            let d = F::from(data.distance(cand as usize, p as usize)).unwrap();
            if d > max_dist {
                max_dist = d;
                if max_dist >= best_dist {
                    break;
                }
            }
        }
        if max_dist < best_dist {
            best_dist = max_dist;
            best_proto = cand;
        }
    }

    (best_dist, best_proto)
}
