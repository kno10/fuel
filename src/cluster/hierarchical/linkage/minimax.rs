use crate::cluster::hierarchical::SetLinkage;
use crate::{DistanceData, Float};

pub struct MinimaxLinkage;
impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, ()> for MinimaxLinkage {
    fn summarize(_data: &D, _members: &[usize]) {}

    fn cluster_distance(
        data: &D, _summary_a: &(), _summary_b: &(), a: &[usize], b: &[usize],
    ) -> (F, Option<usize>) {
        let (d, proto) = minimax_candidate::<D, F>(data, a, b);
        (d, Some(proto))
    }
}

fn minimax_candidate<D: DistanceData<F>, F: Float>(
    data: &D, cx: &[usize], cy: &[usize],
) -> (F, usize) {
    let mut best_dist = F::infinity();
    let mut best_proto = cx[0];

    for &cand in cx.iter().chain(cy.iter()) {
        let mut max_dist = F::zero();
        for &p in cx {
            let d = F::from(data.distance(cand, p)).unwrap();
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
            let d = F::from(data.distance(cand, p)).unwrap();
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
