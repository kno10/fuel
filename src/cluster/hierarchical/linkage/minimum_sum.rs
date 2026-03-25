use crate::cluster::hierarchical::SetLinkage;
use crate::{DistanceData, Float};

/// Linkage criterion that chooses the candidate minimizing the total
/// distance to all points in the two clusters (the usual "minimum-sum").
///
/// The returned prototype is the index of the chosen medoid.
pub struct MinimumSumLinkage;
impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, ()> for MinimumSumLinkage {
    fn summarize(_data: &D, _members: &[usize]) {}

    fn cluster_distance(
        data: &D, _summary_a: &(), _summary_b: &(), a: &[usize], b: &[usize],
    ) -> (F, Option<usize>) {
        let (d, proto) = minimum_sum_candidate(data, a, b);
        (F::from(d).unwrap(), Some(proto))
    }
}

fn minimum_sum_candidate<D: DistanceData<F>, F: Float>(
    data: &D, cx: &[usize], cy: &[usize],
) -> (f64, usize) {
    let mut best_sum = f64::INFINITY;
    let mut best_proto = cx[0];

    for &cand in cx.iter().chain(cy.iter()) {
        let mut sum = 0.0;
        for &p in cx {
            let d: f64 = data.distance(cand, p).to_f64().unwrap();
            sum += d;
            if sum >= best_sum {
                break;
            }
        }
        if sum >= best_sum {
            continue;
        }
        for &p in cy {
            let d: f64 = data.distance(cand, p).to_f64().unwrap();
            sum += d;
            if sum >= best_sum {
                break;
            }
        }
        if sum < best_sum {
            best_sum = sum;
            best_proto = cand;
        }
    }

    (best_sum, best_proto)
}

#[cfg(test)]
mod tests {
    use super::MinimumSumLinkage;
    use crate::TableWithDistance;
    use crate::cluster::hierarchical::SetLinkage;
    use crate::cluster::hierarchical::test_utils::ScalarDistance; // bring trait into scope

    #[test]
    fn minsum_basic() {
        let points = [vec![0.0], vec![1.0], vec![3.0]];
        let data = TableWithDistance::with_distance(&points, ScalarDistance);
        // summaries are unit values for this linkage.
        let (d, proto) = MinimumSumLinkage::cluster_distance(&data, &(), &(), &[0, 1], &[2]);
        assert!(proto.is_some());
        assert!(d >= 0.0);
    }
}
