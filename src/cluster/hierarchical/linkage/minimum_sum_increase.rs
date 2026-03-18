use crate::DistanceData;
use crate::cluster::hierarchical::SetLinkage;
use num_traits::Float;

pub(crate) struct MinimumSumIncreaseSummary<F: Float> {
    total_distance: F,
}

/// Variant of minimum-sum linkage used by HACAM that later applies an
/// additional "total distance" correction during the merge algorithm.
///
/// The raw distance/prototype computation is identical to
/// [`MinimumSumLinkage`]; the adjustment is handled through `adjust_distance`
/// and by maintaining a running total in the summary.
pub struct MinimumSumIncreaseLinkage;
impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, MinimumSumIncreaseSummary<F>>
    for MinimumSumIncreaseLinkage
{
    fn summarize(_data: &D, _members: &[usize]) -> MinimumSumIncreaseSummary<F> {
        MinimumSumIncreaseSummary {
            total_distance: F::zero(),
        }
    }

    fn cluster_distance(
        data: &D,
        _summary_a: &MinimumSumIncreaseSummary<F>,
        _summary_b: &MinimumSumIncreaseSummary<F>,
        a: &[usize],
        b: &[usize],
    ) -> (F, Option<usize>) {
        let (d, proto) = minimum_sum_candidate::<D, F>(data, a, b);
        (F::from(d).unwrap(), Some(proto))
    }

    fn adjust_distance(
        d: F,
        summary_a: &MinimumSumIncreaseSummary<F>,
        summary_b: &MinimumSumIncreaseSummary<F>,
    ) -> F {
        d - summary_a.total_distance - summary_b.total_distance
    }

    fn merge_summary(
        dest: &mut MinimumSumIncreaseSummary<F>,
        source: MinimumSumIncreaseSummary<F>,
        _prototype: Option<usize>,
        distance: F,
    ) {
        dest.total_distance = dest.total_distance + distance + source.total_distance;
    }
}

// reuse identical candidate logic
fn minimum_sum_candidate<D: DistanceData<F>, F: Float>(
    data: &D,
    cx: &[usize],
    cy: &[usize],
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
