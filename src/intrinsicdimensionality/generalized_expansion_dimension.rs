use crate::intrinsicdimensionality::DistanceIDEstimator;

/// Generalized Expansion Dimension estimator.
///
/// Reference:
///
/// M. E. Houle, H. Kashima, M. Nett
/// Generalized expansion dimension
/// 12th International Conference on Data Mining Workshops (ICDMW)
///
/// Uses median-of-ratios over the log-distance scale:
/// \(a_{k,i} = \frac{\ln(1+k) - \ln(1+i)}{\ln d_k - \ln d_i}\),
/// then \(m = \mathrm{median}_k(\mathrm{median}_{i>k} a_{k,i})\).
///
/// Returns `NaN` on insufficient data (fewer than 2 valid distances).
pub fn generalized_expansion_dimension(distances: &[f64]) -> f64 {
    let begin = crate::intrinsicdimensionality::find_begin(distances);
    let k = distances.len() - begin;
    if k < 2 {
        return f64::NAN;
    }

    let last = k - 1;
    let mut meds = Vec::with_capacity(last);

    for kk in 0..last {
        let logdk = distances[begin + kk].ln();
        let log1pk = (kk as f64).ln_1p();

        let mut values = (kk + 1..=last)
            .filter_map(|i| {
                let logdi = distances[begin + i].ln();
                if (logdk - logdi).abs() > 0.0 {
                    let log1pi = (i as f64).ln_1p();
                    Some((log1pk - log1pi) / (logdk - logdi))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        meds.push(if values.is_empty() { f64::NAN } else { median(&mut values) });
    }

    median(&mut meds)
}

pub struct GeneralizedExpansionDimension;

fn median(data: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return f64::NAN;
    }

    let mid = n / 2;
    data.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        data[mid]
    } else {
        0.5 * (data[mid] + data[..mid - 1].iter().copied().fold(data[mid - 1], |max, x| max.max(x)))
    }
}

impl DistanceIDEstimator for GeneralizedExpansionDimension {
    fn estimate_from_distances(distances: &[f64]) -> f64 {
        generalized_expansion_dimension(distances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::KNNIDEstimator;
    use crate::intrinsicdimensionality::test::{
        make_intrinsic_subspace_data, regression_test, test_zeros,
    };

    #[test]
    fn generalized_expansion_dimension_regression() {
        regression_test::<GeneralizedExpansionDimension>(5, 1000, 0, 4.895086664189283);
        regression_test::<GeneralizedExpansionDimension>(7, 1000, 0, 6.853121329865002);
    }

    #[test]
    fn generalized_expansion_dimension_zeros() { test_zeros::<GeneralizedExpansionDimension>(); }

    #[test]
    fn generalized_expansion_dimension_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(10000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = GeneralizedExpansionDimension::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 5.294440321159763;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "ged estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
