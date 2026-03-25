use crate::intrinsicdimensionality::{
    DistanceBasedIntrinsicDimensionalityEstimator, KnnBasedIntrinsicDimensionalityEstimator,
};

pub fn generalized_expansion_dimension(distances: &[f64]) -> f64 {
    let n = distances.len();
    let mut begin = 0;
    while begin < n && distances[begin] <= 0.0 {
        begin += 1;
    }
    let k = n - begin;
    if k < 2 {
        return f64::NAN;
    }

    let last = k - 1;
    if last == 0 {
        return f64::NAN;
    }

    let mut meds = vec![0.0; last];

    for kk in 0..last {
        let logdk = distances[begin + kk].ln();
        let log1pk = (kk as f64).ln_1p();
        let mut values = Vec::with_capacity(last - kk);
        for i in (kk + 1)..=last {
            let logdi = distances[begin + i].ln();
            if (logdk - logdi).abs() > 0.0 {
                let log1pi = (i as f64).ln_1p();
                values.push((log1pk - log1pi) / (logdk - logdi));
            }
        }
        meds[kk] = if values.is_empty() { f64::NAN } else { median(&mut values) };
    }

    median(&mut meds)
}

pub struct GeneralizedExpansionDimension;

fn median(data: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return f64::NAN;
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = n / 2;
    if n % 2 == 1 { data[mid] } else { 0.5 * (data[mid - 1] + data[mid]) }
}

impl DistanceBasedIntrinsicDimensionalityEstimator for GeneralizedExpansionDimension {
    fn estimate_from_distances(distances: &[f64]) -> f64 {
        generalized_expansion_dimension(distances)
    }
}

pub fn generalized_expansion_dimension_from_knn<'a, S, D, F>(
    tree: &S, data: &'a D, query_idx: usize, k: usize,
) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    GeneralizedExpansionDimension::estimate_from_knn(tree, data, query_idx, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::{
        KnnBasedIntrinsicDimensionalityEstimator, make_hypersphere_embedded_data, regression_test,
        test_zeros,
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
        let data = make_hypersphere_embedded_data(10000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = GeneralizedExpansionDimension::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.845045991014738;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "ged estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
