use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::intrinsicdimensionality::DistanceIDEstimator;

const ZERO_PADDING: usize = 100;

pub fn make_intrinsic_subspace_data(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let uniform_large = rand::distributions::Uniform::new(-1.0, 1.0);
    let uniform_small = rand::distributions::Uniform::new(-0.05, 0.05);

    let mut data = Vec::with_capacity(n);
    data.push(vec![0.0; 10]);
    data.push(vec![0.0; 10]); // deliberate duplicate, to test handling of distance zero
    for _ in 2..n {
        let mut pt = vec![0.0; 10];
        for (i, value) in pt.iter_mut().enumerate() {
            *value = if i < 5 { rng.sample(uniform_large) } else { rng.sample(uniform_small) };
        }
        data.push(pt);
    }
    data
}

pub fn regression_test<E>(dim: usize, size: usize, seed: u64, expected: f64)
where
    E: DistanceIDEstimator,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(size + ZERO_PADDING);
    for _ in 0..size {
        let r = rng.r#gen::<f64>();
        data.push(r.powf(1.0 / (dim as f64)));
    }
    data.extend(std::iter::repeat_n(0.0, ZERO_PADDING));
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let nozeros = &data[ZERO_PADDING..];
    let full = &data[..];

    let e1 = E::estimate_from_distances(nozeros);
    let e2 = E::estimate_from_distances(full);

    assert!(e1.is_finite(), "estimate should be finite");
    assert!(e2.is_finite(), "estimate should be finite");
    assert!((e1 - expected).abs() < 0.5, "e1 {} vs expected {}", e1, expected);
    assert!((e2 - expected).abs() < 0.5, "e2 {} vs expected {}", e2, expected);
}

pub fn test_zeros<E>()
where
    E: DistanceIDEstimator,
{
    let _ = E::estimate_from_distances(&[0.0, 0.0, 0.0, 0.0]);
    let _ = E::estimate_from_distances(&[0.0, 0.0, 0.0, 1.0]);
    let _ = E::estimate_from_distances(&[0.0, 0.0, 0.0, 1.0, 2.0]);
}
