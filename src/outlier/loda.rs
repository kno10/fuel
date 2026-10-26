use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::outlier::common::OutlierResult;
use crate::{Float, VectorData};

#[derive(Clone, Debug)]
struct Histogram<F> {
    min: F,
    max: F,
    densities: Vec<F>,
}

impl<F: Float> Histogram<F> {
    fn fit(values: &[F], requested_bins: Option<usize>) -> Self {
        let min = values.iter().copied().fold(F::infinity(), F::min);
        let max = values.iter().copied().fold(F::neg_infinity(), F::max);
        let span = max - min;
        if !span.is_finite() || span <= F::zero() {
            return Self { min, max, densities: vec![F::one()] };
        }

        // The reference implementation evaluates D=2..floor(n/log(n)).
        let n = values.len();
        let dmax = if n > 1 {
            ((n as f64) / (n as f64).ln()).floor() as usize
        } else {
            1
        };
        let mut best_bins = requested_bins.unwrap_or(1usize);
        let mut best_criterion = f64::NEG_INFINITY;
        if requested_bins.is_none() && dmax >= 2 {
            for bins in 2..=dmax {
                let mut counts = vec![0usize; bins];
                for &x in values {
                    let t = ((x - min) / span).to_f64().unwrap_or(0.0).clamp(0.0, 1.0);
                    let bin = ((t * bins as f64) as usize).min(bins - 1);
                    counts[bin] += 1;
                }
                let likelihood: f64 = counts
                    .iter()
                    .filter(|&&count| count > 0)
                    .map(|&count| {
                        let p = count as f64 * bins as f64 / n as f64;
                        count as f64 * p.ln()
                    })
                    .sum();
                let d = bins as f64;
                let criterion = likelihood - (d - 1.0) - d.ln().powf(2.5);
                if criterion > best_criterion {
                    best_criterion = criterion;
                    best_bins = bins;
                }
            }
        }

        let width = span / F::from_usize(best_bins).unwrap_or(F::one());
        let mut counts = vec![0usize; best_bins];
        for &x in values {
            let bin = (((x - min) / width).to_usize().unwrap_or(0)).min(best_bins - 1);
            counts[bin] += 1;
        }
        let n_f = F::from_usize(n).unwrap_or(F::one());
        let densities = counts
            .into_iter()
            .map(|count| F::from_usize(count).unwrap_or(F::zero()) / (n_f * width))
            .collect();
        Self { min, max, densities }
    }

    fn negative_log_density(&self, x: F) -> F {
        let span = self.max - self.min;
        let bin = if span > F::zero() {
            (((x - self.min) / span * F::from_usize(self.densities.len()).unwrap_or(F::one()))
                .floor()
                .to_usize()
                .unwrap_or(0))
                .min(self.densities.len() - 1)
        } else {
            0
        };
        let epsilon = F::epsilon() * F::from_f64(10.0).unwrap_or(F::one());
        -(self.densities[bin] + epsilon).ln()
    }
}

/// LODA (Lightweight On-line Detector of Anomalies).
///
/// Builds random one-dimensional projections, selects a regular histogram
/// for each projection, and averages the negative log densities. Larger
/// scores indicate greater anomaly likelihood. `tau` controls the stopping
/// criterion and `seed` makes the random projections reproducible. Set
/// `n_bins` to use a fixed number of bins instead of automatic selection. Set
/// `n_projections` to use a fixed number of projections instead of stopping by
/// convergence.
pub fn loda<D, F>(
    data: &D, tau: F, n_bins: Option<usize>, n_projections: Option<usize>, seed: u64,
) -> Result<OutlierResult<F>, String>
where
    D: VectorData<F>,
    F: Float,
{
    let n = data.len();
    let dims = data.dims();
    if n == 0 || dims == 0 {
        return Ok(OutlierResult::builder(Vec::new(), "LODA").build());
    }
    if !tau.is_finite() || tau < F::zero() {
        return Err("LODA tau must be finite and non-negative".into());
    }
    if n_bins == Some(0) {
        return Err("LODA n_bins must be positive".into());
    }
    if n_projections == Some(0) {
        return Err("LODA n_projections must be positive".into());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut projections: Vec<Vec<F>> = Vec::new();
    let mut histograms: Vec<Histogram<F>> = Vec::new();
    let mut cumulative_scores = vec![F::zero(); n];
    let mut previous_mean: Option<Vec<F>> = None;
    let mut first_change = F::zero();
    loop {
        crate::poll_interrupted()?;
        let mut projection: Vec<F> = (0..dims)
            .map(|_| {
                let sample: f64 = StandardNormal.sample(&mut rng);
                F::from_f64(sample).unwrap_or(F::zero())
            })
            .collect();
        let norm = projection.iter().map(|x| *x * *x).sum::<F>().sqrt();
        if norm > F::zero() {
            for x in &mut projection { *x /= norm; }
        } else {
            projection[0] = F::one();
        }

        let projected: Vec<F> = (0..n)
            .map(|i| F::vec_dot(data.point(i), &projection, dims))
            .collect();
        let histogram = Histogram::fit(&projected, n_bins);
        let k = projections.len() + 1;
        for (i, &x) in projected.iter().enumerate() {
            cumulative_scores[i] += histogram.negative_log_density(x);
        }
        projections.push(projection);
        histograms.push(histogram);

        if n_projections == Some(k) {
            break;
        }

        let divisor = F::from_usize(k).unwrap_or(F::one());
        let current_mean: Vec<F> = cumulative_scores.iter().map(|x| *x / divisor).collect();
        if let Some(previous) = previous_mean.as_ref() {
            let change = current_mean.iter().zip(previous)
                .map(|(a, b)| (*a - *b).abs()).sum::<F>();
            if k == 2 {
                first_change = change;
            } else if first_change > F::zero() && change / first_change < tau {
                break;
            }
        }
        previous_mean = Some(current_mean);
        if projections.len() >= 1000 { break; }
    }

    let scores: Vec<F> = cumulative_scores
        .into_iter()
        .map(|score| score / F::from_usize(projections.len()).unwrap_or(F::one()))
        .collect();
    Ok(OutlierResult::builder(scores, "LODA").build())
}
