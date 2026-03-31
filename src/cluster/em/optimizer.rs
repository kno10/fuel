use crate::math::{DefaultMath, Math};
use crate::{Float, VectorData as Dataset};
use ndarray::Array2;
use std::iter::Sum; // FIXME: no longer necessary here if we use crate::Float?
use std::ops::{AddAssign, MulAssign, SubAssign}; // FIXME: no longer necessary here if we use crate::Float?

/// Pluggable cluster model for expectation-maximization.
pub trait EmModel<N>
where
    N: Float + Copy,
{
    /// Begin accumulation for a new M-step.
    fn begin_estep(&mut self);

    /// Whether this model needs a first pass before the regular update pass.
    fn needs_two_pass(&self) -> bool {
        false
    }

    /// Optional first-pass update.
    fn first_pass_estep(&mut self, _x: &[N], _responsibility: N) {}

    /// Finalize first pass.
    fn finalize_first_pass_estep(&mut self) {}

    /// Accumulate one data point responsibility.
    fn update_estep(&mut self, x: &[N], responsibility: N);

    /// Finalize the M-step for this model.
    fn finalize_estep(&mut self, weight: N, prior: N);

    /// Estimate log density including the component weight.
    fn estimate_log_density(&self, x: &[N]) -> N;

    /// Current mixture weight.
    fn weight(&self) -> N;

    /// Set mixture weight.
    fn set_weight(&mut self, weight: N);
}

/// Factory for creating initial EM models.
/// EM configuration.
#[derive(Clone, Copy, Debug)]
pub struct EmConfig<N>
where
    N: Float + Copy,
{
    /// Convergence threshold on mean log-likelihood.
    pub delta: N,
    /// Minimum number of iterations.
    pub miniter: usize,
    /// Maximum number of iterations.
    pub maxiter: usize,
    /// Use hard assignments during M-step.
    pub hard: bool,
    /// MAP prior (0 for MLE).
    pub prior: N,
    /// Return soft assignments.
    pub return_soft: bool,
    /// Floor to avoid `-inf` likelihood values.
    pub min_log_likelihood: N,
    /// Expected share of noise in the data (0 disables noise handling).
    pub noise_ratio: N,
}

impl<N> Default for EmConfig<N>
where
    N: Float + Copy,
{
    fn default() -> Self {
        Self {
            delta: N::from(1e-7).unwrap(),
            miniter: 1,
            maxiter: 100,
            hard: false,
            prior: N::zero(),
            return_soft: false,
            min_log_likelihood: N::from(-100_000.0).unwrap(),
            noise_ratio: N::zero(),
        }
    }
}

/// EM result container.
#[derive(Debug)]
pub struct EmResult<N, Mo>
where
    N: Float + Copy,
    Mo: EmModel<N>,
{
    pub models: Vec<Mo>,
    pub assignments: Vec<usize>,
    pub responsibilities: Option<Array2<N>>,
    pub n_iter: usize,
    pub log_likelihood: N,
}

// helper to pick index of maximum value in a slice.  There isn't a
// dedicated `argmax` in the standard library, but the iterator adapters
// can express it succinctly.  `partial_cmp` is used because `Float`
// implements `PartialOrd` rather than `Ord`.
fn argmax<N>(vals: &[N]) -> usize
where
    N: Float + Copy,
{
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Numerically stable `log(sum(exp(x)))`.
pub fn log_sum_exp<N>(x: &[N]) -> N
where
    N: Float + Copy,
{
    let mut max = x[0];
    for &v in x.iter().skip(1) {
        if v > max {
            max = v;
        }
    }
    let cutoff = max - N::from(35.350_506_209).unwrap(); // ln(2^51)
    let mut acc = N::zero();
    for &v in x {
        if v > cutoff {
            acc = acc + if v < max { (v - max).exp() } else { N::one() };
        }
    }
    if acc > N::one() { max + acc.ln() } else { max }
}

fn assign_probabilities_to_instances<N, A, Mo>(
    data: &A,
    models: &[Mo],
    probs: &mut [N],
    scratch: &mut [N],
    min_log_likelihood: N,
    noise_log_density: Option<N>,
    mut noise_probs: Option<&mut [N]>,
) -> (N, Option<N>)
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    A: Dataset<N>,
    Mo: EmModel<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let k = models.len();
    let mut em_sum = N::zero();
    let local_len = k + if noise_log_density.is_some() { 1 } else { 0 };
    let mut local = vec![N::zero(); local_len];
    let mut noise_sum = N::zero();

    for i in 0..n {
        data.load_into(i, scratch, d);
        for j in 0..k {
            let v = models[j].estimate_log_density(scratch);
            local[j] = if v > min_log_likelihood {
                v
            } else {
                min_log_likelihood
            };
        }
        if let Some(log_noise) = noise_log_density {
            local[k] = log_noise.max(min_log_likelihood);
        }
        let logp = log_sum_exp(&local);
        for j in 0..k {
            probs[i * k + j] = (local[j] - logp).exp();
        }
        if noise_log_density.is_some() {
            let noise_prob = (local[k] - logp).exp();
            if let Some(noise_vec) = noise_probs.as_mut() {
                (*noise_vec)[i] = noise_prob;
            }
            noise_sum += noise_prob;
        }
        em_sum += logp;
    }

    (
        em_sum / N::from(n).unwrap(),
        if noise_log_density.is_some() {
            Some(noise_sum / N::from(n).unwrap())
        } else {
            None
        },
    )
}

fn recompute_models_soft<N, A, Mo>(
    data: &A,
    probs: &[N],
    models: &mut [Mo],
    prior: N,
    scratch: &mut [N],
) where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    A: Dataset<N>,
    Mo: EmModel<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let k = models.len();

    let mut needs_two_pass = false;
    for model in models.iter_mut() {
        model.begin_estep();
        needs_two_pass |= model.needs_two_pass();
    }

    if needs_two_pass {
        for i in 0..n {
            data.load_into(i, scratch, d);
            for j in 0..k {
                let p = probs[i * k + j];
                if p > N::from(1e-10).unwrap() {
                    models[j].first_pass_estep(scratch, p);
                }
            }
        }
        for model in models.iter_mut() {
            model.finalize_first_pass_estep();
        }
    }

    // accumulate weights; use math backend to clear the buffer
    let mut wsum = vec![N::zero(); k];
    DefaultMath::<N>::fill(&mut wsum, N::zero(), k);
    for i in 0..n {
        data.load_into(i, scratch, d);
        for j in 0..k {
            let p = probs[i * k + j];
            models[j].update_estep(scratch, p);
            wsum[j] = wsum[j] + p;
        }
    }

    let nf = N::from(n).unwrap();
    let kf = N::from(k).unwrap();
    for j in 0..k {
        let weight = if prior <= N::zero() {
            wsum[j] / nf
        } else {
            (wsum[j] + prior - N::one()) / (nf + prior * kf - kf)
        };
        models[j].finalize_estep(weight, prior);
    }
}

fn recompute_models_hard<N, A, Mo>(
    data: &A,
    probs: &[N],
    noise_probs: Option<&[N]>,
    models: &mut [Mo],
    prior: N,
    scratch: &mut [N],
) where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    A: Dataset<N>,
    Mo: EmModel<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let k = models.len();

    let mut needs_two_pass = false;
    for model in models.iter_mut() {
        model.begin_estep();
        needs_two_pass |= model.needs_two_pass();
    }

    if needs_two_pass {
        for i in 0..n {
            data.load_into(i, scratch, d);
            let slice = &probs[i * k..(i + 1) * k];
            let best = argmax(slice);
            if let Some(noise) = noise_probs
                && noise[i] > slice[best]
            {
                continue;
            }
            models[best].first_pass_estep(scratch, N::one());
        }
        for model in models.iter_mut() {
            model.finalize_first_pass_estep();
        }
    }

    // accumulate weights; we could reuse an existing buffer and clear
    // it with the math kernel's `fill` helper, demonstrating how EM
    // algorithms can lean on `Math` for basic vector operations.
    // e.g. Math::<N>::fill(&mut wsum, N::zero(), k);
    let mut wsum = vec![N::zero(); k];
    DefaultMath::<N>::fill(&mut wsum, N::zero(), k);
    for i in 0..n {
        data.load_into(i, scratch, d);
        let slice = &probs[i * k..(i + 1) * k];
        let best = argmax(slice);
        if let Some(noise) = noise_probs
            && noise[i] > slice[best]
        {
            continue;
        }
        models[best].update_estep(scratch, N::one());
        wsum[best] = wsum[best] + N::one();
    }

    let nf = N::from(n).unwrap();
    let kf = N::from(k).unwrap();
    for j in 0..k {
        let weight = if prior <= N::zero() {
            wsum[j] / nf
        } else {
            (wsum[j] + prior - N::one()) / (nf + prior * kf - kf)
        };
        models[j].finalize_estep(weight, prior);
    }
}

/// Generic EM solver for mixture models.
pub fn expectation_maximization<N, A, Mo>(
    data: &A,
    k: usize,
    mut models: Vec<Mo>,
    config: EmConfig<N>,
) -> EmResult<N, Mo>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    A: Dataset<N>,
    Mo: EmModel<N>,
{
    assert!(k > 0, "k must be positive");
    assert!(data.nrows() > 0, "dataset must not be empty");
    assert!(data.ncols() > 0, "dataset dimensionality must be positive");

    let (n, d) = (data.nrows(), data.ncols());
    assert_eq!(models.len(), k, "wrong number of initial models");

    let mut scratch = vec![N::zero(); d];
    let mut probs = vec![N::zero(); n * k];
    let mut noise_probs = if config.noise_ratio > N::zero() {
        Some(vec![N::zero(); n])
    } else {
        None
    };
    let mut log_noise = config.min_log_likelihood;
    let noise_density = if noise_probs.is_some() {
        Some(log_noise)
    } else {
        None
    };
    let mut noise_slice = noise_probs.as_deref_mut();
    let (mut log_likelihood, mut noise_weight) = assign_probabilities_to_instances::<N, A, Mo>(
        data,
        &models,
        &mut probs,
        &mut scratch,
        config.min_log_likelihood,
        noise_density,
        noise_slice,
    );

    let mut best_log_likelihood = N::neg_infinity();
    let mut last_improvement = 0usize;
    let mut iter = 0usize;

    while iter < config.maxiter {
        iter += 1;
        let old = log_likelihood;

        if config.hard {
            recompute_models_hard::<N, A, Mo>(
                data,
                &probs,
                noise_probs.as_deref(),
                &mut models,
                config.prior,
                &mut scratch,
            );
        } else {
            recompute_models_soft::<N, A, Mo>(
                data,
                &probs,
                &mut models,
                config.prior,
                &mut scratch,
            );
        }

        if config.noise_ratio > N::zero()
            && let Some(nweight) = noise_weight
            && nweight > N::zero()
        {
            log_noise += (config.noise_ratio / nweight).ln();
        } else if config.noise_ratio > N::zero() && noise_weight.is_some() {
            log_noise = log_likelihood + config.noise_ratio.ln();
        }

        noise_slice = noise_probs.as_deref_mut();
        let (new_log_likelihood, new_noise_weight) = assign_probabilities_to_instances(
            data,
            &models,
            &mut probs,
            &mut scratch,
            config.min_log_likelihood,
            if config.noise_ratio > N::zero() {
                Some(log_noise.max(config.min_log_likelihood))
            } else {
                None
            },
            noise_slice,
        );
        log_likelihood = new_log_likelihood;
        noise_weight = new_noise_weight;

        if log_likelihood - best_log_likelihood > config.delta {
            last_improvement = iter;
            best_log_likelihood = log_likelihood;
        }

        if iter >= config.miniter {
            let no_change = (log_likelihood - old).abs() <= config.delta;
            let stale = last_improvement < (iter >> 1);
            if no_change || stale {
                break;
            }
        }
    }

    let mut assignments = Vec::with_capacity(n);
    for i in 0..n {
        let slice = &probs[i * k..(i + 1) * k];
        let mut best = argmax(slice);
        if let Some(noise_vec) = noise_probs.as_ref()
            && noise_vec[i] > slice[best]
        {
            best = k;
        }
        assignments.push(best);
    }

    let responsibilities = if config.return_soft {
        Some(Array2::from_shape_vec((n, k), probs).unwrap())
    } else {
        None
    };

    EmResult {
        models,
        assignments,
        responsibilities,
        n_iter: iter,
        log_likelihood,
    }
}
