use std::iter::Sum; // FIXME: no longer necessary here if we use crate::Float?
use std::ops::{AddAssign, MulAssign, SubAssign};

use ndarray::Array2;

use crate::{
    Float, ParChunksMut, ParMap, VectorData as Dataset, par_row_chunks_try_map,
    par_zip_chunks_try_map_mut,
};

/// Pluggable cluster model for expectation-maximization.
pub trait EmModel<N>
where
    N: Float + Copy,
{
    /// Begin accumulation for a new M-step.
    fn begin_estep(&mut self);

    /// Whether this model needs a first pass before the regular update pass.
    fn needs_two_pass(&self) -> bool { false }

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

    /// Merge the partial M-step accumulation state from `other` into `self`.
    ///
    /// Called after `update_estep` on a parallel chunk to combine per-thread
    /// partial results. For single-pass models (Welford or textbook) this
    /// merges the full state including `wsum`, `mean`, and the variance/
    /// covariance accumulator. For two-pass models it merges only the
    /// second-pass accumulator (variance or covariance), because `wsum` and
    /// `mean` are already fixed after the first pass.
    fn merge_estep(&mut self, other: &Self)
    where
        Self: Sized;

    /// Merge the first-pass accumulation state from `other` into `self`.
    ///
    /// Only meaningful for two-pass models; the default no-op is correct for
    /// all single-pass models. After all partial first-pass results are merged
    /// in, `finalize_first_pass_estep` is called on `self`.
    fn merge_first_pass_estep(&mut self, _other: &Self)
    where
        Self: Sized,
    {
    }
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
    pub point_log_likelihoods: Option<Vec<N>>,
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
            acc += if v < max { (v - max).exp() } else { N::one() };
        }
    }
    if acc > N::one() { max + acc.ln() } else { max }
}

fn assign_probabilities_to_instances<N, A, Mo>(
    data: &A, models: &[Mo], probs: &mut [N], min_log_likelihood: N, noise_log_density: Option<N>,
    mut noise_probs: Option<&mut [N]>,
) -> Result<(N, Option<N>), String>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    A: Dataset<N> + Sync,
    Mo: EmModel<N> + Sync,
{
    let (n, d) = (data.nrows(), data.ncols());
    let k = models.len();
    let local_len = k + if noise_log_density.is_some() { 1 } else { 0 };

    let partial: Vec<(N, N)> = if let Some(np) = noise_probs.as_mut() {
        par_zip_chunks_try_map_mut(np, probs, k, |i0, np_chunk, probs_chunk| {
            let chunk_rows = np_chunk.len();
            let mut sc = vec![N::zero(); d];
            let mut local = vec![N::zero(); local_len];
            let mut em_sum = N::zero();
            let mut noise_sum = N::zero();
            for ri in 0..chunk_rows {
                data.load_into(i0 + ri, &mut sc, d);
                for j in 0..k {
                    local[j] = models[j].estimate_log_density(&sc).max(min_log_likelihood);
                }
                local[k] = noise_log_density.unwrap().max(min_log_likelihood);
                let logp = log_sum_exp(&local);
                for j in 0..k {
                    probs_chunk[ri * k + j] = (local[j] - logp).exp();
                }
                let nv = (local[k] - logp).exp();
                np_chunk[ri] = nv;
                noise_sum += nv;
                em_sum += logp;
            }
            Ok((em_sum, noise_sum))
        })
    } else {
        probs.par_row_chunks_try_map_mut(k, |row_start, probs_chunk| {
            let chunk_rows = probs_chunk.len() / k;
            let mut sc = vec![N::zero(); d];
            let mut local = vec![N::zero(); k];
            let mut em_sum = N::zero();
            for ri in 0..chunk_rows {
                data.load_into(row_start + ri, &mut sc, d);
                for j in 0..k {
                    local[j] = models[j].estimate_log_density(&sc).max(min_log_likelihood);
                }
                let logp = log_sum_exp(&local);
                for j in 0..k {
                    probs_chunk[ri * k + j] = (local[j] - logp).exp();
                }
                em_sum += logp;
            }
            Ok((em_sum, N::zero()))
        })
    }?;

    let (em_total, noise_total) =
        partial.into_iter().fold((N::zero(), N::zero()), |(a, b), (c, e)| (a + c, b + e));
    Ok((
        em_total / N::from(n).unwrap(),
        if noise_log_density.is_some() { Some(noise_total / N::from(n).unwrap()) } else { None },
    ))
}

fn recompute_models_soft<N, A, Mo>(
    data: &A, probs: &[N], models: &mut [Mo], prior: N,
) -> Result<(), String>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    A: Dataset<N> + Sync,
    Mo: EmModel<N> + Clone + Send + Sync,
{
    let (n, d) = (data.nrows(), data.ncols());
    let k = models.len();
    let needs_two_pass = models.iter().any(|m| m.needs_two_pass());

    for model in models.iter_mut() {
        model.begin_estep();
    }

    if needs_two_pass {
        // Parallel first pass: each thread clones the zero-state models and
        // accumulates first-pass statistics (unnormalized weighted mean sums).
        let first_pass: Vec<Vec<Mo>> =
            par_row_chunks_try_map(probs, k, |row_start, probs_chunk| {
                let chunk_rows = probs_chunk.len() / k;
                let mut local: Vec<Mo> = models.to_vec();
                let mut sc = vec![N::zero(); d];
                for ri in 0..chunk_rows {
                    data.load_into(row_start + ri, &mut sc, d);
                    for j in 0..k {
                        let p = probs_chunk[ri * k + j];
                        if p > N::from(1e-10).unwrap() {
                            local[j].first_pass_estep(&sc, p);
                        }
                    }
                }
                Ok(local)
            })?;
        for partial in first_pass {
            for j in 0..k {
                models[j].merge_first_pass_estep(&partial[j]);
            }
        }
        for model in models.iter_mut() {
            model.finalize_first_pass_estep();
        }
    }

    // Parallel main update pass: each thread clones the current model state
    // (zero for single-pass models; finalized-first-pass state for two-pass
    // models) and accumulates per-chunk update statistics.
    let update: Vec<(Vec<Mo>, Vec<N>)> =
        par_row_chunks_try_map(probs, k, |row_start, probs_chunk| {
            let chunk_rows = probs_chunk.len() / k;
            let mut local: Vec<Mo> = models.to_vec();
            let mut partial_wsum = vec![N::zero(); k];
            let mut sc = vec![N::zero(); d];
            for ri in 0..chunk_rows {
                data.load_into(row_start + ri, &mut sc, d);
                for j in 0..k {
                    let p = probs_chunk[ri * k + j];
                    local[j].update_estep(&sc, p);
                    partial_wsum[j] += p;
                }
            }
            Ok((local, partial_wsum))
        })?;

    // Merge partial results: single-pass models use Chan's parallel algorithm
    // inside merge_estep; two-pass models add only the second-pass accumulator.
    let mut wsum = vec![N::zero(); k];
    for (partial_models, partial_wsum) in update {
        for j in 0..k {
            models[j].merge_estep(&partial_models[j]);
            wsum[j] += partial_wsum[j];
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
    Ok(())
}

fn recompute_models_hard<N, A, Mo>(
    data: &A, probs: &[N], noise_probs: Option<&[N]>, models: &mut [Mo], prior: N,
) -> Result<(), String>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    A: Dataset<N> + Sync,
    Mo: EmModel<N> + Clone + Send + Sync,
{
    let (n, d) = (data.nrows(), data.ncols());
    let k = models.len();
    let needs_two_pass = models.iter().any(|m| m.needs_two_pass());

    for model in models.iter_mut() {
        model.begin_estep();
    }

    if needs_two_pass {
        let first_pass: Vec<Vec<Mo>> =
            par_row_chunks_try_map(probs, k, |row_start, probs_chunk| {
                let chunk_rows = probs_chunk.len() / k;
                let mut local: Vec<Mo> = models.to_vec();
                let mut sc = vec![N::zero(); d];
                for ri in 0..chunk_rows {
                    data.load_into(row_start + ri, &mut sc, d);
                    let slice = &probs_chunk[ri * k..(ri + 1) * k];
                    let best = argmax(slice);
                    if let Some(noise) = noise_probs
                        && noise[row_start + ri] > slice[best]
                    {
                        continue;
                    }
                    local[best].first_pass_estep(&sc, N::one());
                }
                Ok(local)
            })?;
        for partial in first_pass {
            for j in 0..k {
                models[j].merge_first_pass_estep(&partial[j]);
            }
        }
        for model in models.iter_mut() {
            model.finalize_first_pass_estep();
        }
    }

    let update: Vec<(Vec<Mo>, Vec<N>)> =
        par_row_chunks_try_map(probs, k, |row_start, probs_chunk| {
            let chunk_rows = probs_chunk.len() / k;
            let mut local: Vec<Mo> = models.to_vec();
            let mut partial_wsum = vec![N::zero(); k];
            let mut sc = vec![N::zero(); d];
            for ri in 0..chunk_rows {
                data.load_into(row_start + ri, &mut sc, d);
                let slice = &probs_chunk[ri * k..(ri + 1) * k];
                let best = argmax(slice);
                if let Some(noise) = noise_probs && noise[row_start + ri] > slice[best] {
                    continue;
                }
                local[best].update_estep(&sc, N::one());
                partial_wsum[best] += N::one();
            }
            Ok((local, partial_wsum))
        })?;

    let mut wsum = vec![N::zero(); k];
    for (partial_models, partial_wsum) in update {
        for j in 0..k {
            models[j].merge_estep(&partial_models[j]);
            wsum[j] += partial_wsum[j];
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
    Ok(())
}

/// Generic EM solver for mixture models.
pub fn expectation_maximization<N, A, Mo>(
    data: &A, k: usize, mut models: Vec<Mo>, config: EmConfig<N>,
) -> Result<EmResult<N, Mo>, String>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    A: Dataset<N> + Sync,
    Mo: EmModel<N> + Clone + Send + Sync,
{
    assert!(k > 0, "k must be positive");
    assert!(data.nrows() > 0, "dataset must not be empty");
    assert!(data.ncols() > 0, "dataset dimensionality must be positive");

    let n = data.nrows();
    assert_eq!(models.len(), k, "wrong number of initial models");
    let mut probs = vec![N::zero(); n * k];
    let mut noise_probs =
        if config.noise_ratio > N::zero() { Some(vec![N::zero(); n]) } else { None };
    let mut log_noise = config.min_log_likelihood;
    let noise_density = if noise_probs.is_some() { Some(log_noise) } else { None };
    let mut noise_slice = noise_probs.as_deref_mut();
    let (mut log_likelihood, mut noise_weight) = assign_probabilities_to_instances::<N, A, Mo>(
        data,
        &models,
        &mut probs,
        config.min_log_likelihood,
        noise_density,
        noise_slice,
    )?;

    let mut best_log_likelihood = N::neg_infinity();
    let mut last_improvement = 0usize;
    let mut iter = 0usize;

    while iter < config.maxiter {
        iter += 1;
        let old = log_likelihood;

        crate::check_interrupted()?;

        if config.hard {
            recompute_models_hard::<N, A, Mo>(
                data,
                &probs,
                noise_probs.as_deref(),
                &mut models,
                config.prior,
            )?;
        } else {
            recompute_models_soft::<N, A, Mo>(data, &probs, &mut models, config.prior)?;
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
        let (new_log_likelihood, new_noise_weight) = assign_probabilities_to_instances::<N, A, Mo>(
            data,
            &models,
            &mut probs,
            config.min_log_likelihood,
            if config.noise_ratio > N::zero() {
                Some(log_noise.max(config.min_log_likelihood))
            } else {
                None
            },
            noise_slice,
        )?;
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
        crate::check_interrupted()?;
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

    let point_log_likelihoods: Vec<N> = (0..n).par_map(|i| {
        let mut scratch = vec![N::zero(); data.ncols()];
        let mut local = vec![N::zero(); k];
        data.load_into(i, &mut scratch, data.ncols());
        for j in 0..k {
            local[j] = models[j].estimate_log_density(&scratch);
        }
        log_sum_exp(&local)
    });

    Ok(EmResult {
        models,
        assignments,
        responsibilities,
        n_iter: iter,
        log_likelihood,
        point_log_likelihoods: Some(point_log_likelihoods),
    })
}
