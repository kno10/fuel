use std::sync::atomic::Ordering::Relaxed;

#[cfg(feature = "python")]
use pyo3::exceptions::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Extension trait that transparently switches between parallel and sequential
/// mapping over a `Range<usize>`.
///
/// With the `parallel` feature enabled, `par_map` uses rayon's `into_par_iter`;
/// without it, it falls back to a plain sequential iterator. In both cases the
/// results are collected into a `Vec<R>`.
///
/// # Example
/// ```ignore
/// use crate::ParMap;
/// let results: Vec<_> = (0..n).par_map(|i| expensive(i));
/// ```
pub trait ParMap {
    fn par_map<F, R>(self, f: F) -> Vec<R>
    where
        F: Fn(usize) -> R + Sync + Send,
        R: Send;

    /// Like `par_map`, but the closure returns `Result<R, String>`.
    /// The parallel path checks `SHUTDOWN_REQUESTED` before each item and
    /// short-circuits on the first `Err`; the sequential path calls
    /// `poll_interrupted()` before each item.
    fn par_try_map<F, R>(self, f: F) -> Result<Vec<R>, String>
    where
        F: Fn(usize) -> Result<R, String> + Sync + Send,
        R: Send;
}

impl ParMap for std::ops::Range<usize> {
    #[inline]
    fn par_map<F, R>(self, f: F) -> Vec<R>
    where
        F: Fn(usize) -> R + Sync + Send,
        R: Send,
    {
        #[cfg(feature = "parallel")]
        {
            self.into_par_iter()
                .map(|i| {
                    // Probe for interrupts: on the main thread this acquires the
                    // GIL (rate-limited) and sets SHUTDOWN_REQUESTED; on worker
                    // threads it only reads that flag.  We ignore the Err return
                    // here because par_map cannot propagate it - callers are
                    // expected to call crate::poll_interrupted()? afterwards.
                    let _ = crate::poll_interrupted();
                    f(i)
                })
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.map(f).collect()
        }
    }

    #[inline]
    fn par_try_map<F, R>(self, f: F) -> Result<Vec<R>, String>
    where
        F: Fn(usize) -> Result<R, String> + Sync + Send,
        R: Send,
    {
        #[cfg(feature = "parallel")]
        {
            self.into_par_iter()
                .map(|i| {
                    if crate::SHUTDOWN_REQUESTED.load(Relaxed) {
                        return Err("interrupted".to_string());
                    }
                    f(i)
                })
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut out = Vec::with_capacity(self.len());
            for i in self {
                if crate::SHUTDOWN_REQUESTED.load(Relaxed) {
                    return Err("interrupted".to_string());
                }
                out.push(f(i)?);
            }
            Ok(out)
        }
    }
}

/// Extension trait for mutable slices that maps over parallel chunks when the
/// `parallel` feature is enabled, and falls back to sequential processing otherwise.
///
/// The closure receives `(i0, chunk)` where `i0` is the absolute starting index
/// of the chunk in the original slice.
///
/// With `parallel`: the slice is split into `num_threads` chunks and processed
/// in parallel via rayon. Without `parallel`: the entire slice is processed as a
/// single chunk in one call.
pub trait ParChunksMut<T: Send> {
    fn par_chunks_map_mut<F, R>(self, f: F) -> Vec<R>
    where
        F: Fn(usize, &mut [T]) -> R + Sync + Send,
        R: Send;
}

impl<T: Send> ParChunksMut<T> for &mut [T] {
    #[inline]
    fn par_chunks_map_mut<F, R>(self, f: F) -> Vec<R>
    where
        F: Fn(usize, &mut [T]) -> R + Sync + Send,
        R: Send,
    {
        #[cfg(feature = "parallel")]
        {
            let chunk_size = self.len().div_ceil(rayon::current_num_threads()).max(1);
            self.par_chunks_mut(chunk_size)
                .enumerate()
                .map(|(ti, chunk)| f(ti * chunk_size, chunk))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            vec![f(0, self)]
        }
    }
}

/// Run `f` on a background thread while the Python main thread polls for signals.
///
/// Calls [`crate::reset_interrupted`], spawns a scoped thread for `f`, then
/// loops releasing the GIL with `park_timeout` and calling `py.check_signals()`
/// on each wake-up. When a signal is detected, sets `SHUTDOWN_REQUESTED` and
/// returns the `KeyboardInterrupt` error immediately; the scoped thread is
/// joined before the scope exits. Worker threads abort early via the flag when
/// they call [`crate::par_try_map`].
///
/// # Errors
/// Returns any `PyResult::Err` propagated from `f` (converted via
/// `pyo3::exceptions::PyRuntimeError`) or from `check_signals`.
#[cfg(feature = "python")]
pub fn py_interruptible<F, R>(py: pyo3::Python<'_>, f: F) -> pyo3::PyResult<R>
where
    F: FnOnce() -> Result<R, String> + Send,
    R: Send,
{
    crate::reset_interrupted();
    std::thread::scope(|scope| {
        let main_thread = std::thread::current();
        let handle = scope.spawn(move || {
            let result = f();
            main_thread.unpark();
            result
        });
        loop {
            py.detach(|| std::thread::park_timeout(std::time::Duration::from_millis(10)));
            if handle.is_finished() {
                break;
            }
            if let Err(e) = py.check_signals() {
                crate::SHUTDOWN_REQUESTED.store(true, Relaxed);
                return Err(e);
            }
        }
        handle
            .join()
            .map_err(|_| PyRuntimeError::new_err("worker thread panicked"))?
            .map_err(|e| PyRuntimeError::new_err(e))
    })
}

/// Maps over paired chunks of two mutable slices, where `b` has `b_per_a` elements
/// for each element of `a`. With `parallel` enabled, processes chunks in parallel
/// via rayon. Without `parallel`, processes the entire pair as a single chunk.
///
/// The closure receives `(i0, a_chunk, b_chunk)` where `i0` is the absolute
/// starting index of the chunk in `a`.
pub fn par_zip_chunks_map_mut<T, U, F, R>(a: &mut [T], b: &mut [U], b_per_a: usize, f: F) -> Vec<R>
where
    T: Send,
    U: Send,
    F: Fn(usize, &mut [T], &mut [U]) -> R + Sync + Send,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let chunk_size = a.len().div_ceil(rayon::current_num_threads()).max(1);
        a.par_chunks_mut(chunk_size)
            .zip(b.par_chunks_mut(chunk_size * b_per_a))
            .enumerate()
            .map(|(ti, (a_chunk, b_chunk))| f(ti * chunk_size, a_chunk, b_chunk))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        vec![f(0, a, b)]
    }
}
