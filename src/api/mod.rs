//! API traits, data storage, and basic search primitives.

pub mod condensed_distance_matrix;
pub mod data;
pub mod float;
pub mod ndarray;
pub mod parallel;
pub mod query;
pub mod search;
pub mod square_distance_matrix;
pub mod tabular;

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

pub use condensed_distance_matrix::*;
pub use data::*;
pub use float::*;
pub use ndarray::*;
pub use parallel::*;
pub use query::*;
pub use search::*;
pub use square_distance_matrix::*;
pub use tabular::*;

/// Set to `true` by the main Python thread when a pending signal is detected.
/// Worker threads (e.g. rayon) read this flag to bail out early without
/// acquiring the GIL.
pub static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Timestamp (ms) of the last GIL-acquiring interrupt check.
/// Written and read exclusively on the Python main thread.
static mut LAST_CHECK_MS: u64 = 0;

/// Interval (ms) at which the main thread should acquire the GIL to check for interrupts.
#[cfg(feature = "python")]
static CHECK_INTERVAL_MS: u64 = 50;

thread_local! {
    /// `true` only on the thread that last called [`reset_interrupted()`].
    /// That function is always invoked on the Python main thread before the GIL
    /// is released, so this reliably identifies the main thread without any
    /// GIL-acquiring detection logic.
    static IS_MAIN_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Reset the global interrupt state before starting a long computation.
///
/// Must be called before releasing the GIL so that a stale flag from a
/// previously-interrupted run does not block the new computation.
pub fn reset_interrupted() {
    SHUTDOWN_REQUESTED.store(false, Ordering::Relaxed);
    // Safety: called on the Python main thread before releasing the GIL.
    unsafe { LAST_CHECK_MS = 0 };
    // Mark this thread as the main thread so poll_interrupted() will do GIL
    // checks here.  reset_interrupted() is always called on the Python main
    // thread, before py.detach() releases the GIL.
    IS_MAIN_THREAD.with(|cell| cell.set(true));
}

/// Poll for a Python interrupt signal; safe to call from any thread including
/// rayon worker threads.
///
/// * **Main Python thread**: acquires the GIL at most once every
///   `CHECK_INTERVAL_MS` milliseconds.  If a `KeyboardInterrupt` (or other
///   pending signal) is found, sets [`SHUTDOWN_REQUESTED`] and returns `Err`.
/// * **Any other thread**: only reads [`SHUTDOWN_REQUESTED`] - no GIL involved.
pub fn poll_interrupted() -> Result<(), String> {
    // Fast path: flag already set (covers both main and worker threads).
    if SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
        return Err("interrupted".to_string());
    }

    // Worker threads have nothing more to do.
    let is_main = IS_MAIN_THREAD.with(|cell| cell.get());
    if !is_main {
        return Ok(());
    }

    // Main thread: rate-limit the expensive GIL acquisition.
    #[cfg(feature = "python")]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now =
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
        // Safety: only reached when is_main == true, i.e. on the main thread.
        let last = unsafe { LAST_CHECK_MS };
        if now.saturating_sub(last) >= CHECK_INTERVAL_MS {
            unsafe { LAST_CHECK_MS = now };
            let result = check_interrupted();
            if result.is_err() {
                SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
            }
            return result;
        }
    }

    Ok(())
}

/// Check if the current operation should be interrupted (e.g., from a Python signal).
///
/// When compiled with the `python` feature and a Python interpreter is attached,
/// this calls `py.check_signals()` so that KeyboardInterrupt is propagated.
/// Otherwise it is a no-op.
pub fn check_interrupted() -> Result<(), String> {
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        if let Some(result) =
            Python::try_attach(|py| py.check_signals().map_err(|err| err.to_string()))
        {
            result?;
        }
    }
    Ok(())
}
