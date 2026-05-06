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
            use rayon::prelude::*;
            self.into_par_iter().map(f).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.map(f).collect()
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
            use rayon::prelude::*;
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

/// Maps over paired chunks of two mutable slices, where `b` has `b_per_a` elements
/// for each element of `a`. With `parallel` enabled, processes chunks in parallel
/// via rayon. Without `parallel`, processes the entire pair as a single chunk.
///
/// The closure receives `(i0, a_chunk, b_chunk)` where `i0` is the absolute
/// starting index of the chunk in `a`.
pub fn par_zip_chunks_map_mut<T, U, F, R>(
    a: &mut [T], b: &mut [U], b_per_a: usize, f: F,
) -> Vec<R>
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
