//! Core mathematical primitives.
//!
//! Public API: plain free functions (`sqdist`, `dot`, `axpy`, ...).
//! Back-end selection (AVX2, unrolled scalar, plain scalar) happens at
//! monomorphisation time through the [`crate::VecOps`] trait.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(not(target_arch = "x86_64"))]
pub mod unroll;

use ndarray::{ArrayBase, ArrayView2, Data, Ix2};

use crate::{Float, VecOps};

/// Minimum number of dimensions to invoke the AVX2 or unrolled back-ends.
const UNROLL_SIZE: usize = 4;
/// Minimum n (columns = points) for the parallel path in `vec_pairwise_sqdist`.
/// Splits centers across threads; each thread calls the full pairwise kernel on its slice.
#[cfg(feature = "parallel")]
const PARALLEL_PAIRWISE_N_THRESHOLD: usize = 1 << 15; // 32k points
/// Minimum n for the parallel path in `vec_row_sqdist` (1 center vs n points).
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_ROW_THRESHOLD: usize = 1 << 10; // 1024 points

// ---------------------------------------------------------------------------
// VecOps impls - back-end selected at monomorphisation, zero runtime overhead
// ---------------------------------------------------------------------------

/// Newtype wrapper to send a raw pointer across rayon threads.
/// Safety: used only in `vec_pairwise_sqdist` n-split, where per-thread writes
/// are to non-overlapping index ranges.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct RawSendPtr<T>(*mut T);
#[cfg(feature = "parallel")]
impl<T> RawSendPtr<T> {
    /// # Safety
    /// Caller must ensure the resulting reference does not alias with any other reference.
    #[inline(always)]
    unsafe fn offset_ptr(self, offset: usize) -> *mut T { unsafe { self.0.add(offset) } }
}
#[cfg(feature = "parallel")]
unsafe impl<T> Send for RawSendPtr<T> {}
#[cfg(feature = "parallel")]
unsafe impl<T> Sync for RawSendPtr<T> {}

impl VecOps for f32 {
    #[inline(always)]
    fn vec_sqdist(v1: &[f32], v2: &[f32], d: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sqdist_f32(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sqdist(v1, v2, d);
        }
        scalar::sqdist(v1, v2, d)
    }
    #[inline(always)]
    fn vec_pairwise_sqdist<'a>(
        points1: ArrayView2<'a, f32>, points2: ArrayView2<'a, f32>, d: usize, out: &mut [f32],
        nrows: usize, ncols: usize,
    ) {
        #[cfg(feature = "parallel")]
        if ncols >= PARALLEL_PAIRWISE_N_THRESHOLD {
            use rayon::prelude::*;
            let nthreads = rayon::current_num_threads().max(1);
            if nrows >= nthreads {
                // Center-split: each thread handles a chunk of centers (rows of points1).
                // Preserves pack-A-once + sweep-B within each thread.
                let row_chunk = nrows.div_ceil(nthreads);
                out.par_chunks_mut(row_chunk * ncols).enumerate().for_each(|(ci, out_block)| {
                    let row_start = ci * row_chunk;
                    let block_rows = out_block.len() / ncols;
                    let p1 = points1.slice(ndarray::s![row_start..row_start + block_rows, ..]);
                    #[cfg(target_arch = "x86_64")]
                    if d >= UNROLL_SIZE {
                        return avx2::pairwise_sqdist_between_f32(
                            p1, points2, d, out_block, block_rows, ncols,
                        );
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    if d >= UNROLL_SIZE {
                        return unroll::pairwise_sqdist_between(
                            p1, points2, d, out_block, block_rows, ncols,
                        );
                    }
                    scalar::pairwise_sqdist_between(p1, points2, d, out_block, block_rows, ncols);
                });
            } else {
                // N-split: each thread handles a chunk of points (columns).
                // Computes all k distances for its point slice into a local buffer,
                // then scatter-copies into the center-major output layout.
                let col_chunk = ncols.div_ceil(nthreads);
                let out_ptr = RawSendPtr(out.as_mut_ptr());
                (0..nthreads).into_par_iter().for_each(move |ci| {
                    let i0 = ci * col_chunk;
                    if i0 >= ncols {
                        return;
                    }
                    let chunk_n = col_chunk.min(ncols - i0);
                    let pts = points2.slice(ndarray::s![i0..i0 + chunk_n, ..]);
                    let mut tmp = vec![0f32; nrows * chunk_n];
                    #[cfg(target_arch = "x86_64")]
                    if d >= UNROLL_SIZE {
                        avx2::pairwise_sqdist_between_f32(
                            points1, pts, d, &mut tmp, nrows, chunk_n,
                        );
                    } else {
                        scalar::pairwise_sqdist_between(points1, pts, d, &mut tmp, nrows, chunk_n);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    if d >= UNROLL_SIZE {
                        unroll::pairwise_sqdist_between(points1, pts, d, &mut tmp, nrows, chunk_n);
                    } else {
                        scalar::pairwise_sqdist_between(points1, pts, d, &mut tmp, nrows, chunk_n);
                    }
                    // scatter-copy: tmp[j*chunk_n..] -> out[j*ncols + i0..]
                    for j in 0..nrows {
                        let src = &tmp[j * chunk_n..(j + 1) * chunk_n];
                        // Safety: thread ci owns [i0..i0+chunk_n] for each row j; ranges non-overlapping.
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                out_ptr.offset_ptr(j * ncols + i0),
                                chunk_n,
                            )
                        };
                        dst.copy_from_slice(src);
                    }
                });
            }
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::pairwise_sqdist_between_f32(points1, points2, d, out, nrows, ncols);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::pairwise_sqdist_between(points1, points2, d, out, nrows, ncols);
        }
        scalar::pairwise_sqdist_between(points1, points2, d, out, nrows, ncols);
    }
    #[inline(always)]
    fn vec_row_sqdist<'a>(
        center: &[f32], points: ArrayView2<'a, f32>, d: usize, out: &mut [f32], n: usize,
    ) {
        #[cfg(feature = "parallel")]
        if n >= PARALLEL_ROW_THRESHOLD {
            use rayon::prelude::*;
            let nthreads = rayon::current_num_threads().max(1);
            let chunk = n.div_ceil(nthreads);
            out.par_chunks_mut(chunk).enumerate().for_each(|(ci, chunk_out)| {
                let jj = ci * chunk;
                let chunk_n = chunk_out.len();
                let pts = points.slice(ndarray::s![jj..jj + chunk_n, ..]);
                #[cfg(target_arch = "x86_64")]
                if d >= UNROLL_SIZE {
                    return avx2::rowdist_f32(center, pts, d, chunk_out, chunk_n);
                }
                #[cfg(not(target_arch = "x86_64"))]
                if d >= UNROLL_SIZE {
                    return unroll::rowdist(center, pts, d, chunk_out, chunk_n);
                }
                scalar::rowdist(center, pts, d, chunk_out, chunk_n);
            });
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::rowdist_f32(center, points, d, out, n);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::rowdist(center, points, d, out, n);
        }
        scalar::rowdist(center, points, d, out, n);
    }
    #[inline(always)]
    fn vec_l1dist(v1: &[f32], v2: &[f32], d: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::l1dist_f32(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::l1dist(v1, v2, d);
        }
        scalar::l1dist(v1, v2, d)
    }
    #[inline(always)]
    fn vec_mul(v1: &mut [f32], v2: &[f32], a: f32, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::mul_f32(v1, v2, a, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::mul(v1, v2, a, d);
        }
        scalar::mul(v1, v2, a, d)
    }
    #[inline(always)]
    fn vec_mul_assign(v: &mut [f32], f: f32, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::mul_assign_f32(v, f, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::mul_assign(v, f, d);
        }
        scalar::mul_assign(v, f, d)
    }
    #[inline(always)]
    fn vec_add_assign(v1: &mut [f32], v2: &[f32], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::add_assign_f32(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::add_assign(v1, v2, d);
        }
        scalar::add_assign(v1, v2, d)
    }
    #[inline(always)]
    fn vec_sub_assign(v1: &mut [f32], v2: &[f32], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sub_assign_f32(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sub_assign(v1, v2, d);
        }
        scalar::sub_assign(v1, v2, d)
    }
    #[inline(always)]
    fn vec_fmamul(v1: &mut [f32], a: f32, v2: &[f32], b: f32, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::fmamul_f32(v1, a, v2, b, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::fmamul(v1, a, v2, b, d);
        }
        scalar::fmamul(v1, a, v2, b, d)
    }
    #[inline(always)]
    fn vec_dot(v1: &[f32], v2: &[f32], d: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::dot_f32(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::dot(v1, v2, d);
        }
        scalar::dot(v1, v2, d)
    }
    #[inline(always)]
    fn vec_axpy(v1: &mut [f32], a: f32, v2: &[f32], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::axpy_f32(v1, a, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::axpy(v1, a, v2, d);
        }
        scalar::axpy(v1, a, v2, d)
    }
    #[inline(always)]
    fn vec_sum(v: &[f32], d: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sum_f32(v, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sum(v, d);
        }
        scalar::sum(v, d)
    }
    #[inline(always)]
    fn vec_add_scalar(v: &mut [f32], s: f32, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::add_scalar_f32(v, s, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::add_scalar(v, s, d);
        }
        scalar::add_scalar(v, s, d)
    }
}

impl VecOps for f64 {
    #[inline(always)]
    fn vec_sqdist(v1: &[f64], v2: &[f64], d: usize) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sqdist_f64(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sqdist(v1, v2, d);
        }
        scalar::sqdist(v1, v2, d)
    }
    #[inline(always)]
    fn vec_pairwise_sqdist<'a>(
        points1: ArrayView2<'a, f64>, points2: ArrayView2<'a, f64>, d: usize, out: &mut [f64],
        nrows: usize, ncols: usize,
    ) {
        #[cfg(feature = "parallel")]
        if ncols >= PARALLEL_PAIRWISE_N_THRESHOLD {
            use rayon::prelude::*;
            let nthreads = rayon::current_num_threads().max(1);
            if nrows >= nthreads {
                let row_chunk = nrows.div_ceil(nthreads);
                out.par_chunks_mut(row_chunk * ncols).enumerate().for_each(|(ci, out_block)| {
                    let row_start = ci * row_chunk;
                    let block_rows = out_block.len() / ncols;
                    let p1 = points1.slice(ndarray::s![row_start..row_start + block_rows, ..]);
                    #[cfg(target_arch = "x86_64")]
                    if d >= UNROLL_SIZE {
                        return avx2::pairwise_sqdist_between_f64(
                            p1, points2, d, out_block, block_rows, ncols,
                        );
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    if d >= UNROLL_SIZE {
                        return unroll::pairwise_sqdist_between(
                            p1, points2, d, out_block, block_rows, ncols,
                        );
                    }
                    scalar::pairwise_sqdist_between(p1, points2, d, out_block, block_rows, ncols);
                });
            } else {
                let col_chunk = ncols.div_ceil(nthreads);
                let out_ptr = RawSendPtr(out.as_mut_ptr());
                (0..nthreads).into_par_iter().for_each(move |ci| {
                    let i0 = ci * col_chunk;
                    if i0 >= ncols {
                        return;
                    }
                    let chunk_n = col_chunk.min(ncols - i0);
                    let pts = points2.slice(ndarray::s![i0..i0 + chunk_n, ..]);
                    let mut tmp = vec![0f64; nrows * chunk_n];
                    #[cfg(target_arch = "x86_64")]
                    if d >= UNROLL_SIZE {
                        avx2::pairwise_sqdist_between_f64(
                            points1, pts, d, &mut tmp, nrows, chunk_n,
                        );
                    } else {
                        scalar::pairwise_sqdist_between(points1, pts, d, &mut tmp, nrows, chunk_n);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    if d >= UNROLL_SIZE {
                        unroll::pairwise_sqdist_between(points1, pts, d, &mut tmp, nrows, chunk_n);
                    } else {
                        scalar::pairwise_sqdist_between(points1, pts, d, &mut tmp, nrows, chunk_n);
                    }
                    for j in 0..nrows {
                        let src = &tmp[j * chunk_n..(j + 1) * chunk_n];
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                out_ptr.offset_ptr(j * ncols + i0),
                                chunk_n,
                            )
                        };
                        dst.copy_from_slice(src);
                    }
                });
            }
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::pairwise_sqdist_between_f64(points1, points2, d, out, nrows, ncols);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::pairwise_sqdist_between(points1, points2, d, out, nrows, ncols);
        }
        scalar::pairwise_sqdist_between(points1, points2, d, out, nrows, ncols);
    }
    #[inline(always)]
    fn vec_row_sqdist<'a>(
        center: &[f64], points: ArrayView2<'a, f64>, d: usize, out: &mut [f64], n: usize,
    ) {
        #[cfg(feature = "parallel")]
        if n >= PARALLEL_ROW_THRESHOLD {
            use rayon::prelude::*;
            let nthreads = rayon::current_num_threads().max(1);
            let chunk = n.div_ceil(nthreads);
            out.par_chunks_mut(chunk).enumerate().for_each(|(ci, chunk_out)| {
                let jj = ci * chunk;
                let chunk_n = chunk_out.len();
                let pts = points.slice(ndarray::s![jj..jj + chunk_n, ..]);
                #[cfg(target_arch = "x86_64")]
                if d >= UNROLL_SIZE {
                    return avx2::rowdist_f64(center, pts, d, chunk_out, chunk_n);
                }
                #[cfg(not(target_arch = "x86_64"))]
                if d >= UNROLL_SIZE {
                    return unroll::rowdist(center, pts, d, chunk_out, chunk_n);
                }
                scalar::rowdist(center, pts, d, chunk_out, chunk_n);
            });
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::rowdist_f64(center, points, d, out, n);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::rowdist(center, points, d, out, n);
        }
        scalar::rowdist(center, points, d, out, n);
    }
    #[inline(always)]
    fn vec_l1dist(v1: &[f64], v2: &[f64], d: usize) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::l1dist_f64(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::l1dist(v1, v2, d);
        }
        scalar::l1dist(v1, v2, d)
    }
    #[inline(always)]
    fn vec_mul(v1: &mut [f64], v2: &[f64], a: f64, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::mul_f64(v1, v2, a, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::mul(v1, v2, a, d);
        }
        scalar::mul(v1, v2, a, d)
    }
    #[inline(always)]
    fn vec_mul_assign(v: &mut [f64], f: f64, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::mul_assign_f64(v, f, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::mul_assign(v, f, d);
        }
        scalar::mul_assign(v, f, d)
    }
    #[inline(always)]
    fn vec_add_assign(v1: &mut [f64], v2: &[f64], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::add_assign_f64(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::add_assign(v1, v2, d);
        }
        scalar::add_assign(v1, v2, d)
    }
    #[inline(always)]
    fn vec_sub_assign(v1: &mut [f64], v2: &[f64], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sub_assign_f64(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sub_assign(v1, v2, d);
        }
        scalar::sub_assign(v1, v2, d)
    }
    #[inline(always)]
    fn vec_fmamul(v1: &mut [f64], a: f64, v2: &[f64], b: f64, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::fmamul_f64(v1, a, v2, b, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::fmamul(v1, a, v2, b, d);
        }
        scalar::fmamul(v1, a, v2, b, d)
    }
    #[inline(always)]
    fn vec_dot(v1: &[f64], v2: &[f64], d: usize) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::dot_f64(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::dot(v1, v2, d);
        }
        scalar::dot(v1, v2, d)
    }
    #[inline(always)]
    fn vec_axpy(v1: &mut [f64], a: f64, v2: &[f64], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::axpy_f64(v1, a, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::axpy(v1, a, v2, d);
        }
        scalar::axpy(v1, a, v2, d)
    }
    #[inline(always)]
    fn vec_sum(v: &[f64], d: usize) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sum_f64(v, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sum(v, d);
        }
        scalar::sum(v, d)
    }
    #[inline(always)]
    fn vec_add_scalar(v: &mut [f64], s: f64, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::add_scalar_f64(v, s, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::add_scalar(v, s, d);
        }
        scalar::add_scalar(v, s, d)
    }
}

// ---------------------------------------------------------------------------
// Public free functions - thin wrappers delegating to VecOps
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two length-`d` vectors.
#[inline(always)]
pub fn sqdist<N: Float>(v1: &[N], v2: &[N], d: usize) -> N { N::vec_sqdist(v1, v2, d) }

/// Pairwise squared distances between two point sets.
///
/// Returns an `nrows*ncols` row-major distance matrix for `points1 x points2`.
#[inline]
pub fn pairwise_sqdist<N: Float, D1: Data<Elem = N>, D2: Data<Elem = N>>(
    points1: &ArrayBase<D1, Ix2>, points2: &ArrayBase<D2, Ix2>,
) -> Vec<N> {
    let nrows = points1.nrows();
    let ncols = points2.nrows();
    let d = points1.ncols();
    debug_assert_eq!(points2.ncols(), d, "point dimensionality mismatch");
    let mut matrix = vec![N::zero(); nrows.checked_mul(ncols).expect("point count overflow")];
    if nrows == 0 || ncols == 0 {
        return matrix;
    }
    N::vec_pairwise_sqdist(points1.view(), points2.view(), d, &mut matrix, nrows, ncols);
    matrix
}

/// L1 (Manhattan) distance between two vectors.
#[inline(always)]
pub fn l1dist<N: Float>(v1: &[N], v2: &[N], d: usize) -> N { N::vec_l1dist(v1, v2, d) }

/// Set `v1[i] = v2[i] * a` for `i` in `0..d`.
#[inline(always)]
pub fn mul<N: Float>(v1: &mut [N], v2: &[N], a: N, d: usize) { N::vec_mul(v1, v2, a, d) }

/// In-place multiply by a scalar: `v[i] *= f`.
#[inline(always)]
pub fn mul_assign<N: Float>(v: &mut [N], f: N, d: usize) { N::vec_mul_assign(v, f, d) }

/// Alias for `mul_assign`.
#[inline(always)]
pub fn scale<N: Float>(v: &mut [N], f: N, d: usize) { N::vec_mul_assign(v, f, d) }

/// In-place addition: `v1[i] += v2[i]`.
#[inline(always)]
pub fn add_assign<N: Float>(v1: &mut [N], v2: &[N], d: usize) { N::vec_add_assign(v1, v2, d) }

/// In-place subtraction: `v1[i] -= v2[i]`.
#[inline(always)]
pub fn sub_assign<N: Float>(v1: &mut [N], v2: &[N], d: usize) { N::vec_sub_assign(v1, v2, d) }

/// FMA followed by a multiplication: `v1 = (v1 * a + v2) * b`.
#[inline(always)]
pub fn fmamul<N: Float>(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) {
    N::vec_fmamul(v1, a, v2, b, d)
}

/// Dot product of two vectors.
#[inline(always)]
pub fn dot<N: Float>(v1: &[N], v2: &[N], d: usize) -> N { N::vec_dot(v1, v2, d) }

/// Squared L2 norm of a vector.
#[inline(always)]
pub fn sqnorm<N: Float>(v: &[N], d: usize) -> N { N::vec_dot(v, v, d) }

/// Euclidean (L2) norm of a vector.
#[inline(always)]
pub fn norm<N: Float>(v: &[N], d: usize) -> N { sqnorm(v, d).sqrt() }

/// In-place scaled addition (AXPY): `v1[i] += a * v2[i]` for `i` in `0..d`.
#[inline(always)]
pub fn axpy<N: Float>(v1: &mut [N], a: N, v2: &[N], d: usize) { N::vec_axpy(v1, a, v2, d) }

/// Combined scaled sum: `v1 := a * v1 + b * v2`.
#[inline(always)]
pub fn axpby<N: Float>(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) {
    N::vec_mul_assign(v1, a, d);
    N::vec_axpy(v1, b, v2, d);
}

/// Compute the sum of the first `d` elements of a slice.
#[inline(always)]
pub fn sum<N: Float>(v: &[N], d: usize) -> N { N::vec_sum(v, d) }

/// Add a scalar to every element: `v[i] += s`.
#[inline(always)]
pub fn add_scalar<N: Float>(v: &mut [N], s: N, d: usize) { N::vec_add_scalar(v, s, d) }

/// Copy `d` elements from `v2` into `v1`.
#[inline(always)]
pub fn copy<N: Copy>(v1: &mut [N], v2: &[N], d: usize) {
    debug_assert!(v1.len() >= d && v2.len() >= d);
    unsafe {
        std::ptr::copy_nonoverlapping(v2.as_ptr(), v1.as_mut_ptr(), d);
    }
}

/// Fill a slice with a constant value.
#[inline(always)]
pub fn fill<N: Copy>(v: &mut [N], val: N, d: usize) {
    debug_assert!(v.len() >= d);
    for i in 0..d {
        unsafe {
            *v.get_unchecked_mut(i) = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, ShapeBuilder};

    use super::*;

    fn squared_matrix<N: Float + std::fmt::Debug + PartialEq>(points: Array2<N>) -> Vec<N> {
        pairwise_sqdist(&points, &points)
    }

    #[test]
    fn pairwise_sqdist_f32_row_major() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0f32, 1.0, 2.0, 3.0, 1.0, 4.0]).unwrap();
        let got = squared_matrix(points);
        let expected = vec![0.0, 8.0, 10.0, 8.0, 0.0, 2.0, 10.0, 2.0, 0.0];
        assert_eq!(got, expected);
    }

    #[test]
    fn pairwise_sqdist_f64_vectorized() {
        let points = Array2::from_shape_vec(
            (3, 4),
            vec![1.0f64, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let got = squared_matrix(points);
        let expected = vec![0.0, 20.0, 30.0, 20.0, 0.0, 30.0, 30.0, 30.0, 0.0];
        assert_eq!(got, expected);
    }

    // Verify vec_row_sqdist produces the same results for C-order, Fortran-order,
    // and the scalar reference for various (n, d) sizes.
    fn check_row_sqdist_layouts<N>(n: usize, d: usize)
    where
        N: Float + std::fmt::Debug + PartialOrd,
    {
        let center: Vec<N> = (0..d).map(|i| N::from_f64(i as f64 * 0.1 + 0.5).unwrap()).collect();
        let data: Vec<N> =
            (0..n * d).map(|i| N::from_f64(i as f64 * 0.03 + 0.01).unwrap()).collect();

        // C-order (standard ndarray layout)
        let points_c = Array2::from_shape_vec((n, d), data.clone()).unwrap();
        // Fortran-order with the same logical values: fill column-by-column so
        // that points_f[[j, k]] == points_c[[j, k]].
        let data_f: Vec<N> = (0..d)
            .flat_map(|k| {
                (0..n).map(move |j| N::from_f64((j * d + k) as f64 * 0.03 + 0.01).unwrap())
            })
            .collect();
        let points_f = Array2::from_shape_vec((n, d).f(), data_f).unwrap();

        let mut out_c = vec![N::zero(); n];
        let mut out_f = vec![N::zero(); n];
        N::vec_row_sqdist(&center, points_c.view(), d, &mut out_c, n);
        N::vec_row_sqdist(&center, points_f.view(), d, &mut out_f, n);

        let rel_eps = N::from_f64(1e-5).unwrap();
        for j in 0..n {
            let expected: N = (0..d)
                .map(|k| {
                    let diff = center[k] - points_c[[j, k]];
                    diff * diff
                })
                .sum();
            // Use relative tolerance: abs_err / max(expected, 1) < rel_eps
            let scale = if expected > N::one() { expected } else { N::one() };
            let err_c = if out_c[j] > expected { out_c[j] - expected } else { expected - out_c[j] };
            let err_f = if out_f[j] > expected { out_f[j] - expected } else { expected - out_f[j] };
            assert!(
                err_c / scale < rel_eps,
                "C-order j={j}: got {:?}, expected {:?}",
                out_c[j],
                expected
            );
            assert!(
                err_f / scale < rel_eps,
                "Fortran j={j}: got {:?}, expected {:?}",
                out_f[j],
                expected
            );
        }
    }

    #[test]
    fn vec_row_sqdist_layouts_f64() {
        check_row_sqdist_layouts::<f64>(64, 32);
        check_row_sqdist_layouts::<f64>(65, 32); // tail block (65 % NR_F64 != 0)
        check_row_sqdist_layouts::<f64>(13, 7); // n < NR_F64, odd d
        check_row_sqdist_layouts::<f64>(100, 16);
    }

    #[test]
    fn vec_row_sqdist_layouts_f32() {
        check_row_sqdist_layouts::<f32>(64, 32);
        check_row_sqdist_layouts::<f32>(65, 32); // tail block (65 % NR_F32 != 0)
        check_row_sqdist_layouts::<f32>(9, 5); // n < NR_F32, odd d
        check_row_sqdist_layouts::<f32>(100, 16);
    }

    fn check_row_sqdist_strided<N>(n: usize, d: usize)
    where
        N: Float + std::fmt::Debug + PartialOrd,
    {
        let center: Vec<N> = (0..d).map(|i| N::from_f64(i as f64 * 0.2 + 0.25).unwrap()).collect();
        let data: Vec<N> =
            (0..n * d * 2).map(|i| N::from_f64(i as f64 * 0.01 + 0.03).unwrap()).collect();
        let base = Array2::from_shape_vec((n, d * 2), data).unwrap();
        let points = base.slice(ndarray::s![.., ..;2]);

        let mut out = vec![N::zero(); n];
        N::vec_row_sqdist(&center, points, d, &mut out, n);

        for j in 0..n {
            let expected: N = (0..d)
                .map(|k| {
                    let diff = center[k] - points[[j, k]];
                    diff * diff
                })
                .sum();
            let err = if out[j] > expected { out[j] - expected } else { expected - out[j] };
            let scale = if expected > N::one() { expected } else { N::one() };
            assert!(
                err / scale < N::from_f64(1e-5).unwrap(),
                "strided j={j}: got {:?}, expected {:?}",
                out[j],
                expected
            );
        }
    }

    #[test]
    fn vec_row_sqdist_strided_f64() {
        check_row_sqdist_strided::<f64>(17, 11);
        check_row_sqdist_strided::<f64>(64, 5);
    }

    #[test]
    fn vec_row_sqdist_strided_f32() {
        check_row_sqdist_strided::<f32>(17, 11);
        check_row_sqdist_strided::<f32>(64, 5);
    }
}
