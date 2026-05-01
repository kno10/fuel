//! AVX2-accelerated implementations of the vector math primitives.
//!
//! Concrete `_f32` / `_f64` free functions; no generics, no `TypeId`.
//! Called from `vecops.rs` which provides the `VecOps` trait impls.
//! Assumes AVX2 and FMA are available (compile with `-C target-feature=+avx2,+fma`
//! or `-C target-cpu=native`).

use std::arch::x86_64::*;
use std::cell::RefCell;

use ndarray::ArrayView2;

thread_local! {
    static ROW_PANEL_F32: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static ROW_PANEL_F64: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

// Tile sizes for the pairwise-sqdist micro-kernels.
// NR = number of j-columns covered by one YMM register (8 f32 / 4 f64).
// MR_SDIST = number of i-rows in the register tile; limited by available
// accumulator registers after reserving registers for b and diff temporaries.
const NR_F32: usize = 8;
const MR_SDIST_F32: usize = 8; // 8 accumulators; b(1)+d(1)+acc0..7(8) = 10 YMM used
const NR_F64: usize = 4;
const MR_SDIST_F64: usize = 4; // 4 accumulators; b(1)+d(1)+acc0..3(4) = 6 YMM used
// Prefetch lookahead in units of 4-step unroll iterations.
const PREFETCH_DIST: usize = 12;

#[inline(always)]
fn pack_panel_f32(
    panel: &mut [f32], points: ArrayView2<'_, f32>, jj: usize, nc: usize, d: usize,
) {
    debug_assert!(panel.len() >= NR_F32 * d);

    for j_local in 0..nc {
        let row = points.row(jj + j_local);
        for k in 0..d {
            panel[k * NR_F32 + j_local] = row[k];
        }
    }

    if nc < NR_F32 {
        for k in 0..d {
            panel[k * NR_F32 + nc..k * NR_F32 + NR_F32].fill(0.0);
        }
    }
}

#[inline(always)]
fn pack_panel_f64(
    panel: &mut [f64], points: ArrayView2<'_, f64>, jj: usize, nc: usize, d: usize,
) {
    debug_assert!(panel.len() >= NR_F64 * d);

    for j_local in 0..nc {
        let row = points.row(jj + j_local);
        for k in 0..d {
            panel[k * NR_F64 + j_local] = row[k];
        }
    }

    if nc < NR_F64 {
        for k in 0..d {
            panel[k * NR_F64 + nc..k * NR_F64 + NR_F64].fill(0.0);
        }
    }
}

// ----- horizontal-sum helpers ------------------------------------------------

/// Horizontal sum of all 8 lanes of a __m256 into a single f32.
#[inline(always)]
unsafe fn horizontal_sum_f32(v: __m256) -> f32 {
    // Add upper and lower 128-bit halves: [v0+v4, v1+v5, v2+v6, v3+v7]
    let lo = unsafe { _mm256_castps256_ps128(v) };
    let hi = unsafe { _mm256_extractf128_ps(v, 1) };
    let s128 = unsafe { _mm_add_ps(lo, hi) };
    // Shuffle: [s1, s1, s3, s3], then add pairs -> s64[0] = s0+s1, s64[2] = s2+s3
    let shuf = unsafe { _mm_movehdup_ps(s128) };
    let s64 = unsafe { _mm_add_ps(s128, shuf) };
    // Move s64[2] into lane 0 and add -> final sum
    let hi64 = unsafe { _mm_movehl_ps(s64, s64) };
    let s32 = unsafe { _mm_add_ss(s64, hi64) };
    unsafe { _mm_cvtss_f32(s32) }
}

/// Horizontal sum of all 4 lanes of a __m256d into a single f64.
#[inline(always)]
unsafe fn horizontal_sum_f64(v: __m256d) -> f64 {
    // Add upper and lower 128-bit halves: [v0+v2, v1+v3]
    let lo = unsafe { _mm256_castpd256_pd128(v) };
    let hi = unsafe { _mm256_extractf128_pd(v, 1) };
    let s128 = unsafe { _mm_add_pd(lo, hi) };
    // Add the two remaining elements
    let hi64 = unsafe { _mm_unpackhi_pd(s128, s128) };
    let s64 = unsafe { _mm_add_sd(s128, hi64) };
    unsafe { _mm_cvtsd_f64(s64) }
}

// ----- sqdist ----------------------------------------------------------------

/// Squared Euclidean distance between two f32 slices of length `d`.
///
/// Uses 4 independent accumulators for d>=32 to hide the 4-cycle FMA latency,
/// then a register-only horizontal reduction instead of a stack spill+sum.
#[inline(always)]
pub(super) fn sqdist_f32(v1: &[f32], v2: &[f32], d: usize) -> f32 {
    let mut i = 0;
    let mut acc0 = unsafe { _mm256_setzero_ps() };
    let mut acc1 = unsafe { _mm256_setzero_ps() };
    let mut acc2 = unsafe { _mm256_setzero_ps() };
    let mut acc3 = unsafe { _mm256_setzero_ps() };
    // 4-accumulator unrolled loop to expose FMA-level ILP for d >= 32.
    while i + 32 <= d {
        unsafe {
            let a0 = _mm256_loadu_ps(v1.as_ptr().add(i));
            let b0 = _mm256_loadu_ps(v2.as_ptr().add(i));
            let d0 = _mm256_sub_ps(a0, b0);
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);

            let a1 = _mm256_loadu_ps(v1.as_ptr().add(i + 8));
            let b1 = _mm256_loadu_ps(v2.as_ptr().add(i + 8));
            let d1 = _mm256_sub_ps(a1, b1);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);

            let a2 = _mm256_loadu_ps(v1.as_ptr().add(i + 16));
            let b2 = _mm256_loadu_ps(v2.as_ptr().add(i + 16));
            let d2 = _mm256_sub_ps(a2, b2);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);

            let a3 = _mm256_loadu_ps(v1.as_ptr().add(i + 24));
            let b3 = _mm256_loadu_ps(v2.as_ptr().add(i + 24));
            let d3 = _mm256_sub_ps(a3, b3);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
        }
        i += 32;
    }
    let mut acc = unsafe {
        _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3))
    };
    while i + 8 <= d {
        unsafe {
            let a = _mm256_loadu_ps(v1.as_ptr().add(i));
            let b = _mm256_loadu_ps(v2.as_ptr().add(i));
            let diff = _mm256_sub_ps(a, b);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }
        i += 8;
    }
    let mut sum = unsafe { horizontal_sum_f32(acc) };
    while i < d {
        let x = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
        sum = x.mul_add(x, sum);
        i += 1;
    }
    sum
}

/// Squared Euclidean distance between two f64 slices of length `d`.
///
/// Uses 4 independent accumulators for d>=16 to hide the 4-cycle FMA latency,
/// then a register-only horizontal reduction.
#[inline(always)]
pub(super) fn sqdist_f64(v1: &[f64], v2: &[f64], d: usize) -> f64 {
    let mut i = 0;
    let mut acc0 = unsafe { _mm256_setzero_pd() };
    let mut acc1 = unsafe { _mm256_setzero_pd() };
    let mut acc2 = unsafe { _mm256_setzero_pd() };
    let mut acc3 = unsafe { _mm256_setzero_pd() };
    while i + 16 <= d {
        unsafe {
            let a0 = _mm256_loadu_pd(v1.as_ptr().add(i));
            let b0 = _mm256_loadu_pd(v2.as_ptr().add(i));
            let d0 = _mm256_sub_pd(a0, b0);
            acc0 = _mm256_fmadd_pd(d0, d0, acc0);

            let a1 = _mm256_loadu_pd(v1.as_ptr().add(i + 4));
            let b1 = _mm256_loadu_pd(v2.as_ptr().add(i + 4));
            let d1 = _mm256_sub_pd(a1, b1);
            acc1 = _mm256_fmadd_pd(d1, d1, acc1);

            let a2 = _mm256_loadu_pd(v1.as_ptr().add(i + 8));
            let b2 = _mm256_loadu_pd(v2.as_ptr().add(i + 8));
            let d2 = _mm256_sub_pd(a2, b2);
            acc2 = _mm256_fmadd_pd(d2, d2, acc2);

            let a3 = _mm256_loadu_pd(v1.as_ptr().add(i + 12));
            let b3 = _mm256_loadu_pd(v2.as_ptr().add(i + 12));
            let d3 = _mm256_sub_pd(a3, b3);
            acc3 = _mm256_fmadd_pd(d3, d3, acc3);
        }
        i += 16;
    }
    let mut acc = unsafe {
        _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3))
    };
    while i + 4 <= d {
        unsafe {
            let a = _mm256_loadu_pd(v1.as_ptr().add(i));
            let b = _mm256_loadu_pd(v2.as_ptr().add(i));
            let diff = _mm256_sub_pd(a, b);
            acc = _mm256_fmadd_pd(diff, diff, acc);
        }
        i += 4;
    }
    let mut sum = unsafe { horizontal_sum_f64(acc) };
    while i < d {
        let x = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
        sum = x.mul_add(x, sum);
        i += 1;
    }
    sum
}

// ----- l1dist ----------------------------------------------------------------

/// Manhattan distance between two f32 slices of length `d`.
///
/// 4 independent accumulators hide the 1-cycle ADD latency and allow
/// two-ADD-per-cycle throughput; horizontal reduction via register shuffle.
#[inline(always)]
pub(super) fn l1dist_f32(v1: &[f32], v2: &[f32], d: usize) -> f32 {
    let mut i = 0;
    let abs_mask = unsafe { _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff)) };
    let mut acc0 = unsafe { _mm256_setzero_ps() };
    let mut acc1 = unsafe { _mm256_setzero_ps() };
    let mut acc2 = unsafe { _mm256_setzero_ps() };
    let mut acc3 = unsafe { _mm256_setzero_ps() };
    while i + 32 <= d {
        unsafe {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(v1.as_ptr().add(i)), _mm256_loadu_ps(v2.as_ptr().add(i)));
            acc0 = _mm256_add_ps(acc0, _mm256_and_ps(d0, abs_mask));
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(v1.as_ptr().add(i + 8)), _mm256_loadu_ps(v2.as_ptr().add(i + 8)));
            acc1 = _mm256_add_ps(acc1, _mm256_and_ps(d1, abs_mask));
            let d2 = _mm256_sub_ps(_mm256_loadu_ps(v1.as_ptr().add(i + 16)), _mm256_loadu_ps(v2.as_ptr().add(i + 16)));
            acc2 = _mm256_add_ps(acc2, _mm256_and_ps(d2, abs_mask));
            let d3 = _mm256_sub_ps(_mm256_loadu_ps(v1.as_ptr().add(i + 24)), _mm256_loadu_ps(v2.as_ptr().add(i + 24)));
            acc3 = _mm256_add_ps(acc3, _mm256_and_ps(d3, abs_mask));
        }
        i += 32;
    }
    let mut acc = unsafe {
        _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3))
    };
    while i + 8 <= d {
        unsafe {
            let diff = _mm256_sub_ps(_mm256_loadu_ps(v1.as_ptr().add(i)), _mm256_loadu_ps(v2.as_ptr().add(i)));
            acc = _mm256_add_ps(acc, _mm256_and_ps(diff, abs_mask));
        }
        i += 8;
    }
    let mut sum = unsafe { horizontal_sum_f32(acc) };
    while i < d {
        sum += (unsafe { *v1.get_unchecked(i) } - unsafe { *v2.get_unchecked(i) }).abs();
        i += 1;
    }
    sum
}

/// Manhattan distance between two f64 slices of length `d`.
#[inline(always)]
pub(super) fn l1dist_f64(v1: &[f64], v2: &[f64], d: usize) -> f64 {
    let mut i = 0;
    let abs_mask = unsafe { _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fff_ffff_ffff_ffff)) };
    let mut acc0 = unsafe { _mm256_setzero_pd() };
    let mut acc1 = unsafe { _mm256_setzero_pd() };
    let mut acc2 = unsafe { _mm256_setzero_pd() };
    let mut acc3 = unsafe { _mm256_setzero_pd() };
    while i + 16 <= d {
        unsafe {
            let d0 = _mm256_sub_pd(_mm256_loadu_pd(v1.as_ptr().add(i)), _mm256_loadu_pd(v2.as_ptr().add(i)));
            acc0 = _mm256_add_pd(acc0, _mm256_and_pd(d0, abs_mask));
            let d1 = _mm256_sub_pd(_mm256_loadu_pd(v1.as_ptr().add(i + 4)), _mm256_loadu_pd(v2.as_ptr().add(i + 4)));
            acc1 = _mm256_add_pd(acc1, _mm256_and_pd(d1, abs_mask));
            let d2 = _mm256_sub_pd(_mm256_loadu_pd(v1.as_ptr().add(i + 8)), _mm256_loadu_pd(v2.as_ptr().add(i + 8)));
            acc2 = _mm256_add_pd(acc2, _mm256_and_pd(d2, abs_mask));
            let d3 = _mm256_sub_pd(_mm256_loadu_pd(v1.as_ptr().add(i + 12)), _mm256_loadu_pd(v2.as_ptr().add(i + 12)));
            acc3 = _mm256_add_pd(acc3, _mm256_and_pd(d3, abs_mask));
        }
        i += 16;
    }
    let mut acc = unsafe {
        _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3))
    };
    while i + 4 <= d {
        unsafe {
            let diff = _mm256_sub_pd(_mm256_loadu_pd(v1.as_ptr().add(i)), _mm256_loadu_pd(v2.as_ptr().add(i)));
            acc = _mm256_add_pd(acc, _mm256_and_pd(diff, abs_mask));
        }
        i += 4;
    }
    let mut sum = unsafe { horizontal_sum_f64(acc) };
    while i < d {
        sum += (unsafe { *v1.get_unchecked(i) } - unsafe { *v2.get_unchecked(i) }).abs();
        i += 1;
    }
    sum
}

// ----- mul (v1 = v2 * a) -----------------------------------------------------

#[inline(always)]
pub(super) fn mul_f32(v1: &mut [f32], v2: &[f32], a: f32, d: usize) {
    let scalar = unsafe { _mm256_set1_ps(a) };
    let mut i = 0;
    while i + 32 <= d {
        unsafe {
            _mm256_storeu_ps(v1.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_loadu_ps(v2.as_ptr().add(i)), scalar));
            _mm256_storeu_ps(v1.as_mut_ptr().add(i + 8), _mm256_mul_ps(_mm256_loadu_ps(v2.as_ptr().add(i + 8)), scalar));
            _mm256_storeu_ps(v1.as_mut_ptr().add(i + 16), _mm256_mul_ps(_mm256_loadu_ps(v2.as_ptr().add(i + 16)), scalar));
            _mm256_storeu_ps(v1.as_mut_ptr().add(i + 24), _mm256_mul_ps(_mm256_loadu_ps(v2.as_ptr().add(i + 24)), scalar));
        }
        i += 32;
    }
    while i + 8 <= d {
        unsafe {
            _mm256_storeu_ps(v1.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_loadu_ps(v2.as_ptr().add(i)), scalar));
        }
        i += 8;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a };
        i += 1;
    }
}

#[inline(always)]
pub(super) fn mul_f64(v1: &mut [f64], v2: &[f64], a: f64, d: usize) {
    let scalar = unsafe { _mm256_set1_pd(a) };
    let mut i = 0;
    while i + 16 <= d {
        unsafe {
            _mm256_storeu_pd(v1.as_mut_ptr().add(i), _mm256_mul_pd(_mm256_loadu_pd(v2.as_ptr().add(i)), scalar));
            _mm256_storeu_pd(v1.as_mut_ptr().add(i + 4), _mm256_mul_pd(_mm256_loadu_pd(v2.as_ptr().add(i + 4)), scalar));
            _mm256_storeu_pd(v1.as_mut_ptr().add(i + 8), _mm256_mul_pd(_mm256_loadu_pd(v2.as_ptr().add(i + 8)), scalar));
            _mm256_storeu_pd(v1.as_mut_ptr().add(i + 12), _mm256_mul_pd(_mm256_loadu_pd(v2.as_ptr().add(i + 12)), scalar));
        }
        i += 16;
    }
    while i + 4 <= d {
        unsafe {
            _mm256_storeu_pd(v1.as_mut_ptr().add(i), _mm256_mul_pd(_mm256_loadu_pd(v2.as_ptr().add(i)), scalar));
        }
        i += 4;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a };
        i += 1;
    }
}

// ----- mul_assign (v *= f) ---------------------------------------------------

#[inline(always)]
pub(super) fn mul_assign_f32(v: &mut [f32], f: f32, d: usize) {
    let scalar = unsafe { _mm256_set1_ps(f) };
    let mut i = 0;
    while i + 32 <= d {
        unsafe {
            let p0 = v.as_mut_ptr().add(i);
            _mm256_storeu_ps(p0, _mm256_mul_ps(_mm256_loadu_ps(p0), scalar));
            let p1 = v.as_mut_ptr().add(i + 8);
            _mm256_storeu_ps(p1, _mm256_mul_ps(_mm256_loadu_ps(p1), scalar));
            let p2 = v.as_mut_ptr().add(i + 16);
            _mm256_storeu_ps(p2, _mm256_mul_ps(_mm256_loadu_ps(p2), scalar));
            let p3 = v.as_mut_ptr().add(i + 24);
            _mm256_storeu_ps(p3, _mm256_mul_ps(_mm256_loadu_ps(p3), scalar));
        }
        i += 32;
    }
    while i + 8 <= d {
        unsafe {
            let ptr = v.as_mut_ptr().add(i);
            _mm256_storeu_ps(ptr, _mm256_mul_ps(_mm256_loadu_ps(ptr), scalar));
        }
        i += 8;
    }
    while i < d {
        unsafe { *v.get_unchecked_mut(i) *= f };
        i += 1;
    }
}

#[inline(always)]
pub(super) fn mul_assign_f64(v: &mut [f64], f: f64, d: usize) {
    let scalar = unsafe { _mm256_set1_pd(f) };
    let mut i = 0;
    while i + 16 <= d {
        unsafe {
            let p0 = v.as_mut_ptr().add(i);
            _mm256_storeu_pd(p0, _mm256_mul_pd(_mm256_loadu_pd(p0), scalar));
            let p1 = v.as_mut_ptr().add(i + 4);
            _mm256_storeu_pd(p1, _mm256_mul_pd(_mm256_loadu_pd(p1), scalar));
            let p2 = v.as_mut_ptr().add(i + 8);
            _mm256_storeu_pd(p2, _mm256_mul_pd(_mm256_loadu_pd(p2), scalar));
            let p3 = v.as_mut_ptr().add(i + 12);
            _mm256_storeu_pd(p3, _mm256_mul_pd(_mm256_loadu_pd(p3), scalar));
        }
        i += 16;
    }
    while i + 4 <= d {
        unsafe {
            let ptr = v.as_mut_ptr().add(i);
            _mm256_storeu_pd(ptr, _mm256_mul_pd(_mm256_loadu_pd(ptr), scalar));
        }
        i += 4;
    }
    while i < d {
        unsafe { *v.get_unchecked_mut(i) *= f };
        i += 1;
    }
}

// ----- add_assign (v1 += v2) -------------------------------------------------

#[inline(always)]
pub(super) fn add_assign_f32(v1: &mut [f32], v2: &[f32], d: usize) {
    let mut i = 0;
    while i + 32 <= d {
        unsafe {
            let p0 = v1.as_mut_ptr().add(i);
            _mm256_storeu_ps(p0, _mm256_add_ps(_mm256_loadu_ps(p0), _mm256_loadu_ps(v2.as_ptr().add(i))));
            let p1 = v1.as_mut_ptr().add(i + 8);
            _mm256_storeu_ps(p1, _mm256_add_ps(_mm256_loadu_ps(p1), _mm256_loadu_ps(v2.as_ptr().add(i + 8))));
            let p2 = v1.as_mut_ptr().add(i + 16);
            _mm256_storeu_ps(p2, _mm256_add_ps(_mm256_loadu_ps(p2), _mm256_loadu_ps(v2.as_ptr().add(i + 16))));
            let p3 = v1.as_mut_ptr().add(i + 24);
            _mm256_storeu_ps(p3, _mm256_add_ps(_mm256_loadu_ps(p3), _mm256_loadu_ps(v2.as_ptr().add(i + 24))));
        }
        i += 32;
    }
    while i + 8 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_ps(ptr1, _mm256_add_ps(_mm256_loadu_ps(ptr1), _mm256_loadu_ps(v2.as_ptr().add(i))));
        }
        i += 8;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) += *v2.get_unchecked(i) };
        i += 1;
    }
}

#[inline(always)]
pub(super) fn add_assign_f64(v1: &mut [f64], v2: &[f64], d: usize) {
    let mut i = 0;
    while i + 16 <= d {
        unsafe {
            let p0 = v1.as_mut_ptr().add(i);
            _mm256_storeu_pd(p0, _mm256_add_pd(_mm256_loadu_pd(p0), _mm256_loadu_pd(v2.as_ptr().add(i))));
            let p1 = v1.as_mut_ptr().add(i + 4);
            _mm256_storeu_pd(p1, _mm256_add_pd(_mm256_loadu_pd(p1), _mm256_loadu_pd(v2.as_ptr().add(i + 4))));
            let p2 = v1.as_mut_ptr().add(i + 8);
            _mm256_storeu_pd(p2, _mm256_add_pd(_mm256_loadu_pd(p2), _mm256_loadu_pd(v2.as_ptr().add(i + 8))));
            let p3 = v1.as_mut_ptr().add(i + 12);
            _mm256_storeu_pd(p3, _mm256_add_pd(_mm256_loadu_pd(p3), _mm256_loadu_pd(v2.as_ptr().add(i + 12))));
        }
        i += 16;
    }
    while i + 4 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_pd(ptr1, _mm256_add_pd(_mm256_loadu_pd(ptr1), _mm256_loadu_pd(v2.as_ptr().add(i))));
        }
        i += 4;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) += *v2.get_unchecked(i) };
        i += 1;
    }
}

// ----- sub_assign (v1 -= v2) -------------------------------------------------

#[inline(always)]
pub(super) fn sub_assign_f32(v1: &mut [f32], v2: &[f32], d: usize) {
    let mut i = 0;
    while i + 32 <= d {
        unsafe {
            let p0 = v1.as_mut_ptr().add(i);
            _mm256_storeu_ps(p0, _mm256_sub_ps(_mm256_loadu_ps(p0), _mm256_loadu_ps(v2.as_ptr().add(i))));
            let p1 = v1.as_mut_ptr().add(i + 8);
            _mm256_storeu_ps(p1, _mm256_sub_ps(_mm256_loadu_ps(p1), _mm256_loadu_ps(v2.as_ptr().add(i + 8))));
            let p2 = v1.as_mut_ptr().add(i + 16);
            _mm256_storeu_ps(p2, _mm256_sub_ps(_mm256_loadu_ps(p2), _mm256_loadu_ps(v2.as_ptr().add(i + 16))));
            let p3 = v1.as_mut_ptr().add(i + 24);
            _mm256_storeu_ps(p3, _mm256_sub_ps(_mm256_loadu_ps(p3), _mm256_loadu_ps(v2.as_ptr().add(i + 24))));
        }
        i += 32;
    }
    while i + 8 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_ps(ptr1, _mm256_sub_ps(_mm256_loadu_ps(ptr1), _mm256_loadu_ps(v2.as_ptr().add(i))));
        }
        i += 8;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i) };
        i += 1;
    }
}

#[inline(always)]
pub(super) fn sub_assign_f64(v1: &mut [f64], v2: &[f64], d: usize) {
    let mut i = 0;
    while i + 16 <= d {
        unsafe {
            let p0 = v1.as_mut_ptr().add(i);
            _mm256_storeu_pd(p0, _mm256_sub_pd(_mm256_loadu_pd(p0), _mm256_loadu_pd(v2.as_ptr().add(i))));
            let p1 = v1.as_mut_ptr().add(i + 4);
            _mm256_storeu_pd(p1, _mm256_sub_pd(_mm256_loadu_pd(p1), _mm256_loadu_pd(v2.as_ptr().add(i + 4))));
            let p2 = v1.as_mut_ptr().add(i + 8);
            _mm256_storeu_pd(p2, _mm256_sub_pd(_mm256_loadu_pd(p2), _mm256_loadu_pd(v2.as_ptr().add(i + 8))));
            let p3 = v1.as_mut_ptr().add(i + 12);
            _mm256_storeu_pd(p3, _mm256_sub_pd(_mm256_loadu_pd(p3), _mm256_loadu_pd(v2.as_ptr().add(i + 12))));
        }
        i += 16;
    }
    while i + 4 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_pd(ptr1, _mm256_sub_pd(_mm256_loadu_pd(ptr1), _mm256_loadu_pd(v2.as_ptr().add(i))));
        }
        i += 4;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i) };
        i += 1;
    }
}

// ----- fmamul (v1 = (v1*a + v2)*b) ------------------------------------------

#[inline(always)]
pub(super) fn fmamul_f32(v1: &mut [f32], a: f32, v2: &[f32], b: f32, d: usize) {
    let va = unsafe { _mm256_set1_ps(a) };
    let vb = unsafe { _mm256_set1_ps(b) };
    let mut i = 0;
    while i + 32 <= d {
        unsafe {
            let p0 = v1.as_mut_ptr().add(i);
            _mm256_storeu_ps(p0, _mm256_mul_ps(_mm256_fmadd_ps(_mm256_loadu_ps(p0), va, _mm256_loadu_ps(v2.as_ptr().add(i))), vb));
            let p1 = v1.as_mut_ptr().add(i + 8);
            _mm256_storeu_ps(p1, _mm256_mul_ps(_mm256_fmadd_ps(_mm256_loadu_ps(p1), va, _mm256_loadu_ps(v2.as_ptr().add(i + 8))), vb));
            let p2 = v1.as_mut_ptr().add(i + 16);
            _mm256_storeu_ps(p2, _mm256_mul_ps(_mm256_fmadd_ps(_mm256_loadu_ps(p2), va, _mm256_loadu_ps(v2.as_ptr().add(i + 16))), vb));
            let p3 = v1.as_mut_ptr().add(i + 24);
            _mm256_storeu_ps(p3, _mm256_mul_ps(_mm256_fmadd_ps(_mm256_loadu_ps(p3), va, _mm256_loadu_ps(v2.as_ptr().add(i + 24))), vb));
        }
        i += 32;
    }
    while i + 8 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            let x = _mm256_fmadd_ps(_mm256_loadu_ps(ptr1), va, _mm256_loadu_ps(v2.as_ptr().add(i)));
            _mm256_storeu_ps(ptr1, _mm256_mul_ps(x, vb));
        }
        i += 8;
    }
    while i < d {
        unsafe {
            *v1.get_unchecked_mut(i) = v1.get_unchecked(i).mul_add(a, *v2.get_unchecked(i)) * b;
        }
        i += 1;
    }
}

#[inline(always)]
pub(super) fn fmamul_f64(v1: &mut [f64], a: f64, v2: &[f64], b: f64, d: usize) {
    let va = unsafe { _mm256_set1_pd(a) };
    let vb = unsafe { _mm256_set1_pd(b) };
    let mut i = 0;
    while i + 16 <= d {
        unsafe {
            let p0 = v1.as_mut_ptr().add(i);
            _mm256_storeu_pd(p0, _mm256_mul_pd(_mm256_fmadd_pd(_mm256_loadu_pd(p0), va, _mm256_loadu_pd(v2.as_ptr().add(i))), vb));
            let p1 = v1.as_mut_ptr().add(i + 4);
            _mm256_storeu_pd(p1, _mm256_mul_pd(_mm256_fmadd_pd(_mm256_loadu_pd(p1), va, _mm256_loadu_pd(v2.as_ptr().add(i + 4))), vb));
            let p2 = v1.as_mut_ptr().add(i + 8);
            _mm256_storeu_pd(p2, _mm256_mul_pd(_mm256_fmadd_pd(_mm256_loadu_pd(p2), va, _mm256_loadu_pd(v2.as_ptr().add(i + 8))), vb));
            let p3 = v1.as_mut_ptr().add(i + 12);
            _mm256_storeu_pd(p3, _mm256_mul_pd(_mm256_fmadd_pd(_mm256_loadu_pd(p3), va, _mm256_loadu_pd(v2.as_ptr().add(i + 12))), vb));
        }
        i += 16;
    }
    while i + 4 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            let x = _mm256_fmadd_pd(_mm256_loadu_pd(ptr1), va, _mm256_loadu_pd(v2.as_ptr().add(i)));
            _mm256_storeu_pd(ptr1, _mm256_mul_pd(x, vb));
        }
        i += 4;
    }
    while i < d {
        unsafe {
            *v1.get_unchecked_mut(i) = v1.get_unchecked(i).mul_add(a, *v2.get_unchecked(i)) * b;
        }
        i += 1;
    }
}

// ----- dot -------------------------------------------------------------------

#[inline(always)]
pub(super) fn dot_f32(v1: &[f32], v2: &[f32], d: usize) -> f32 {
    let mut i = 0;
    let mut acc0 = unsafe { _mm256_setzero_ps() };
    let mut acc1 = unsafe { _mm256_setzero_ps() };
    let mut acc2 = unsafe { _mm256_setzero_ps() };
    let mut acc3 = unsafe { _mm256_setzero_ps() };
    while i + 32 <= d {
        unsafe {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(v1.as_ptr().add(i)),
                _mm256_loadu_ps(v2.as_ptr().add(i)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(v1.as_ptr().add(i + 8)),
                _mm256_loadu_ps(v2.as_ptr().add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(v1.as_ptr().add(i + 16)),
                _mm256_loadu_ps(v2.as_ptr().add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(v1.as_ptr().add(i + 24)),
                _mm256_loadu_ps(v2.as_ptr().add(i + 24)),
                acc3,
            );
        }
        i += 32;
    }
    let mut acc = unsafe {
        _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3))
    };
    while i + 8 <= d {
        unsafe {
            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(v1.as_ptr().add(i)),
                _mm256_loadu_ps(v2.as_ptr().add(i)),
                acc,
            );
        }
        i += 8;
    }
    let mut sum = unsafe { horizontal_sum_f32(acc) };
    while i < d {
        sum = unsafe { v1.get_unchecked(i).mul_add(*v2.get_unchecked(i), sum) };
        i += 1;
    }
    sum
}

#[inline(always)]
pub(super) fn dot_f64(v1: &[f64], v2: &[f64], d: usize) -> f64 {
    let mut i = 0;
    let mut acc0 = unsafe { _mm256_setzero_pd() };
    let mut acc1 = unsafe { _mm256_setzero_pd() };
    let mut acc2 = unsafe { _mm256_setzero_pd() };
    let mut acc3 = unsafe { _mm256_setzero_pd() };
    while i + 16 <= d {
        unsafe {
            acc0 = _mm256_fmadd_pd(
                _mm256_loadu_pd(v1.as_ptr().add(i)),
                _mm256_loadu_pd(v2.as_ptr().add(i)),
                acc0,
            );
            acc1 = _mm256_fmadd_pd(
                _mm256_loadu_pd(v1.as_ptr().add(i + 4)),
                _mm256_loadu_pd(v2.as_ptr().add(i + 4)),
                acc1,
            );
            acc2 = _mm256_fmadd_pd(
                _mm256_loadu_pd(v1.as_ptr().add(i + 8)),
                _mm256_loadu_pd(v2.as_ptr().add(i + 8)),
                acc2,
            );
            acc3 = _mm256_fmadd_pd(
                _mm256_loadu_pd(v1.as_ptr().add(i + 12)),
                _mm256_loadu_pd(v2.as_ptr().add(i + 12)),
                acc3,
            );
        }
        i += 16;
    }
    let mut acc = unsafe {
        _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3))
    };
    while i + 4 <= d {
        unsafe {
            acc = _mm256_fmadd_pd(
                _mm256_loadu_pd(v1.as_ptr().add(i)),
                _mm256_loadu_pd(v2.as_ptr().add(i)),
                acc,
            );
        }
        i += 4;
    }
    let mut sum = unsafe { horizontal_sum_f64(acc) };
    while i < d {
        sum = unsafe { v1.get_unchecked(i).mul_add(*v2.get_unchecked(i), sum) };
        i += 1;
    }
    sum
}

// ----- pairwise-sqdist micro-kernels -----------------------------------------
//
// Layout convention (shared by both):
//   a_packed[k * MR + i_local] = points1[ii_block*MR + i_local][k]
//   b_panel [k * NR + j_local] = points2[jj_block*NR + j_local][k]
//
// Each kernel computes the MR x NR squared-distance tile directly, accumulating
// (a[k] - b[k])^2 per lane.  No norm decomposition -- avoids cancellation.

// f32: 8 output rows x 8 output cols = 8x8 register tile.
// Each acc_i is one __m256 accumulating sqdist for row i vs all 8 j-columns.
// b is loaded once per k-step; each of 8 a-values is broadcast+sub+fmadd in
// turn, keeping register pressure at b(1)+d(1)+acc0..7(8) = 10 YMM.
// 4-way k-unroll with software prefetch.
#[inline(always)]
unsafe fn micro_sqdist_f32(d: usize, a_packed: *const f32, b_packed: *const f32, out: *mut f32) {
    // MR=8 rows, NR=8 cols
    let mut acc0 = unsafe { _mm256_setzero_ps() };
    let mut acc1 = unsafe { _mm256_setzero_ps() };
    let mut acc2 = unsafe { _mm256_setzero_ps() };
    let mut acc3 = unsafe { _mm256_setzero_ps() };
    let mut acc4 = unsafe { _mm256_setzero_ps() };
    let mut acc5 = unsafe { _mm256_setzero_ps() };
    let mut acc6 = unsafe { _mm256_setzero_ps() };
    let mut acc7 = unsafe { _mm256_setzero_ps() };

    // One k-step: load 8 b-values (one YMM), then for each of 8 a-values:
    // broadcast, subtract from b, fmadd diff^2 into the row accumulator.
    macro_rules! step {
        ($offset:expr) => {{
            unsafe {
                let b = _mm256_loadu_ps(b_packed.add($offset * NR_F32));
                let ap = a_packed.add($offset * MR_SDIST_F32);
                let d0 = _mm256_sub_ps(_mm256_broadcast_ss(&*ap), b);
                acc0 = _mm256_fmadd_ps(d0, d0, acc0);
                let d1 = _mm256_sub_ps(_mm256_broadcast_ss(&*ap.add(1)), b);
                acc1 = _mm256_fmadd_ps(d1, d1, acc1);
                let d2 = _mm256_sub_ps(_mm256_broadcast_ss(&*ap.add(2)), b);
                acc2 = _mm256_fmadd_ps(d2, d2, acc2);
                let d3 = _mm256_sub_ps(_mm256_broadcast_ss(&*ap.add(3)), b);
                acc3 = _mm256_fmadd_ps(d3, d3, acc3);
                let d4 = _mm256_sub_ps(_mm256_broadcast_ss(&*ap.add(4)), b);
                acc4 = _mm256_fmadd_ps(d4, d4, acc4);
                let d5 = _mm256_sub_ps(_mm256_broadcast_ss(&*ap.add(5)), b);
                acc5 = _mm256_fmadd_ps(d5, d5, acc5);
                let d6 = _mm256_sub_ps(_mm256_broadcast_ss(&*ap.add(6)), b);
                acc6 = _mm256_fmadd_ps(d6, d6, acc6);
                let d7 = _mm256_sub_ps(_mm256_broadcast_ss(&*ap.add(7)), b);
                acc7 = _mm256_fmadd_ps(d7, d7, acc7);
            }
        }};
    }

    let k_unroll = d / 4;
    let k_rem = d % 4;

    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf = (p + PREFETCH_DIST / 4) * 4;
            unsafe {
                // MR=8: each unrolled iteration spans 2 cache lines for A; prefetch both.
                _mm_prefetch(a_packed.add(pf * MR_SDIST_F32) as *const i8, _MM_HINT_T0);
                _mm_prefetch(a_packed.add(pf * MR_SDIST_F32 + 16) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b_packed.add(pf * NR_F32) as *const i8, _MM_HINT_T0);
            }
        }
        step!(base);
        step!(base + 1);
        step!(base + 2);
        step!(base + 3);
    }
    let base = k_unroll * 4;
    for r in 0..k_rem {
        step!(base + r);
    }

    unsafe {
        _mm256_storeu_ps(out, acc0);
        _mm256_storeu_ps(out.add(NR_F32), acc1);
        _mm256_storeu_ps(out.add(2 * NR_F32), acc2);
        _mm256_storeu_ps(out.add(3 * NR_F32), acc3);
        _mm256_storeu_ps(out.add(4 * NR_F32), acc4);
        _mm256_storeu_ps(out.add(5 * NR_F32), acc5);
        _mm256_storeu_ps(out.add(6 * NR_F32), acc6);
        _mm256_storeu_ps(out.add(7 * NR_F32), acc7);
    }
}

// f64: 4 output rows x 4 output cols = 4x4 register tile.
// b(1)+d(1)+acc0..3(4) = 6 YMM used.
#[inline(always)]
unsafe fn micro_sqdist_f64(d: usize, a_packed: *const f64, b_packed: *const f64, out: *mut f64) {
    let mut acc0 = unsafe { _mm256_setzero_pd() };
    let mut acc1 = unsafe { _mm256_setzero_pd() };
    let mut acc2 = unsafe { _mm256_setzero_pd() };
    let mut acc3 = unsafe { _mm256_setzero_pd() };

    macro_rules! step {
        ($offset:expr) => {{
            unsafe {
                let b = _mm256_loadu_pd(b_packed.add($offset * NR_F64));
                let ap = a_packed.add($offset * MR_SDIST_F64);
                let d0 = _mm256_sub_pd(_mm256_broadcast_sd(&*ap), b);
                acc0 = _mm256_fmadd_pd(d0, d0, acc0);
                let d1 = _mm256_sub_pd(_mm256_broadcast_sd(&*ap.add(1)), b);
                acc1 = _mm256_fmadd_pd(d1, d1, acc1);
                let d2 = _mm256_sub_pd(_mm256_broadcast_sd(&*ap.add(2)), b);
                acc2 = _mm256_fmadd_pd(d2, d2, acc2);
                let d3 = _mm256_sub_pd(_mm256_broadcast_sd(&*ap.add(3)), b);
                acc3 = _mm256_fmadd_pd(d3, d3, acc3);
            }
        }};
    }

    let k_unroll = d / 4;
    let k_rem = d % 4;

    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf = (p + PREFETCH_DIST / 4) * 4;
            unsafe {
                // MR=4: each unrolled iteration spans 2 cache lines for A; prefetch both.
                _mm_prefetch(a_packed.add(pf * MR_SDIST_F64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(a_packed.add(pf * MR_SDIST_F64 + 8) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b_packed.add(pf * NR_F64) as *const i8, _MM_HINT_T0);
            }
        }
        step!(base);
        step!(base + 1);
        step!(base + 2);
        step!(base + 3);
    }
    let base = k_unroll * 4;
    for r in 0..k_rem {
        step!(base + r);
    }

    unsafe {
        _mm256_storeu_pd(out, acc0);
        _mm256_storeu_pd(out.add(NR_F64), acc1);
        _mm256_storeu_pd(out.add(2 * NR_F64), acc2);
        _mm256_storeu_pd(out.add(3 * NR_F64), acc3);
    }
}

#[inline(always)]
pub(super) fn pairwise_sqdist_between_f32(
    points1: ArrayView2<'_, f32>, points2: ArrayView2<'_, f32>, d: usize, out: &mut [f32],
    nrows: usize, ncols: usize,
) {
    assert_eq!(out.len(), nrows * ncols);

    let n_a_blocks = (nrows + MR_SDIST_F32 - 1) / MR_SDIST_F32;
    let n_b_blocks = (ncols + NR_F32 - 1) / NR_F32;

    // Pre-pack A once: a_full[ii_block * MR * d + k * MR + i_local] = points1[ii+i_local][k]
    let mut a_full = vec![0f32; n_a_blocks * MR_SDIST_F32 * d];
    for ii_block in 0..n_a_blocks {
        let ii = ii_block * MR_SDIST_F32;
        let nr = (nrows - ii).min(MR_SDIST_F32);
        for i_local in 0..nr {
            let row = points1.row(ii + i_local);
            for k in 0..d {
                a_full[ii_block * MR_SDIST_F32 * d + k * MR_SDIST_F32 + i_local] = row[k];
            }
        }
    }

    // Pack B one jj-block at a time, then sweep all ii blocks.
    let mut b_panel = vec![0f32; NR_F32 * d];
    let mut tile = [0f32; MR_SDIST_F32 * NR_F32];

    for jj_block in 0..n_b_blocks {
        let jj = jj_block * NR_F32;
        let nc = (ncols - jj).min(NR_F32);

        pack_panel_f32(&mut b_panel, points2, jj, nc, d);

        for ii_block in 0..n_a_blocks {
            let ii = ii_block * MR_SDIST_F32;
            let nr = (nrows - ii).min(MR_SDIST_F32);
            let a_ptr = unsafe { a_full.as_ptr().add(ii_block * MR_SDIST_F32 * d) };
            unsafe { micro_sqdist_f32(d, a_ptr, b_panel.as_ptr(), tile.as_mut_ptr()) };

            for i_local in 0..nr {
                for j_local in 0..nc {
                    out[(ii + i_local) * ncols + (jj + j_local)] = tile[i_local * NR_F32 + j_local];
                }
            }
        }
    }
}

#[inline(always)]
pub(super) fn pairwise_sqdist_between_f64(
    points1: ArrayView2<'_, f64>, points2: ArrayView2<'_, f64>, d: usize, out: &mut [f64],
    nrows: usize, ncols: usize,
) {
    assert_eq!(out.len(), nrows * ncols);

    let n_a_blocks = (nrows + MR_SDIST_F64 - 1) / MR_SDIST_F64;
    let n_b_blocks = (ncols + NR_F64 - 1) / NR_F64;

    let mut a_full = vec![0f64; n_a_blocks * MR_SDIST_F64 * d];
    for ii_block in 0..n_a_blocks {
        let ii = ii_block * MR_SDIST_F64;
        let nr = (nrows - ii).min(MR_SDIST_F64);
        for i_local in 0..nr {
            let row = points1.row(ii + i_local);
            for k in 0..d {
                a_full[ii_block * MR_SDIST_F64 * d + k * MR_SDIST_F64 + i_local] = row[k];
            }
        }
    }

    let mut b_panel = vec![0f64; NR_F64 * d];
    let mut tile = [0f64; MR_SDIST_F64 * NR_F64];

    for jj_block in 0..n_b_blocks {
        let jj = jj_block * NR_F64;
        let nc = (ncols - jj).min(NR_F64);

        pack_panel_f64(&mut b_panel, points2, jj, nc, d);

        for ii_block in 0..n_a_blocks {
            let ii = ii_block * MR_SDIST_F64;
            let nr = (nrows - ii).min(MR_SDIST_F64);
            let a_ptr = unsafe { a_full.as_ptr().add(ii_block * MR_SDIST_F64 * d) };
            unsafe { micro_sqdist_f64(d, a_ptr, b_panel.as_ptr(), tile.as_mut_ptr()) };

            for i_local in 0..nr {
                for j_local in 0..nc {
                    out[(ii + i_local) * ncols + (jj + j_local)] = tile[i_local * NR_F64 + j_local];
                }
            }
        }
    }
}

/// ----- axpy (v1 += a * v2) ---------------------------------------------------
#[inline(always)]
pub(super) fn axpy_f32(v1: &mut [f32], a: f32, v2: &[f32], d: usize) {
    let va = unsafe { _mm256_set1_ps(a) };
    let mut i = 0;
    while i + 32 <= d {
        unsafe {
            let p0 = v1.as_mut_ptr().add(i);
            _mm256_storeu_ps(p0, _mm256_fmadd_ps(_mm256_loadu_ps(v2.as_ptr().add(i)), va, _mm256_loadu_ps(p0)));
            let p1 = v1.as_mut_ptr().add(i + 8);
            _mm256_storeu_ps(p1, _mm256_fmadd_ps(_mm256_loadu_ps(v2.as_ptr().add(i + 8)), va, _mm256_loadu_ps(p1)));
            let p2 = v1.as_mut_ptr().add(i + 16);
            _mm256_storeu_ps(p2, _mm256_fmadd_ps(_mm256_loadu_ps(v2.as_ptr().add(i + 16)), va, _mm256_loadu_ps(p2)));
            let p3 = v1.as_mut_ptr().add(i + 24);
            _mm256_storeu_ps(p3, _mm256_fmadd_ps(_mm256_loadu_ps(v2.as_ptr().add(i + 24)), va, _mm256_loadu_ps(p3)));
        }
        i += 32;
    }
    while i + 8 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_ps(ptr1, _mm256_fmadd_ps(_mm256_loadu_ps(v2.as_ptr().add(i)), va, _mm256_loadu_ps(ptr1)));
        }
        i += 8;
    }
    while i < d {
        unsafe {
            *v1.get_unchecked_mut(i) = v2.get_unchecked(i).mul_add(a, *v1.get_unchecked(i));
        }
        i += 1;
    }
}

#[inline(always)]
pub(super) fn axpy_f64(v1: &mut [f64], a: f64, v2: &[f64], d: usize) {
    let va = unsafe { _mm256_set1_pd(a) };
    let mut i = 0;
    while i + 16 <= d {
        unsafe {
            let p0 = v1.as_mut_ptr().add(i);
            _mm256_storeu_pd(p0, _mm256_fmadd_pd(_mm256_loadu_pd(v2.as_ptr().add(i)), va, _mm256_loadu_pd(p0)));
            let p1 = v1.as_mut_ptr().add(i + 4);
            _mm256_storeu_pd(p1, _mm256_fmadd_pd(_mm256_loadu_pd(v2.as_ptr().add(i + 4)), va, _mm256_loadu_pd(p1)));
            let p2 = v1.as_mut_ptr().add(i + 8);
            _mm256_storeu_pd(p2, _mm256_fmadd_pd(_mm256_loadu_pd(v2.as_ptr().add(i + 8)), va, _mm256_loadu_pd(p2)));
            let p3 = v1.as_mut_ptr().add(i + 12);
            _mm256_storeu_pd(p3, _mm256_fmadd_pd(_mm256_loadu_pd(v2.as_ptr().add(i + 12)), va, _mm256_loadu_pd(p3)));
        }
        i += 16;
    }
    while i + 4 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_pd(ptr1, _mm256_fmadd_pd(_mm256_loadu_pd(v2.as_ptr().add(i)), va, _mm256_loadu_pd(ptr1)));
        }
        i += 4;
    }
    while i < d {
        unsafe {
            *v1.get_unchecked_mut(i) = v2.get_unchecked(i).mul_add(a, *v1.get_unchecked(i));
        }
        i += 1;
    }
}

// ----- sum -------------------------------------------------------------------

#[inline(always)]
pub(super) fn sum_f32(v: &[f32], d: usize) -> f32 {
    let mut i = 0;
    let mut acc = unsafe { _mm256_setzero_ps() };
    while i + 8 <= d {
        unsafe { acc = _mm256_add_ps(acc, _mm256_loadu_ps(v.as_ptr().add(i))) };
        i += 8;
    }
    let mut sum = unsafe { horizontal_sum_f32(acc) };
    while i < d {
        sum += unsafe { *v.get_unchecked(i) };
        i += 1;
    }
    sum
}

#[inline(always)]
pub(super) fn sum_f64(v: &[f64], d: usize) -> f64 {
    let mut i = 0;
    let mut acc = unsafe { _mm256_setzero_pd() };
    while i + 4 <= d {
        unsafe { acc = _mm256_add_pd(acc, _mm256_loadu_pd(v.as_ptr().add(i))) };
        i += 4;
    }
    let mut sum = unsafe { horizontal_sum_f64(acc) };
    while i < d {
        sum += unsafe { *v.get_unchecked(i) };
        i += 1;
    }
    sum
}

// ----- add_scalar (v += s) ---------------------------------------------------

#[inline(always)]
pub(super) fn add_scalar_f32(v: &mut [f32], s: f32, d: usize) {
    let vs = unsafe { _mm256_set1_ps(s) };
    let mut i = 0;
    while i + 32 <= d {
        unsafe {
            let p0 = v.as_mut_ptr().add(i);
            _mm256_storeu_ps(p0, _mm256_add_ps(_mm256_loadu_ps(p0), vs));
            let p1 = v.as_mut_ptr().add(i + 8);
            _mm256_storeu_ps(p1, _mm256_add_ps(_mm256_loadu_ps(p1), vs));
            let p2 = v.as_mut_ptr().add(i + 16);
            _mm256_storeu_ps(p2, _mm256_add_ps(_mm256_loadu_ps(p2), vs));
            let p3 = v.as_mut_ptr().add(i + 24);
            _mm256_storeu_ps(p3, _mm256_add_ps(_mm256_loadu_ps(p3), vs));
        }
        i += 32;
    }
    while i + 8 <= d {
        unsafe {
            let ptr = v.as_mut_ptr().add(i);
            _mm256_storeu_ps(ptr, _mm256_add_ps(_mm256_loadu_ps(ptr), vs));
        }
        i += 8;
    }
    while i < d {
        unsafe { *v.get_unchecked_mut(i) += s };
        i += 1;
    }
}

#[inline(always)]
pub(super) fn add_scalar_f64(v: &mut [f64], s: f64, d: usize) {
    let vs = unsafe { _mm256_set1_pd(s) };
    let mut i = 0;
    while i + 16 <= d {
        unsafe {
            let p0 = v.as_mut_ptr().add(i);
            _mm256_storeu_pd(p0, _mm256_add_pd(_mm256_loadu_pd(p0), vs));
            let p1 = v.as_mut_ptr().add(i + 4);
            _mm256_storeu_pd(p1, _mm256_add_pd(_mm256_loadu_pd(p1), vs));
            let p2 = v.as_mut_ptr().add(i + 8);
            _mm256_storeu_pd(p2, _mm256_add_pd(_mm256_loadu_pd(p2), vs));
            let p3 = v.as_mut_ptr().add(i + 12);
            _mm256_storeu_pd(p3, _mm256_add_pd(_mm256_loadu_pd(p3), vs));
        }
        i += 16;
    }
    while i + 4 <= d {
        unsafe {
            let ptr = v.as_mut_ptr().add(i);
            _mm256_storeu_pd(ptr, _mm256_add_pd(_mm256_loadu_pd(ptr), vs));
        }
        i += 4;
    }
    while i < d {
        unsafe { *v.get_unchecked_mut(i) += s };
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// 1-row distance kernels: squared distance from one center to n points.
// MR=1, NR=8 (f32) / NR=4 (f64).  Uses 1 acc + 1 b + 1 diff = 3 YMM.
// ---------------------------------------------------------------------------

// f32: 1 output row x NR_F32=8 output cols.  3 YMM total.
#[inline(always)]
unsafe fn micro_rowdist_f32(d: usize, center: *const f32, b_packed: *const f32, out: *mut f32) {
    let mut acc = unsafe { _mm256_setzero_ps() };

    macro_rules! step {
        ($offset:expr) => {{
            unsafe {
                let b = _mm256_loadu_ps(b_packed.add($offset * NR_F32));
                let c = _mm256_broadcast_ss(&*center.add($offset));
                let diff = _mm256_sub_ps(c, b);
                acc = _mm256_fmadd_ps(diff, diff, acc);
            }
        }};
    }

    let k_unroll = d / 4;
    let k_rem = d % 4;

    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf = (p + PREFETCH_DIST / 4) * 4;
            unsafe {
                _mm_prefetch(b_packed.add(pf * NR_F32) as *const i8, _MM_HINT_T0);
            }
        }
        step!(base);
        step!(base + 1);
        step!(base + 2);
        step!(base + 3);
    }
    let base = k_unroll * 4;
    for r in 0..k_rem {
        step!(base + r);
    }

    unsafe {
        _mm256_storeu_ps(out, acc);
    }
}

// f32: 1 center x NR_F32=8 points direct from Fortran-order layout.
// `base` points to points[jj, 0]; loads NR_F32 values at column k via `base + k * col_stride`.
#[inline(always)]
unsafe fn micro_rowdist_fortran_f32(
    d: usize, center: *const f32, base: *const f32, col_stride: usize, out: *mut f32,
) {
    let mut acc = unsafe { _mm256_setzero_ps() };

    macro_rules! step {
        ($offset:expr) => {{
            unsafe {
                let b = _mm256_loadu_ps(base.add($offset * col_stride));
                let c = _mm256_broadcast_ss(&*center.add($offset));
                let diff = _mm256_sub_ps(c, b);
                acc = _mm256_fmadd_ps(diff, diff, acc);
            }
        }};
    }

    let k_unroll = d / 4;
    let k_rem = d % 4;
    for p in 0..k_unroll {
        let base_k = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf = (p + PREFETCH_DIST / 4) * 4;
            unsafe {
                _mm_prefetch(base.add(pf * col_stride) as *const i8, _MM_HINT_T0);
            }
        }
        step!(base_k);
        step!(base_k + 1);
        step!(base_k + 2);
        step!(base_k + 3);
    }
    let base_k = k_unroll * 4;
    for r in 0..k_rem {
        step!(base_k + r);
    }
    unsafe {
        _mm256_storeu_ps(out, acc);
    }
}

/// Squared distances from a single `center` to each of the `n` rows in `points`.
pub(super) fn rowdist_f32(
    center: &[f32], points: ArrayView2<'_, f32>, d: usize, out: &mut [f32], n: usize,
) {
    assert_eq!(out.len(), n);
    let strides = points.strides();

    if strides[1] == 1 {
        // C-order: rows are contiguous; SIMD over d, one point at a time.
        for j in 0..n {
            let row = points.row(j);
            out[j] = sqdist_f32(center, row.as_slice().expect("C-order row is contiguous"), d);
        }
        return;
    }

    if strides[0] == 1 && strides[1] > 0 {
        // Fortran-order: columns are contiguous; NR-way SIMD without packing.
        let col_stride = strides[1] as usize;
        let data_ptr = points.as_ptr();
        let n_b_blocks = (n + NR_F32 - 1) / NR_F32;
        let mut tile = [0f32; NR_F32];
        for jj_block in 0..n_b_blocks {
            let jj = jj_block * NR_F32;
            let nc = (n - jj).min(NR_F32);
            if nc == NR_F32 {
                unsafe {
                    micro_rowdist_fortran_f32(
                        d,
                        center.as_ptr(),
                        data_ptr.add(jj),
                        col_stride,
                        tile.as_mut_ptr(),
                    );
                }
                out[jj..jj + NR_F32].copy_from_slice(&tile);
            } else {
                // Tail: scalar fallback for the last nc < NR_F32 points.
                for j_local in 0..nc {
                    let j = jj + j_local;
                    let mut sum = 0f32;
                    for k in 0..d {
                        let diff = center[k] - unsafe { *data_ptr.add(j + k * col_stride) };
                        sum += diff * diff;
                    }
                    out[j] = sum;
                }
            }
        }
        return;
    }

    // General strides: pack into b_panel and use the packed micro-kernel.
    let n_b_blocks = (n + NR_F32 - 1) / NR_F32;
    ROW_PANEL_F32.with(|panel| {
        let mut b_panel = panel.borrow_mut();
        b_panel.resize(NR_F32 * d, 0.0);
        let mut tile = [0f32; NR_F32];

        for jj_block in 0..n_b_blocks {
            let jj = jj_block * NR_F32;
            let nc = (n - jj).min(NR_F32);

            pack_panel_f32(&mut b_panel, points, jj, nc, d);

            unsafe { micro_rowdist_f32(d, center.as_ptr(), b_panel.as_ptr(), tile.as_mut_ptr()) };
            out[jj..jj + nc].copy_from_slice(&tile[..nc]);
        }
    });
}

// f64: 1 output row x NR_F64=4 output cols.  3 YMM total.
#[inline(always)]
unsafe fn micro_rowdist_f64(d: usize, center: *const f64, b_packed: *const f64, out: *mut f64) {
    let mut acc = unsafe { _mm256_setzero_pd() };

    macro_rules! step {
        ($offset:expr) => {{
            unsafe {
                let b = _mm256_loadu_pd(b_packed.add($offset * NR_F64));
                let c = _mm256_broadcast_sd(&*center.add($offset));
                let diff = _mm256_sub_pd(c, b);
                acc = _mm256_fmadd_pd(diff, diff, acc);
            }
        }};
    }

    let k_unroll = d / 4;
    let k_rem = d % 4;

    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf = (p + PREFETCH_DIST / 4) * 4;
            unsafe {
                _mm_prefetch(b_packed.add(pf * NR_F64) as *const i8, _MM_HINT_T0);
            }
        }
        step!(base);
        step!(base + 1);
        step!(base + 2);
        step!(base + 3);
    }
    let base = k_unroll * 4;
    for r in 0..k_rem {
        step!(base + r);
    }

    unsafe {
        _mm256_storeu_pd(out, acc);
    }
}

// f64: 1 center x NR_F64=4 points direct from Fortran-order layout.
// `base` points to points[jj, 0]; loads NR_F64 values at column k via `base + k * col_stride`.
#[inline(always)]
unsafe fn micro_rowdist_fortran_f64(
    d: usize, center: *const f64, base: *const f64, col_stride: usize, out: *mut f64,
) {
    let mut acc = unsafe { _mm256_setzero_pd() };

    macro_rules! step {
        ($offset:expr) => {{
            unsafe {
                let b = _mm256_loadu_pd(base.add($offset * col_stride));
                let c = _mm256_broadcast_sd(&*center.add($offset));
                let diff = _mm256_sub_pd(c, b);
                acc = _mm256_fmadd_pd(diff, diff, acc);
            }
        }};
    }

    let k_unroll = d / 4;
    let k_rem = d % 4;
    for p in 0..k_unroll {
        let base_k = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf = (p + PREFETCH_DIST / 4) * 4;
            unsafe {
                _mm_prefetch(base.add(pf * col_stride) as *const i8, _MM_HINT_T0);
            }
        }
        step!(base_k);
        step!(base_k + 1);
        step!(base_k + 2);
        step!(base_k + 3);
    }
    let base_k = k_unroll * 4;
    for r in 0..k_rem {
        step!(base_k + r);
    }
    unsafe {
        _mm256_storeu_pd(out, acc);
    }
}

/// Squared distances from a single `center` to each of the `n` rows in `points`.
pub(super) fn rowdist_f64(
    center: &[f64], points: ArrayView2<'_, f64>, d: usize, out: &mut [f64], n: usize,
) {
    assert_eq!(out.len(), n);
    let strides = points.strides();

    if strides[1] == 1 {
        // C-order: rows are contiguous; SIMD over d, one point at a time.
        for j in 0..n {
            let row = points.row(j);
            out[j] = sqdist_f64(center, row.as_slice().expect("C-order row is contiguous"), d);
        }
        return;
    }

    if strides[0] == 1 && strides[1] > 0 {
        // Fortran-order: columns are contiguous; NR-way SIMD without packing.
        let col_stride = strides[1] as usize;
        let data_ptr = points.as_ptr();
        let n_b_blocks = (n + NR_F64 - 1) / NR_F64;
        let mut tile = [0f64; NR_F64];
        for jj_block in 0..n_b_blocks {
            let jj = jj_block * NR_F64;
            let nc = (n - jj).min(NR_F64);
            if nc == NR_F64 {
                unsafe {
                    micro_rowdist_fortran_f64(
                        d,
                        center.as_ptr(),
                        data_ptr.add(jj),
                        col_stride,
                        tile.as_mut_ptr(),
                    );
                }
                out[jj..jj + NR_F64].copy_from_slice(&tile);
            } else {
                // Tail: scalar fallback for the last nc < NR_F64 points.
                for j_local in 0..nc {
                    let j = jj + j_local;
                    let mut sum = 0f64;
                    for k in 0..d {
                        let diff = center[k] - unsafe { *data_ptr.add(j + k * col_stride) };
                        sum += diff * diff;
                    }
                    out[j] = sum;
                }
            }
        }
        return;
    }

    // General strides: pack into b_panel and use the packed micro-kernel.
    let n_b_blocks = (n + NR_F64 - 1) / NR_F64;
    ROW_PANEL_F64.with(|panel| {
        let mut b_panel = panel.borrow_mut();
        b_panel.resize(NR_F64 * d, 0.0);
        let mut tile = [0f64; NR_F64];

        for jj_block in 0..n_b_blocks {
            let jj = jj_block * NR_F64;
            let nc = (n - jj).min(NR_F64);

            pack_panel_f64(&mut b_panel, points, jj, nc, d);

            unsafe { micro_rowdist_f64(d, center.as_ptr(), b_panel.as_ptr(), tile.as_mut_ptr()) };
            out[jj..jj + nc].copy_from_slice(&tile[..nc]);
        }
    });
}
