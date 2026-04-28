//! AVX2-accelerated implementations of the vector math primitives.
//!
//! Concrete `_f32` / `_f64` free functions; no generics, no `TypeId`.
//! Called from `vecops.rs` which provides the `VecOps` trait impls.
//! Assumes AVX2 and FMA are available (compile with `-C target-feature=+avx2,+fma`
//! or `-C target-cpu=native`).

use std::arch::x86_64::*;

// Tile sizes for the pairwise-sqdist micro-kernels.
// NR = number of j-columns covered by one YMM register (8 f32 / 4 f64).
// MR_SDIST = number of i-rows in the register tile; limited by available
// accumulator registers after reserving registers for b and diff temporaries.
const NR_F32: usize = 8;
const MR_SDIST_F32: usize = 4; // 4 accumulators * 3 regs per row (acc,d,b) = 12 YMM used
const NR_F64: usize = 4;
const MR_SDIST_F64: usize = 2; // 2 accumulators
// Prefetch lookahead in units of 4-step unroll iterations.
const PREFETCH_DIST: usize = 12;

// ----- sqdist ----------------------------------------------------------------

#[inline(always)]
pub(super) fn sqdist_f32(v1: &[f32], v2: &[f32], d: usize) -> f32 {
    let mut i = 0;
    let mut acc = unsafe { _mm256_setzero_ps() };
    while i + 8 <= d {
        unsafe {
            let a = _mm256_loadu_ps(v1.as_ptr().add(i));
            let b = _mm256_loadu_ps(v2.as_ptr().add(i));
            let diff = _mm256_sub_ps(a, b);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }
        i += 8;
    }
    let mut buf = [0f32; 8];
    unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), acc) };
    let mut sum: f32 = buf.iter().copied().sum();
    while i < d {
        let x = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
        sum = x.mul_add(x, sum);
        i += 1;
    }
    sum
}

#[inline(always)]
pub(super) fn sqdist_f64(v1: &[f64], v2: &[f64], d: usize) -> f64 {
    let mut i = 0;
    let mut acc = unsafe { _mm256_setzero_pd() };
    while i + 4 <= d {
        unsafe {
            let a = _mm256_loadu_pd(v1.as_ptr().add(i));
            let b = _mm256_loadu_pd(v2.as_ptr().add(i));
            let diff = _mm256_sub_pd(a, b);
            acc = _mm256_fmadd_pd(diff, diff, acc);
        }
        i += 4;
    }
    let mut buf = [0f64; 4];
    unsafe { _mm256_storeu_pd(buf.as_mut_ptr(), acc) };
    let mut sum: f64 = buf.iter().copied().sum();
    while i < d {
        let x = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
        sum = x.mul_add(x, sum);
        i += 1;
    }
    sum
}

// ----- l1dist ----------------------------------------------------------------

#[inline(always)]
pub(super) fn l1dist_f32(v1: &[f32], v2: &[f32], d: usize) -> f32 {
    let mut i = 0;
    let mut acc = unsafe { _mm256_setzero_ps() };
    let abs_mask = unsafe { _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff)) };
    while i + 8 <= d {
        unsafe {
            let a = _mm256_loadu_ps(v1.as_ptr().add(i));
            let b = _mm256_loadu_ps(v2.as_ptr().add(i));
            let diff = _mm256_sub_ps(a, b);
            let abs = _mm256_and_ps(diff, abs_mask);
            acc = _mm256_add_ps(acc, abs);
        }
        i += 8;
    }
    let mut buf = [0f32; 8];
    unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), acc) };
    let mut sum: f32 = buf.iter().copied().sum();
    while i < d {
        sum += (unsafe { *v1.get_unchecked(i) } - unsafe { *v2.get_unchecked(i) }).abs();
        i += 1;
    }
    sum
}

#[inline(always)]
pub(super) fn l1dist_f64(v1: &[f64], v2: &[f64], d: usize) -> f64 {
    let mut i = 0;
    let mut acc = unsafe { _mm256_setzero_pd() };
    let abs_mask = unsafe { _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fff_ffff_ffff_ffff)) };
    while i + 4 <= d {
        unsafe {
            let a = _mm256_loadu_pd(v1.as_ptr().add(i));
            let b = _mm256_loadu_pd(v2.as_ptr().add(i));
            let diff = _mm256_sub_pd(a, b);
            let abs = _mm256_and_pd(diff, abs_mask);
            acc = _mm256_add_pd(acc, abs);
        }
        i += 4;
    }
    let mut buf = [0f64; 4];
    unsafe { _mm256_storeu_pd(buf.as_mut_ptr(), acc) };
    let mut sum: f64 = buf.iter().copied().sum();
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
    while i + 8 <= d {
        unsafe {
            let b = _mm256_loadu_ps(v2.as_ptr().add(i));
            _mm256_storeu_ps(v1.as_mut_ptr().add(i), _mm256_mul_ps(b, scalar));
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
    while i + 4 <= d {
        unsafe {
            let b = _mm256_loadu_pd(v2.as_ptr().add(i));
            _mm256_storeu_pd(v1.as_mut_ptr().add(i), _mm256_mul_pd(b, scalar));
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
    while i + 8 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_ps(
                ptr1,
                _mm256_add_ps(_mm256_loadu_ps(ptr1), _mm256_loadu_ps(v2.as_ptr().add(i))),
            );
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
    while i + 4 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_pd(
                ptr1,
                _mm256_add_pd(_mm256_loadu_pd(ptr1), _mm256_loadu_pd(v2.as_ptr().add(i))),
            );
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
    while i + 8 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_ps(
                ptr1,
                _mm256_sub_ps(_mm256_loadu_ps(ptr1), _mm256_loadu_ps(v2.as_ptr().add(i))),
            );
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
    while i + 4 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_pd(
                ptr1,
                _mm256_sub_pd(_mm256_loadu_pd(ptr1), _mm256_loadu_pd(v2.as_ptr().add(i))),
            );
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
    while i + 8 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            let mut x = _mm256_loadu_ps(ptr1);
            x = _mm256_fmadd_ps(x, va, _mm256_loadu_ps(v2.as_ptr().add(i)));
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
    while i + 4 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            let mut x = _mm256_loadu_pd(ptr1);
            x = _mm256_fmadd_pd(x, va, _mm256_loadu_pd(v2.as_ptr().add(i)));
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
    let mut acc = unsafe { _mm256_setzero_ps() };
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
    let mut buf = [0f32; 8];
    unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), acc) };
    let mut sum: f32 = buf.iter().copied().sum();
    while i < d {
        sum = unsafe { v1.get_unchecked(i).mul_add(*v2.get_unchecked(i), sum) };
        i += 1;
    }
    sum
}

#[inline(always)]
pub(super) fn dot_f64(v1: &[f64], v2: &[f64], d: usize) -> f64 {
    let mut i = 0;
    let mut acc = unsafe { _mm256_setzero_pd() };
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
    let mut buf = [0f64; 4];
    unsafe { _mm256_storeu_pd(buf.as_mut_ptr(), acc) };
    let mut sum: f64 = buf.iter().copied().sum();
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

// f32: 4 output rows x 8 output cols = 4x8 register tile.
// Each acc_i is one __m256 accumulating sqdist for row i vs all 8 j-columns.
// 4-way k-unroll with software prefetch.
#[inline(always)]
unsafe fn micro_sqdist_f32(d: usize, a_packed: *const f32, b_packed: *const f32, out: *mut f32) {
    // MR=4 rows, NR=8 cols
    let mut acc0 = unsafe { _mm256_setzero_ps() };
    let mut acc1 = unsafe { _mm256_setzero_ps() };
    let mut acc2 = unsafe { _mm256_setzero_ps() };
    let mut acc3 = unsafe { _mm256_setzero_ps() };

    // One k-step: load 8 b-values (one YMM), broadcast each of 4 a-values, fmadd diff^2.
    macro_rules! step {
        ($offset:expr) => {{
            unsafe {
                // b_panel row $offset: 8 floats = one YMM
                let b = _mm256_loadu_ps(b_packed.add($offset * NR_F32));
                // a row 0..3 for this k
                let a0 = _mm256_broadcast_ss(&*a_packed.add($offset * MR_SDIST_F32));
                let a1 = _mm256_broadcast_ss(&*a_packed.add($offset * MR_SDIST_F32 + 1));
                let a2 = _mm256_broadcast_ss(&*a_packed.add($offset * MR_SDIST_F32 + 2));
                let a3 = _mm256_broadcast_ss(&*a_packed.add($offset * MR_SDIST_F32 + 3));
                let d0 = _mm256_sub_ps(a0, b);
                let d1 = _mm256_sub_ps(a1, b);
                let d2 = _mm256_sub_ps(a2, b);
                let d3 = _mm256_sub_ps(a3, b);
                acc0 = _mm256_fmadd_ps(d0, d0, acc0);
                acc1 = _mm256_fmadd_ps(d1, d1, acc1);
                acc2 = _mm256_fmadd_ps(d2, d2, acc2);
                acc3 = _mm256_fmadd_ps(d3, d3, acc3);
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
                _mm_prefetch(a_packed.add(pf * MR_SDIST_F32) as *const i8, _MM_HINT_T0);
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
    }
}

// f64: 2 output rows x 4 output cols = 2x4 register tile.
#[inline(always)]
unsafe fn micro_sqdist_f64(d: usize, a_packed: *const f64, b_packed: *const f64, out: *mut f64) {
    let mut acc0 = unsafe { _mm256_setzero_pd() };
    let mut acc1 = unsafe { _mm256_setzero_pd() };

    macro_rules! step {
        ($offset:expr) => {{
            unsafe {
                let b = _mm256_loadu_pd(b_packed.add($offset * NR_F64));
                let a0 = _mm256_broadcast_sd(&*a_packed.add($offset * MR_SDIST_F64));
                let a1 = _mm256_broadcast_sd(&*a_packed.add($offset * MR_SDIST_F64 + 1));
                let d0 = _mm256_sub_pd(a0, b);
                let d1 = _mm256_sub_pd(a1, b);
                acc0 = _mm256_fmadd_pd(d0, d0, acc0);
                acc1 = _mm256_fmadd_pd(d1, d1, acc1);
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
                _mm_prefetch(a_packed.add(pf * MR_SDIST_F64) as *const i8, _MM_HINT_T0);
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
    }
}

#[inline(always)]
pub(super) fn pairwise_sqdist_between_f32<D1, D2>(
    points1: &[D1], points2: &[D2], d: usize, out: &mut [f32], nrows: usize, ncols: usize,
) where
    D1: AsRef<[f32]>,
    D2: AsRef<[f32]>,
{
    assert_eq!(out.len(), nrows * ncols);

    let n_a_blocks = (nrows + MR_SDIST_F32 - 1) / MR_SDIST_F32;
    let n_b_blocks = (ncols + NR_F32 - 1) / NR_F32;

    // Pre-pack A once: a_full[ii_block * MR * d + k * MR + i_local] = points1[ii+i_local][k]
    let mut a_full = vec![0f32; n_a_blocks * MR_SDIST_F32 * d];
    for ii_block in 0..n_a_blocks {
        let ii = ii_block * MR_SDIST_F32;
        let nr = (nrows - ii).min(MR_SDIST_F32);
        for i_local in 0..nr {
            let row = points1[ii + i_local].as_ref();
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

        b_panel.fill(0.0);
        for j_local in 0..nc {
            let row = points2[jj + j_local].as_ref();
            for k in 0..d {
                b_panel[k * NR_F32 + j_local] = row[k];
            }
        }

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
pub(super) fn pairwise_sqdist_between_f64<D1, D2>(
    points1: &[D1], points2: &[D2], d: usize, out: &mut [f64], nrows: usize, ncols: usize,
) where
    D1: AsRef<[f64]>,
    D2: AsRef<[f64]>,
{
    assert_eq!(out.len(), nrows * ncols);

    let n_a_blocks = (nrows + MR_SDIST_F64 - 1) / MR_SDIST_F64;
    let n_b_blocks = (ncols + NR_F64 - 1) / NR_F64;

    let mut a_full = vec![0f64; n_a_blocks * MR_SDIST_F64 * d];
    for ii_block in 0..n_a_blocks {
        let ii = ii_block * MR_SDIST_F64;
        let nr = (nrows - ii).min(MR_SDIST_F64);
        for i_local in 0..nr {
            let row = points1[ii + i_local].as_ref();
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

        b_panel.fill(0.0);
        for j_local in 0..nc {
            let row = points2[jj + j_local].as_ref();
            for k in 0..d {
                b_panel[k * NR_F64 + j_local] = row[k];
            }
        }

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
    while i + 8 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_ps(
                ptr1,
                _mm256_fmadd_ps(_mm256_loadu_ps(v2.as_ptr().add(i)), va, _mm256_loadu_ps(ptr1)),
            );
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
    while i + 4 <= d {
        unsafe {
            let ptr1 = v1.as_mut_ptr().add(i);
            _mm256_storeu_pd(
                ptr1,
                _mm256_fmadd_pd(_mm256_loadu_pd(v2.as_ptr().add(i)), va, _mm256_loadu_pd(ptr1)),
            );
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
    let mut buf = [0f32; 8];
    unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), acc) };
    let mut sum: f32 = buf.iter().copied().sum();
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
    let mut buf = [0f64; 4];
    unsafe { _mm256_storeu_pd(buf.as_mut_ptr(), acc) };
    let mut sum: f64 = buf.iter().copied().sum();
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
