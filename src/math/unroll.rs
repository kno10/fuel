//! Unrolled (LANES = 8) implementations of the vector math primitives.
//!
//! Used on non-x86_64 architectures (e.g. ARM) when d >= UNROLL_SIZE, giving
//! the compiler a fixed-width inner loop body suitable for auto-vectorisation
//! (e.g. ARM NEON / VFPv4).

use ndarray::ArrayView2;

use crate::Float;

const LANES: usize = 8;
const MR: usize = 4;
const NR: usize = LANES;

#[inline(always)]
pub(super) fn sqdist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    let mut vsum = [N::zero(); LANES];
    for i in (0..sd).step_by(LANES) {
        let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                let x = *vv.get_unchecked(j) - *cc.get_unchecked(j);
                *vsum.get_unchecked_mut(j) += x * x;
            }
        }
    }
    let mut sum = vsum.iter().copied().sum::<N>();
    for i in sd..d {
        let x = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
        sum += x * x;
    }
    sum
}

#[inline(always)]
pub(super) fn l1dist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    let mut vsum = [N::zero(); LANES];
    for i in (0..sd).step_by(LANES) {
        let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                let diff = *vv.get_unchecked(j) - *cc.get_unchecked(j);
                *vsum.get_unchecked_mut(j) += diff.abs();
            }
        }
    }
    let mut sum = vsum.iter().copied().sum::<N>();
    for i in sd..d {
        let diff = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
        sum += diff.abs();
    }
    sum
}

#[inline(always)]
pub(super) fn mul<N>(v1: &mut [N], v2: &[N], a: N, d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *b1.get_unchecked_mut(j) = *b2.get_unchecked(j) * a;
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a;
        }
    }
}

#[inline(always)]
pub(super) fn mul_assign<N>(v: &mut [N], f: N, d: usize)
where
    N: Float,
{
    assert!(v.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let b = &mut v[i..(i + LANES)];
        for j in 0..LANES {
            unsafe {
                *b.get_unchecked_mut(j) *= f;
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v.get_unchecked_mut(i) *= f;
        }
    }
}

#[inline(always)]
pub(super) fn add_assign<N>(v1: &mut [N], v2: &[N], d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *b1.get_unchecked_mut(j) += *b2.get_unchecked(j);
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v1.get_unchecked_mut(i) += *v2.get_unchecked(i);
        }
    }
}

#[inline(always)]
pub(super) fn sub_assign<N>(v1: &mut [N], v2: &[N], d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *b1.get_unchecked_mut(j) -= *b2.get_unchecked(j);
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i);
        }
    }
}

#[inline(always)]
pub(super) fn fmamul<N>(v1: &mut [N], a: N, v2: &[N], b: N, d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);

    if cfg!(any(
        target_feature = "fma",
        target_feature = "neon",
        target_feature = "vfp4"
    )) {
        for i in (0..sd).step_by(LANES) {
            let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
            for j in 0..LANES {
                unsafe {
                    let fma =
                        num_traits::Float::mul_add(*b1.get_unchecked(j), a, *b2.get_unchecked(j));
                    *b1.get_unchecked_mut(j) = fma * b;
                }
            }
        }
        for i in sd..d {
            unsafe {
                let fma = num_traits::Float::mul_add(*v1.get_unchecked(i), a, *v2.get_unchecked(i));
                *v1.get_unchecked_mut(i) = fma * b;
            }
        }
        return;
    }

    for i in (0..sd).step_by(LANES) {
        let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *b1.get_unchecked_mut(j) = (*b1.get_unchecked(j) * a + *b2.get_unchecked(j)) * b;
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v1.get_unchecked_mut(i) = (*v1.get_unchecked(i) * a + *v2.get_unchecked(i)) * b;
        }
    }
}

#[inline(always)]
pub(super) fn dot<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    let mut vsum = [N::zero(); LANES];
    for i in (0..sd).step_by(LANES) {
        let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *vsum.get_unchecked_mut(j) += *vv.get_unchecked(j) * *cc.get_unchecked(j);
            }
        }
    }
    let mut sum = vsum.iter().copied().sum::<N>();
    for i in sd..d {
        sum += unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) };
    }
    sum
}

// Micro-tile dimensions mirror AVX2 (MR=4, NR=8) for cache-friendly auto-vectorisation.
// The inner NR=LANES=8 loop is a fixed-width body that compilers can auto-vectorise
// with NEON, VFPv4, or any SIMD backend, given -C target-cpu=native.
#[inline(always)]
pub(super) fn pairwise_sqdist_between<N>(
    points1: ArrayView2<'_, N>, points2: ArrayView2<'_, N>, d: usize, out: &mut [N], nrows: usize,
    ncols: usize,
) where
    N: Float,
{
    assert_eq!(out.len(), nrows * ncols);

    let n_a_blocks = (nrows + MR - 1) / MR;
    let n_b_blocks = (ncols + NR - 1) / NR;

    // Pre-pack A once: a_full[ii_block * MR * d + k * MR + i_local] = points1[ii+i_local][k]
    let mut a_full = vec![N::zero(); n_a_blocks * MR * d];
    for ii_block in 0..n_a_blocks {
        let ii = ii_block * MR;
        let nr = (nrows - ii).min(MR);
        for i_local in 0..nr {
            let row = points1.row(ii + i_local);
            for k in 0..d {
                a_full[ii_block * MR * d + k * MR + i_local] = row[k];
            }
        }
    }

    // Pack B one jj-block at a time, then sweep all ii-blocks (A stays hot in L1).
    let mut b_panel = vec![N::zero(); NR * d];
    // MR * NR = 4 * 8 = 32; N: Copy so array initialisation is valid.
    let mut tile = [N::zero(); MR * NR];

    for jj_block in 0..n_b_blocks {
        let jj = jj_block * NR;
        let nc = (ncols - jj).min(NR);

        for x in b_panel.iter_mut() {
            *x = N::zero();
        }
        for j_local in 0..nc {
            let row = points2.row(jj + j_local);
            for k in 0..d {
                b_panel[k * NR + j_local] = row[k];
            }
        }

        for ii_block in 0..n_a_blocks {
            let ii = ii_block * MR;
            let nr = (nrows - ii).min(MR);

            for x in tile.iter_mut() {
                *x = N::zero();
            }
            let a_panel = &a_full[ii_block * MR * d..];

            // Inner micro-kernel: NR-wide inner loop is fixed-width => auto-vectorisable.
            for k in 0..d {
                let a_base = k * MR;
                let b_base = k * NR;
                for i in 0..MR {
                    let av = unsafe { *a_panel.get_unchecked(a_base + i) };
                    let bb = unsafe { b_panel.get_unchecked(b_base..b_base + NR) };
                    let tt = unsafe { tile.get_unchecked_mut(i * NR..i * NR + NR) };
                    for j in 0..NR {
                        unsafe {
                            let diff = av - *bb.get_unchecked(j);
                            *tt.get_unchecked_mut(j) += diff * diff;
                        }
                    }
                }
            }

            for i_local in 0..nr {
                for j_local in 0..nc {
                    out[(ii + i_local) * ncols + (jj + j_local)] = tile[i_local * NR + j_local];
                }
            }
        }
    }
}

#[inline(always)]
pub(super) fn axpy<N>(v1: &mut [N], a: N, v2: &[N], d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *b1.get_unchecked_mut(j) += *b2.get_unchecked(j) * a;
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v1.get_unchecked_mut(i) += *v2.get_unchecked(i) * a;
        }
    }
}

#[inline(always)]
pub(super) fn sum<N>(v: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v.len() >= d);
    let sd = d & !(LANES - 1);
    let mut vsum = [N::zero(); LANES];
    for i in (0..sd).step_by(LANES) {
        let chunk = &v[i..(i + LANES)];
        for j in 0..LANES {
            unsafe {
                *vsum.get_unchecked_mut(j) += *chunk.get_unchecked(j);
            }
        }
    }
    let mut s = vsum.iter().copied().sum::<N>();
    for i in sd..d {
        s += unsafe { *v.get_unchecked(i) };
    }
    s
}

#[inline(always)]
pub(super) fn add_scalar<N>(v: &mut [N], s: N, d: usize)
where
    N: Float,
{
    assert!(v.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let b = &mut v[i..(i + LANES)];
        for j in 0..LANES {
            unsafe {
                *b.get_unchecked_mut(j) += s;
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v.get_unchecked_mut(i) += s;
        }
    }
}
/// Squared distances from a single `center` to each of the `n` rows in `points`.
/// Dispatches on memory layout: C-order, Fortran-order, or general strides.
pub(super) fn rowdist<N>(center: &[N], points: ArrayView2<'_, N>, d: usize, out: &mut [N], n: usize)
where
    N: Float,
{
    assert_eq!(out.len(), n);
    let strides = points.strides();

    if strides[1] == 1 {
        // C-order: rows are contiguous; auto-vectorises over d, one point at a time.
        for j in 0..n {
            let row = points.row(j);
            out[j] = sqdist(center, row.as_slice().expect("C-order row is contiguous"), d);
        }
        return;
    }

    if strides[0] == 1 && strides[1] > 0 {
        // Fortran-order: columns are contiguous; NR-wide inner loop without packing.
        let col_stride = strides[1] as usize;
        let data_ptr = points.as_ptr();
        let n_b_blocks = (n + NR - 1) / NR;
        let mut tile = [N::zero(); NR];
        for jj_block in 0..n_b_blocks {
            let jj = jj_block * NR;
            let nc = (n - jj).min(NR);
            for t in tile[..nc].iter_mut() {
                *t = N::zero();
            }
            for k in 0..d {
                let av = center[k];
                for j_local in 0..nc {
                    let bv = unsafe { *data_ptr.add(jj + j_local + k * col_stride) };
                    let diff = av - bv;
                    tile[j_local] = tile[j_local] + diff * diff;
                }
            }
            out[jj..jj + nc].copy_from_slice(&tile[..nc]);
        }
        return;
    }

    // General strides: pack into b_panel and use NR-wide inner loop.
    let n_b_blocks = (n + NR - 1) / NR;
    let mut b_panel = vec![N::zero(); NR * d];
    let mut tile = [N::zero(); NR];

    for jj_block in 0..n_b_blocks {
        let jj = jj_block * NR;
        let nc = (n - jj).min(NR);

        for x in b_panel.iter_mut() {
            *x = N::zero();
        }
        for j_local in 0..nc {
            let row = points.row(jj + j_local);
            for k in 0..d {
                b_panel[k * NR + j_local] = row[k];
            }
        }

        for x in tile.iter_mut() {
            *x = N::zero();
        }
        for k in 0..d {
            let av = center[k];
            let b_base = k * NR;
            for j in 0..NR {
                let diff = av - b_panel[b_base + j];
                tile[j] += diff * diff;
            }
        }
        out[jj..jj + nc].copy_from_slice(&tile[..nc]);
    }
}
