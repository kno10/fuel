//! ARM NEON-accelerated implementations of the vector math primitives.
//!
//! This backend targets aarch64, including Apple Silicon.  NEON is mandatory on
//! aarch64, so the dispatcher can call these functions without runtime feature
//! checks.  The dense pairwise kernels and hot one-vector primitives use
//! explicit NEON intrinsics, with the unrolled backend kept for strided row
//! fallbacks.

use std::arch::aarch64::*;

use ndarray::ArrayView2;

use super::unroll;

#[inline(always)]
pub fn sqdist_f32(v1: &[f32], v2: &[f32], d: usize) -> f32 {
    unsafe {
        assert!(v1.len() >= d && v2.len() >= d);
        let mut i = 0;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        while i + 16 <= d {
            let a0 = vld1q_f32(v1.as_ptr().add(i));
            let b0 = vld1q_f32(v2.as_ptr().add(i));
            let diff0 = vsubq_f32(a0, b0);
            acc0 = vfmaq_f32(acc0, diff0, diff0);

            let a1 = vld1q_f32(v1.as_ptr().add(i + 4));
            let b1 = vld1q_f32(v2.as_ptr().add(i + 4));
            let diff1 = vsubq_f32(a1, b1);
            acc1 = vfmaq_f32(acc1, diff1, diff1);

            let a2 = vld1q_f32(v1.as_ptr().add(i + 8));
            let b2 = vld1q_f32(v2.as_ptr().add(i + 8));
            let diff2 = vsubq_f32(a2, b2);
            acc2 = vfmaq_f32(acc2, diff2, diff2);

            let a3 = vld1q_f32(v1.as_ptr().add(i + 12));
            let b3 = vld1q_f32(v2.as_ptr().add(i + 12));
            let diff3 = vsubq_f32(a3, b3);
            acc3 = vfmaq_f32(acc3, diff3, diff3);
            i += 16;
        }

        let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i + 4 <= d {
            let a = vld1q_f32(v1.as_ptr().add(i));
            let b = vld1q_f32(v2.as_ptr().add(i));
            let diff = vsubq_f32(a, b);
            acc = vfmaq_f32(acc, diff, diff);
            i += 4;
        }

        let mut sum = vaddvq_f32(acc);
        while i < d {
            let x = *v1.get_unchecked(i) - *v2.get_unchecked(i);
            sum = x.mul_add(x, sum);
            i += 1;
        }
        sum
    }
}

#[inline(always)]
pub fn sqdist_f64(v1: &[f64], v2: &[f64], d: usize) -> f64 {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    let mut acc0 = unsafe { vdupq_n_f64(0.0) };
    let mut acc1 = unsafe { vdupq_n_f64(0.0) };
    let mut acc2 = unsafe { vdupq_n_f64(0.0) };
    let mut acc3 = unsafe { vdupq_n_f64(0.0) };

    while i + 8 <= d {
        unsafe {
            let a0 = vld1q_f64(v1.as_ptr().add(i));
            let b0 = vld1q_f64(v2.as_ptr().add(i));
            let diff0 = vsubq_f64(a0, b0);
            acc0 = vfmaq_f64(acc0, diff0, diff0);

            let a1 = vld1q_f64(v1.as_ptr().add(i + 2));
            let b1 = vld1q_f64(v2.as_ptr().add(i + 2));
            let diff1 = vsubq_f64(a1, b1);
            acc1 = vfmaq_f64(acc1, diff1, diff1);

            let a2 = vld1q_f64(v1.as_ptr().add(i + 4));
            let b2 = vld1q_f64(v2.as_ptr().add(i + 4));
            let diff2 = vsubq_f64(a2, b2);
            acc2 = vfmaq_f64(acc2, diff2, diff2);

            let a3 = vld1q_f64(v1.as_ptr().add(i + 6));
            let b3 = vld1q_f64(v2.as_ptr().add(i + 6));
            let diff3 = vsubq_f64(a3, b3);
            acc3 = vfmaq_f64(acc3, diff3, diff3);
        }
        i += 8;
    }

    let mut acc = unsafe { vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3)) };
    while i + 2 <= d {
        unsafe {
            let a = vld1q_f64(v1.as_ptr().add(i));
            let b = vld1q_f64(v2.as_ptr().add(i));
            let diff = vsubq_f64(a, b);
            acc = vfmaq_f64(acc, diff, diff);
        }
        i += 2;
    }

    let mut sum = unsafe { vaddvq_f64(acc) };
    while i < d {
        let x = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
        sum = x.mul_add(x, sum);
        i += 1;
    }
    sum
}

#[inline(always)]
pub fn l1dist_f32(v1: &[f32], v2: &[f32], d: usize) -> f32 {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    let mut acc0 = unsafe { vdupq_n_f32(0.0) };
    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };
    let mut acc3 = unsafe { vdupq_n_f32(0.0) };

    while i + 16 <= d {
        unsafe {
            let d0 =
                vabsq_f32(vsubq_f32(vld1q_f32(v1.as_ptr().add(i)), vld1q_f32(v2.as_ptr().add(i))));
            let d1 = vabsq_f32(vsubq_f32(
                vld1q_f32(v1.as_ptr().add(i + 4)),
                vld1q_f32(v2.as_ptr().add(i + 4)),
            ));
            let d2 = vabsq_f32(vsubq_f32(
                vld1q_f32(v1.as_ptr().add(i + 8)),
                vld1q_f32(v2.as_ptr().add(i + 8)),
            ));
            let d3 = vabsq_f32(vsubq_f32(
                vld1q_f32(v1.as_ptr().add(i + 12)),
                vld1q_f32(v2.as_ptr().add(i + 12)),
            ));
            acc0 = vaddq_f32(acc0, d0);
            acc1 = vaddq_f32(acc1, d1);
            acc2 = vaddq_f32(acc2, d2);
            acc3 = vaddq_f32(acc3, d3);
        }
        i += 16;
    }

    let mut acc = unsafe { vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)) };
    while i + 4 <= d {
        unsafe {
            let diff =
                vabsq_f32(vsubq_f32(vld1q_f32(v1.as_ptr().add(i)), vld1q_f32(v2.as_ptr().add(i))));
            acc = vaddq_f32(acc, diff);
        }
        i += 4;
    }

    let mut sum = unsafe { vaddvq_f32(acc) };
    while i < d {
        sum += unsafe { (*v1.get_unchecked(i) - *v2.get_unchecked(i)).abs() };
        i += 1;
    }
    sum
}

#[inline(always)]
pub fn l1dist_f64(v1: &[f64], v2: &[f64], d: usize) -> f64 {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    let mut acc0 = unsafe { vdupq_n_f64(0.0) };
    let mut acc1 = unsafe { vdupq_n_f64(0.0) };
    let mut acc2 = unsafe { vdupq_n_f64(0.0) };
    let mut acc3 = unsafe { vdupq_n_f64(0.0) };

    while i + 8 <= d {
        unsafe {
            let d0 =
                vabsq_f64(vsubq_f64(vld1q_f64(v1.as_ptr().add(i)), vld1q_f64(v2.as_ptr().add(i))));
            let d1 = vabsq_f64(vsubq_f64(
                vld1q_f64(v1.as_ptr().add(i + 2)),
                vld1q_f64(v2.as_ptr().add(i + 2)),
            ));
            let d2 = vabsq_f64(vsubq_f64(
                vld1q_f64(v1.as_ptr().add(i + 4)),
                vld1q_f64(v2.as_ptr().add(i + 4)),
            ));
            let d3 = vabsq_f64(vsubq_f64(
                vld1q_f64(v1.as_ptr().add(i + 6)),
                vld1q_f64(v2.as_ptr().add(i + 6)),
            ));
            acc0 = vaddq_f64(acc0, d0);
            acc1 = vaddq_f64(acc1, d1);
            acc2 = vaddq_f64(acc2, d2);
            acc3 = vaddq_f64(acc3, d3);
        }
        i += 8;
    }

    let mut acc = unsafe { vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3)) };
    while i + 2 <= d {
        unsafe {
            let diff =
                vabsq_f64(vsubq_f64(vld1q_f64(v1.as_ptr().add(i)), vld1q_f64(v2.as_ptr().add(i))));
            acc = vaddq_f64(acc, diff);
        }
        i += 2;
    }

    let mut sum = unsafe { vaddvq_f64(acc) };
    while i < d {
        sum += unsafe { (*v1.get_unchecked(i) - *v2.get_unchecked(i)).abs() };
        i += 1;
    }
    sum
}

#[inline(always)]
pub fn mul_f32(v1: &mut [f32], v2: &[f32], a: f32, d: usize) {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    let aa = unsafe { vdupq_n_f32(a) };
    while i + 4 <= d {
        unsafe {
            vst1q_f32(v1.as_mut_ptr().add(i), vmulq_f32(vld1q_f32(v2.as_ptr().add(i)), aa));
        }
        i += 4;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a };
        i += 1;
    }
}

#[inline(always)]
pub fn mul_f64(v1: &mut [f64], v2: &[f64], a: f64, d: usize) {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    let aa = unsafe { vdupq_n_f64(a) };
    while i + 2 <= d {
        unsafe {
            vst1q_f64(v1.as_mut_ptr().add(i), vmulq_f64(vld1q_f64(v2.as_ptr().add(i)), aa));
        }
        i += 2;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a };
        i += 1;
    }
}

#[inline(always)]
pub fn mul_assign_f32(v: &mut [f32], f: f32, d: usize) {
    assert!(v.len() >= d);
    let mut i = 0;
    let ff = unsafe { vdupq_n_f32(f) };
    while i + 4 <= d {
        unsafe {
            vst1q_f32(v.as_mut_ptr().add(i), vmulq_f32(vld1q_f32(v.as_ptr().add(i)), ff));
        }
        i += 4;
    }
    while i < d {
        unsafe { *v.get_unchecked_mut(i) *= f };
        i += 1;
    }
}

#[inline(always)]
pub fn mul_assign_f64(v: &mut [f64], f: f64, d: usize) {
    assert!(v.len() >= d);
    let mut i = 0;
    let ff = unsafe { vdupq_n_f64(f) };
    while i + 2 <= d {
        unsafe {
            vst1q_f64(v.as_mut_ptr().add(i), vmulq_f64(vld1q_f64(v.as_ptr().add(i)), ff));
        }
        i += 2;
    }
    while i < d {
        unsafe { *v.get_unchecked_mut(i) *= f };
        i += 1;
    }
}

#[inline(always)]
pub fn add_assign_f32(v1: &mut [f32], v2: &[f32], d: usize) {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    while i + 4 <= d {
        unsafe {
            vst1q_f32(
                v1.as_mut_ptr().add(i),
                vaddq_f32(vld1q_f32(v1.as_ptr().add(i)), vld1q_f32(v2.as_ptr().add(i))),
            );
        }
        i += 4;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) += *v2.get_unchecked(i) };
        i += 1;
    }
}

#[inline(always)]
pub fn add_assign_f64(v1: &mut [f64], v2: &[f64], d: usize) {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    while i + 2 <= d {
        unsafe {
            vst1q_f64(
                v1.as_mut_ptr().add(i),
                vaddq_f64(vld1q_f64(v1.as_ptr().add(i)), vld1q_f64(v2.as_ptr().add(i))),
            );
        }
        i += 2;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) += *v2.get_unchecked(i) };
        i += 1;
    }
}

#[inline(always)]
pub fn sub_assign_f32(v1: &mut [f32], v2: &[f32], d: usize) {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    while i + 4 <= d {
        unsafe {
            vst1q_f32(
                v1.as_mut_ptr().add(i),
                vsubq_f32(vld1q_f32(v1.as_ptr().add(i)), vld1q_f32(v2.as_ptr().add(i))),
            );
        }
        i += 4;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i) };
        i += 1;
    }
}

#[inline(always)]
pub fn sub_assign_f64(v1: &mut [f64], v2: &[f64], d: usize) {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    while i + 2 <= d {
        unsafe {
            vst1q_f64(
                v1.as_mut_ptr().add(i),
                vsubq_f64(vld1q_f64(v1.as_ptr().add(i)), vld1q_f64(v2.as_ptr().add(i))),
            );
        }
        i += 2;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i) };
        i += 1;
    }
}

#[inline(always)]
pub fn fmamul_f32(v1: &mut [f32], a: f32, v2: &[f32], b: f32, d: usize) {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    unsafe {
        let aa = vdupq_n_f32(a);
        let bb = vdupq_n_f32(b);
        while i + 4 <= d {
            let x = vld1q_f32(v1.as_ptr().add(i));
            let y = vld1q_f32(v2.as_ptr().add(i));
            vst1q_f32(v1.as_mut_ptr().add(i), vmulq_f32(vfmaq_f32(y, x, aa), bb));
            i += 4;
        }
        while i < d {
            *v1.get_unchecked_mut(i) = (*v1.get_unchecked(i)).mul_add(a, *v2.get_unchecked(i)) * b;
            i += 1;
        }
    }
}

#[inline(always)]
pub fn fmamul_f64(v1: &mut [f64], a: f64, v2: &[f64], b: f64, d: usize) {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    unsafe {
        let aa = vdupq_n_f64(a);
        let bb = vdupq_n_f64(b);
        while i + 2 <= d {
            let x = vld1q_f64(v1.as_ptr().add(i));
            let y = vld1q_f64(v2.as_ptr().add(i));
            vst1q_f64(v1.as_mut_ptr().add(i), vmulq_f64(vfmaq_f64(y, x, aa), bb));
            i += 2;
        }
        while i < d {
            *v1.get_unchecked_mut(i) = (*v1.get_unchecked(i)).mul_add(a, *v2.get_unchecked(i)) * b;
            i += 1;
        }
    }
}

#[inline(always)]
pub fn dot_f32(v1: &[f32], v2: &[f32], d: usize) -> f32 {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        while i + 16 <= d {
            acc0 = vfmaq_f32(acc0, vld1q_f32(v1.as_ptr().add(i)), vld1q_f32(v2.as_ptr().add(i)));
            acc1 = vfmaq_f32(
                acc1,
                vld1q_f32(v1.as_ptr().add(i + 4)),
                vld1q_f32(v2.as_ptr().add(i + 4)),
            );
            acc2 = vfmaq_f32(
                acc2,
                vld1q_f32(v1.as_ptr().add(i + 8)),
                vld1q_f32(v2.as_ptr().add(i + 8)),
            );
            acc3 = vfmaq_f32(
                acc3,
                vld1q_f32(v1.as_ptr().add(i + 12)),
                vld1q_f32(v2.as_ptr().add(i + 12)),
            );
            i += 16;
        }

        let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i + 4 <= d {
            acc = vfmaq_f32(acc, vld1q_f32(v1.as_ptr().add(i)), vld1q_f32(v2.as_ptr().add(i)));
            i += 4;
        }

        let mut sum = vaddvq_f32(acc);
        while i < d {
            sum += *v1.get_unchecked(i) * *v2.get_unchecked(i);
            i += 1;
        }
        sum
    }
}

#[inline(always)]
pub fn dot_f64(v1: &[f64], v2: &[f64], d: usize) -> f64 {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    unsafe {
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);
        let mut acc2 = vdupq_n_f64(0.0);
        let mut acc3 = vdupq_n_f64(0.0);

        while i + 8 <= d {
            acc0 = vfmaq_f64(acc0, vld1q_f64(v1.as_ptr().add(i)), vld1q_f64(v2.as_ptr().add(i)));
            acc1 = vfmaq_f64(
                acc1,
                vld1q_f64(v1.as_ptr().add(i + 2)),
                vld1q_f64(v2.as_ptr().add(i + 2)),
            );
            acc2 = vfmaq_f64(
                acc2,
                vld1q_f64(v1.as_ptr().add(i + 4)),
                vld1q_f64(v2.as_ptr().add(i + 4)),
            );
            acc3 = vfmaq_f64(
                acc3,
                vld1q_f64(v1.as_ptr().add(i + 6)),
                vld1q_f64(v2.as_ptr().add(i + 6)),
            );
            i += 8;
        }

        let mut acc = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
        while i + 2 <= d {
            acc = vfmaq_f64(acc, vld1q_f64(v1.as_ptr().add(i)), vld1q_f64(v2.as_ptr().add(i)));
            i += 2;
        }

        let mut sum = vaddvq_f64(acc);
        while i < d {
            sum += *v1.get_unchecked(i) * *v2.get_unchecked(i);
            i += 1;
        }
        sum
    }
}

#[inline(always)]
pub fn pairwise_sqdist_between_f32(
    points1: ArrayView2<'_, f32>, points2: ArrayView2<'_, f32>, d: usize, out: &mut [f32],
    nrows: usize, ncols: usize,
) {
    const MR: usize = 4;
    const NR: usize = 4;
    assert_eq!(out.len(), nrows * ncols);

    let mut b_panel = vec![0.0f32; NR * d];
    let mut lane = [0.0f32; NR];

    for jj in (0..ncols).step_by(NR) {
        let nc = (ncols - jj).min(NR);
        b_panel.fill(0.0);
        for j_local in 0..nc {
            let row = points2.row(jj + j_local);
            for k in 0..d {
                b_panel[k * NR + j_local] = row[k];
            }
        }

        for ii in (0..nrows).step_by(MR) {
            let nr = (nrows - ii).min(MR);
            let mut acc = [unsafe { vdupq_n_f32(0.0) }; MR];

            for k in 0..d {
                let b = unsafe { vld1q_f32(b_panel.as_ptr().add(k * NR)) };
                for i_local in 0..nr {
                    let av = unsafe { vdupq_n_f32(*points1.uget((ii + i_local, k))) };
                    let diff = unsafe { vsubq_f32(av, b) };
                    acc[i_local] = unsafe { vfmaq_f32(acc[i_local], diff, diff) };
                }
            }

            for i_local in 0..nr {
                unsafe { vst1q_f32(lane.as_mut_ptr(), acc[i_local]) };
                out[(ii + i_local) * ncols + jj..(ii + i_local) * ncols + jj + nc]
                    .copy_from_slice(&lane[..nc]);
            }
        }
    }
}

#[inline(always)]
pub fn pairwise_sqdist_between_f64(
    points1: ArrayView2<'_, f64>, points2: ArrayView2<'_, f64>, d: usize, out: &mut [f64],
    nrows: usize, ncols: usize,
) {
    const MR: usize = 4;
    const NR: usize = 2;
    assert_eq!(out.len(), nrows * ncols);

    let mut b_panel = vec![0.0f64; NR * d];
    let mut lane = [0.0f64; NR];

    for jj in (0..ncols).step_by(NR) {
        let nc = (ncols - jj).min(NR);
        b_panel.fill(0.0);
        for j_local in 0..nc {
            let row = points2.row(jj + j_local);
            for k in 0..d {
                b_panel[k * NR + j_local] = row[k];
            }
        }

        for ii in (0..nrows).step_by(MR) {
            let nr = (nrows - ii).min(MR);
            let mut acc = [unsafe { vdupq_n_f64(0.0) }; MR];

            for k in 0..d {
                let b = unsafe { vld1q_f64(b_panel.as_ptr().add(k * NR)) };
                for i_local in 0..nr {
                    let av = unsafe { vdupq_n_f64(*points1.uget((ii + i_local, k))) };
                    let diff = unsafe { vsubq_f64(av, b) };
                    acc[i_local] = unsafe { vfmaq_f64(acc[i_local], diff, diff) };
                }
            }

            for i_local in 0..nr {
                unsafe { vst1q_f64(lane.as_mut_ptr(), acc[i_local]) };
                out[(ii + i_local) * ncols + jj..(ii + i_local) * ncols + jj + nc]
                    .copy_from_slice(&lane[..nc]);
            }
        }
    }
}

#[inline(always)]
pub fn axpy_f32(v1: &mut [f32], a: f32, v2: &[f32], d: usize) {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    let aa = unsafe { vdupq_n_f32(a) };
    while i + 4 <= d {
        unsafe {
            let x = vld1q_f32(v1.as_ptr().add(i));
            let y = vld1q_f32(v2.as_ptr().add(i));
            vst1q_f32(v1.as_mut_ptr().add(i), vfmaq_f32(x, y, aa));
        }
        i += 4;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) += a * *v2.get_unchecked(i) };
        i += 1;
    }
}

#[inline(always)]
pub fn axpy_f64(v1: &mut [f64], a: f64, v2: &[f64], d: usize) {
    assert!(v1.len() >= d && v2.len() >= d);
    let mut i = 0;
    let aa = unsafe { vdupq_n_f64(a) };
    while i + 2 <= d {
        unsafe {
            let x = vld1q_f64(v1.as_ptr().add(i));
            let y = vld1q_f64(v2.as_ptr().add(i));
            vst1q_f64(v1.as_mut_ptr().add(i), vfmaq_f64(x, y, aa));
        }
        i += 2;
    }
    while i < d {
        unsafe { *v1.get_unchecked_mut(i) += a * *v2.get_unchecked(i) };
        i += 1;
    }
}

#[inline(always)]
pub fn sum_f32(v: &[f32], d: usize) -> f32 {
    assert!(v.len() >= d);
    let mut i = 0;
    let mut acc0 = unsafe { vdupq_n_f32(0.0) };
    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };
    let mut acc3 = unsafe { vdupq_n_f32(0.0) };

    while i + 16 <= d {
        unsafe {
            acc0 = vaddq_f32(acc0, vld1q_f32(v.as_ptr().add(i)));
            acc1 = vaddq_f32(acc1, vld1q_f32(v.as_ptr().add(i + 4)));
            acc2 = vaddq_f32(acc2, vld1q_f32(v.as_ptr().add(i + 8)));
            acc3 = vaddq_f32(acc3, vld1q_f32(v.as_ptr().add(i + 12)));
        }
        i += 16;
    }

    let mut acc = unsafe { vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)) };
    while i + 4 <= d {
        unsafe {
            acc = vaddq_f32(acc, vld1q_f32(v.as_ptr().add(i)));
        }
        i += 4;
    }

    let mut sum = unsafe { vaddvq_f32(acc) };
    while i < d {
        sum += unsafe { *v.get_unchecked(i) };
        i += 1;
    }
    sum
}

#[inline(always)]
pub fn sum_f64(v: &[f64], d: usize) -> f64 {
    assert!(v.len() >= d);
    let mut i = 0;
    let mut acc0 = unsafe { vdupq_n_f64(0.0) };
    let mut acc1 = unsafe { vdupq_n_f64(0.0) };
    let mut acc2 = unsafe { vdupq_n_f64(0.0) };
    let mut acc3 = unsafe { vdupq_n_f64(0.0) };

    while i + 8 <= d {
        unsafe {
            acc0 = vaddq_f64(acc0, vld1q_f64(v.as_ptr().add(i)));
            acc1 = vaddq_f64(acc1, vld1q_f64(v.as_ptr().add(i + 2)));
            acc2 = vaddq_f64(acc2, vld1q_f64(v.as_ptr().add(i + 4)));
            acc3 = vaddq_f64(acc3, vld1q_f64(v.as_ptr().add(i + 6)));
        }
        i += 8;
    }

    let mut acc = unsafe { vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3)) };
    while i + 2 <= d {
        unsafe {
            acc = vaddq_f64(acc, vld1q_f64(v.as_ptr().add(i)));
        }
        i += 2;
    }

    let mut sum = unsafe { vaddvq_f64(acc) };
    while i < d {
        sum += unsafe { *v.get_unchecked(i) };
        i += 1;
    }
    sum
}

#[inline(always)]
pub fn add_scalar_f32(v: &mut [f32], s: f32, d: usize) {
    assert!(v.len() >= d);
    let mut i = 0;
    let ss = unsafe { vdupq_n_f32(s) };
    while i + 4 <= d {
        unsafe {
            vst1q_f32(v.as_mut_ptr().add(i), vaddq_f32(vld1q_f32(v.as_ptr().add(i)), ss));
        }
        i += 4;
    }
    while i < d {
        unsafe { *v.get_unchecked_mut(i) += s };
        i += 1;
    }
}

#[inline(always)]
pub fn add_scalar_f64(v: &mut [f64], s: f64, d: usize) {
    assert!(v.len() >= d);
    let mut i = 0;
    let ss = unsafe { vdupq_n_f64(s) };
    while i + 2 <= d {
        unsafe {
            vst1q_f64(v.as_mut_ptr().add(i), vaddq_f64(vld1q_f64(v.as_ptr().add(i)), ss));
        }
        i += 2;
    }
    while i < d {
        unsafe { *v.get_unchecked_mut(i) += s };
        i += 1;
    }
}

#[inline(always)]
pub fn rowdist_f32(
    center: &[f32], points: ArrayView2<'_, f32>, d: usize, out: &mut [f32], n: usize,
) {
    let strides = points.strides();
    if strides[1] == 1 {
        assert_eq!(out.len(), n);
        for j in 0..n {
            let row = points.row(j);
            out[j] = sqdist_f32(center, row.as_slice().expect("C-order row is contiguous"), d);
        }
        return;
    }

    if strides[0] == 1 && strides[1] > 0 {
        const NR: usize = 4;
        let col_stride = strides[1] as usize;
        let data_ptr = points.as_ptr();
        let n_b_blocks = n.div_ceil(NR);
        let mut tile = [0f32; NR];
        for jj_block in 0..n_b_blocks {
            let jj = jj_block * NR;
            let nc = (n - jj).min(NR);
            if nc == NR {
                unsafe {
                    micro_rowdist_fortran_f32(
                        d,
                        center.as_ptr(),
                        data_ptr.add(jj),
                        col_stride,
                        tile.as_mut_ptr(),
                    );
                }
                out[jj..jj + NR].copy_from_slice(&tile);
            } else {
                for j_local in 0..nc {
                    let j = jj + j_local;
                    let mut sum = 0f32;
                    for (k, &c) in center.iter().enumerate().take(d) {
                        let diff = c - unsafe { *data_ptr.add(j + k * col_stride) };
                        sum += diff * diff;
                    }
                    out[j] = sum;
                }
            }
        }
        return;
    }
    unroll::rowdist(center, points, d, out, n);
}

#[inline(always)]
pub fn rowdist_f64(
    center: &[f64], points: ArrayView2<'_, f64>, d: usize, out: &mut [f64], n: usize,
) {
    let strides = points.strides();
    if strides[1] == 1 {
        assert_eq!(out.len(), n);
        for j in 0..n {
            let row = points.row(j);
            out[j] = sqdist_f64(center, row.as_slice().expect("C-order row is contiguous"), d);
        }
        return;
    }

    if strides[0] == 1 && strides[1] > 0 {
        const NR: usize = 2;
        let col_stride = strides[1] as usize;
        let data_ptr = points.as_ptr();
        let n_b_blocks = n.div_ceil(NR);
        let mut tile = [0f64; NR];
        for jj_block in 0..n_b_blocks {
            let jj = jj_block * NR;
            let nc = (n - jj).min(NR);
            if nc == NR {
                unsafe {
                    micro_rowdist_fortran_f64(
                        d,
                        center.as_ptr(),
                        data_ptr.add(jj),
                        col_stride,
                        tile.as_mut_ptr(),
                    );
                }
                out[jj..jj + NR].copy_from_slice(&tile);
            } else {
                for j_local in 0..nc {
                    let j = jj + j_local;
                    let mut sum = 0f64;
                    for (k, &c) in center.iter().enumerate().take(d) {
                        let diff = c - unsafe { *data_ptr.add(j + k * col_stride) };
                        sum += diff * diff;
                    }
                    out[j] = sum;
                }
            }
        }
        return;
    }
    unroll::rowdist(center, points, d, out, n);
}

/// Squared distance from one center to NR=4 points direct from Fortran-order layout.
#[inline(always)]
fn micro_rowdist_fortran_f32(
    d: usize, center: *const f32, base: *const f32, col_stride: usize, out: *mut f32,
) {
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let k_unroll = d / 4;

        for p in 0..k_unroll {
            let base_k = p * 4;

            let b0 = vld1q_f32(base.add(base_k * col_stride));
            let c0 = vdupq_n_f32(*center.add(base_k));
            let d0 = vsubq_f32(c0, b0);
            acc0 = vfmaq_f32(acc0, d0, d0);

            let b1 = vld1q_f32(base.add((base_k + 1) * col_stride));
            let c1 = vdupq_n_f32(*center.add(base_k + 1));
            let d1 = vsubq_f32(c1, b1);
            acc1 = vfmaq_f32(acc1, d1, d1);

            let b2 = vld1q_f32(base.add((base_k + 2) * col_stride));
            let c2 = vdupq_n_f32(*center.add(base_k + 2));
            let d2 = vsubq_f32(c2, b2);
            acc2 = vfmaq_f32(acc2, d2, d2);

            let b3 = vld1q_f32(base.add((base_k + 3) * col_stride));
            let c3 = vdupq_n_f32(*center.add(base_k + 3));
            let d3 = vsubq_f32(c3, b3);
            acc3 = vfmaq_f32(acc3, d3, d3);
        }

        let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        for k in (k_unroll * 4)..d {
            let b = vld1q_f32(base.add(k * col_stride));
            let c = vdupq_n_f32(*center.add(k));
            let diff = vsubq_f32(c, b);
            acc = vfmaq_f32(acc, diff, diff);
        }
        vst1q_f32(out, acc);
    }
}

/// Squared distance from one center to NR=2 points direct from Fortran-order layout.
#[inline(always)]
fn micro_rowdist_fortran_f64(
    d: usize, center: *const f64, base: *const f64, col_stride: usize, out: *mut f64,
) {
    unsafe {
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);
        let mut acc2 = vdupq_n_f64(0.0);
        let mut acc3 = vdupq_n_f64(0.0);

        let k_unroll = d / 4;

        for p in 0..k_unroll {
            let base_k = p * 4;

            let b0 = vld1q_f64(base.add(base_k * col_stride));
            let c0 = vdupq_n_f64(*center.add(base_k));
            let d0 = vsubq_f64(c0, b0);
            acc0 = vfmaq_f64(acc0, d0, d0);

            let b1 = vld1q_f64(base.add((base_k + 1) * col_stride));
            let c1 = vdupq_n_f64(*center.add(base_k + 1));
            let d1 = vsubq_f64(c1, b1);
            acc1 = vfmaq_f64(acc1, d1, d1);

            let b2 = vld1q_f64(base.add((base_k + 2) * col_stride));
            let c2 = vdupq_n_f64(*center.add(base_k + 2));
            let d2 = vsubq_f64(c2, b2);
            acc2 = vfmaq_f64(acc2, d2, d2);

            let b3 = vld1q_f64(base.add((base_k + 3) * col_stride));
            let c3 = vdupq_n_f64(*center.add(base_k + 3));
            let d3 = vsubq_f64(c3, b3);
            acc3 = vfmaq_f64(acc3, d3, d3);
        }

        let mut acc = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
        for k in (k_unroll * 4)..d {
            let b = vld1q_f64(base.add(k * col_stride));
            let c = vdupq_n_f64(*center.add(k));
            let diff = vsubq_f64(c, b);
            acc = vfmaq_f64(acc, diff, diff);
        }
        vst1q_f64(out, acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqdist() {
        crate::math::tests::verify_sqdist_f32(|v1, v2, d| sqdist_f32(v1, v2, d));
        crate::math::tests::verify_sqdist_f64(|v1, v2, d| sqdist_f64(v1, v2, d));
    }

    #[test]
    fn test_l1dist() {
        crate::math::tests::verify_l1dist_f32(|v1, v2, d| l1dist_f32(v1, v2, d));
        crate::math::tests::verify_l1dist_f64(|v1, v2, d| l1dist_f64(v1, v2, d));
    }

    #[test]
    fn test_mul() {
        crate::math::tests::verify_mul_f32(|v1, v2, a, d| mul_f32(v1, v2, a, d));
        crate::math::tests::verify_mul_f64(|v1, v2, a, d| mul_f64(v1, v2, a, d));
    }

    #[test]
    fn test_mul_assign() {
        crate::math::tests::verify_mul_assign_f32(|v, f, d| mul_assign_f32(v, f, d));
        crate::math::tests::verify_mul_assign_f64(|v, f, d| mul_assign_f64(v, f, d));
    }

    #[test]
    fn test_add_assign() {
        crate::math::tests::verify_add_assign_f32(|v1, v2, d| add_assign_f32(v1, v2, d));
        crate::math::tests::verify_add_assign_f64(|v1, v2, d| add_assign_f64(v1, v2, d));
    }

    #[test]
    fn test_sub_assign() {
        crate::math::tests::verify_sub_assign_f32(|v1, v2, d| sub_assign_f32(v1, v2, d));
        crate::math::tests::verify_sub_assign_f64(|v1, v2, d| sub_assign_f64(v1, v2, d));
    }

    #[test]
    fn test_fmamul() {
        crate::math::tests::verify_fmamul_f32(|v1, a, v2, b, d| fmamul_f32(v1, a, v2, b, d));
        crate::math::tests::verify_fmamul_f64(|v1, a, v2, b, d| fmamul_f64(v1, a, v2, b, d));
    }

    #[test]
    fn test_dot() {
        crate::math::tests::verify_dot_f32(|v1, v2, d| dot_f32(v1, v2, d));
        crate::math::tests::verify_dot_f64(|v1, v2, d| dot_f64(v1, v2, d));
    }

    #[test]
    fn test_axpy() {
        crate::math::tests::verify_axpy_f32(|v1, a, v2, d| axpy_f32(v1, a, v2, d));
        crate::math::tests::verify_axpy_f64(|v1, a, v2, d| axpy_f64(v1, a, v2, d));
    }

    #[test]
    fn test_sum() {
        crate::math::tests::verify_sum_f32(|v, d| sum_f32(v, d));
        crate::math::tests::verify_sum_f64(|v, d| sum_f64(v, d));
    }

    #[test]
    fn test_add_scalar() {
        crate::math::tests::verify_add_scalar_f32(|v, s, d| add_scalar_f32(v, s, d));
        crate::math::tests::verify_add_scalar_f64(|v, s, d| add_scalar_f64(v, s, d));
    }

    #[test]
    fn test_rowdist() {
        crate::math::tests::verify_rowdist_f32(|c, p, d, o, n| rowdist_f32(c, p, d, o, n));
        crate::math::tests::verify_rowdist_f64(|c, p, d, o, n| rowdist_f64(c, p, d, o, n));
    }

    #[test]
    fn test_pairwise_sqdist() {
        crate::math::tests::verify_pairwise_sqdist_f32(|p1, p2, d, o, r, c| {
            pairwise_sqdist_between_f32(p1, p2, d, o, r, c)
        });
        crate::math::tests::verify_pairwise_sqdist_f64(|p1, p2, d, o, r, c| {
            pairwise_sqdist_between_f64(p1, p2, d, o, r, c)
        });
    }
}
