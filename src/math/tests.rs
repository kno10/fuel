//! Numerical verification utilities for math primitives.

use ndarray::{Array2, ArrayView2};

use crate::Float;

/// Simple absolute difference check.
pub fn approx_eq<N: Float + std::fmt::Debug>(a: N, b: N, eps: N) -> bool {
    let diff = if a > b { a - b } else { b - a };
    if diff < eps {
        return true;
    }

    let max_val = if a > b { a } else { b };
    diff / max_val < eps
}

/// Standard test sizes to check edge cases (divisibility by 4, 8, etc.).
pub fn get_test_sizes() -> Vec<usize> { vec![7, 8, 9, 15, 16, 17, 31, 32, 33] }

// --- f32 verifiers ---

pub fn verify_sqdist_f32<F>(f: F)
where
    F: Fn(&[f32], &[f32], usize) -> f32,
{
    for d in get_test_sizes() {
        let v1: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..d).map(|i| i as f32 * -0.2).collect();
        let got = f(&v1, &v2, d);
        let expected = crate::math::scalar::sqdist(&v1, &v2, d);
        assert!(approx_eq(got, expected, 1e-6), "d={}: got {:?}, expected {:?}", d, got, expected);
    }
}

pub fn verify_l1dist_f32<F>(f: F)
where
    F: Fn(&[f32], &[f32], usize) -> f32,
{
    for d in get_test_sizes() {
        let v1: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..d).map(|i| i as f32 * -0.2).collect();
        let got = f(&v1, &v2, d);
        let expected = crate::math::scalar::l1dist(&v1, &v2, d);
        assert!(approx_eq(got, expected, 1e-6), "d={}: got {:?}, expected {:?}", d, got, expected);
    }
}

pub fn verify_mul_f32<F>(f: F)
where
    F: Fn(&mut [f32], &[f32], f32, usize),
{
    for d in get_test_sizes() {
        let v2: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let a = 2.5f32;
        let mut v1_got = vec![0.0; d];
        let mut v1_exp = vec![0.0; d];
        f(&mut v1_got, &v2, a, d);
        crate::math::scalar::mul(&mut v1_exp, &v2, a, d);
        for i in 0..d {
            assert!(
                approx_eq(v1_got[i], v1_exp[i], 1e-6),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v1_got[i],
                v1_exp[i]
            );
        }
    }
}

pub fn verify_mul_assign_f32<F>(f: F)
where
    F: Fn(&mut [f32], f32, usize),
{
    for d in get_test_sizes() {
        let v_init: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let f_val = 2.5f32;
        let mut v_got = v_init.clone();
        let mut v_exp = v_init.clone();
        f(&mut v_got, f_val, d);
        crate::math::scalar::mul_assign(&mut v_exp, f_val, d);
        for i in 0..d {
            assert!(
                approx_eq(v_got[i], v_exp[i], 1e-6),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v_got[i],
                v_exp[i]
            );
        }
    }
}

pub fn verify_add_assign_f32<F>(f: F)
where
    F: Fn(&mut [f32], &[f32], usize),
{
    for d in get_test_sizes() {
        let v1_init: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..d).map(|i| i as f32 * -0.2).collect();
        let mut v1_got = v1_init.clone();
        let mut v1_exp = v1_init.clone();
        f(&mut v1_got, &v2, d);
        crate::math::scalar::add_assign(&mut v1_exp, &v2, d);
        for i in 0..d {
            assert!(
                approx_eq(v1_got[i], v1_exp[i], 1e-6),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v1_got[i],
                v1_exp[i]
            );
        }
    }
}

pub fn verify_sub_assign_f32<F>(f: F)
where
    F: Fn(&mut [f32], &[f32], usize),
{
    for d in get_test_sizes() {
        let v1_init: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..d).map(|i| i as f32 * -0.2).collect();
        let mut v1_got = v1_init.clone();
        let mut v1_exp = v1_init.clone();
        f(&mut v1_got, &v2, d);
        crate::math::scalar::sub_assign(&mut v1_exp, &v2, d);
        for i in 0..d {
            assert!(
                approx_eq(v1_got[i], v1_exp[i], 1e-6),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v1_got[i],
                v1_exp[i]
            );
        }
    }
}

pub fn verify_fmamul_f32<F>(f: F)
where
    F: Fn(&mut [f32], f32, &[f32], f32, usize),
{
    for d in get_test_sizes() {
        let v1_init: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..d).map(|i| i as f32 * -0.2).collect();
        let a = 1.5f32;
        let b = 0.8f32;
        let mut v1_got = v1_init.clone();
        let mut v1_exp = v1_init.clone();
        f(&mut v1_got, a, &v2, b, d);
        crate::math::scalar::fmamul(&mut v1_exp, a, &v2, b, d);
        for i in 0..d {
            assert!(
                approx_eq(v1_got[i], v1_exp[i], 1e-6),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v1_got[i],
                v1_exp[i]
            );
        }
    }
}

pub fn verify_dot_f32<F>(f: F)
where
    F: Fn(&[f32], &[f32], usize) -> f32,
{
    for d in get_test_sizes() {
        let v1: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..d).map(|i| i as f32 * -0.2).collect();
        let got = f(&v1, &v2, d);
        let expected = crate::math::scalar::dot(&v1, &v2, d);
        assert!(approx_eq(got, expected, 1e-6), "d={}: got {:?}, expected {:?}", d, got, expected);
    }
}

pub fn verify_axpy_f32<F>(f: F)
where
    F: Fn(&mut [f32], f32, &[f32], usize),
{
    for d in get_test_sizes() {
        let v1_init: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..d).map(|i| i as f32 * -0.2).collect();
        let a = 1.5f32;
        let mut v1_got = v1_init.clone();
        let mut v1_exp = v1_init.clone();
        f(&mut v1_got, a, &v2, d);
        crate::math::scalar::axpy(&mut v1_exp, a, &v2, d);
        for i in 0..d {
            assert!(
                approx_eq(v1_got[i], v1_exp[i], 1e-6),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v1_got[i],
                v1_exp[i]
            );
        }
    }
}

pub fn verify_sum_f32<F>(f: F)
where
    F: Fn(&[f32], usize) -> f32,
{
    for d in get_test_sizes() {
        let v: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let got = f(&v, d);
        let expected = crate::math::scalar::sum(&v, d);
        assert!(approx_eq(got, expected, 1e-6), "d={}: got {:?}, expected {:?}", d, got, expected);
    }
}

pub fn verify_add_scalar_f32<F>(f: F)
where
    F: Fn(&mut [f32], f32, usize),
{
    for d in get_test_sizes() {
        let v_init: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let s = 2.5f32;
        let mut v_got = v_init.clone();
        let mut v_exp = v_init.clone();
        f(&mut v_got, s, d);
        crate::math::scalar::add_scalar(&mut v_exp, s, d);
        for i in 0..d {
            assert!(
                approx_eq(v_got[i], v_exp[i], 1e-6),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v_got[i],
                v_exp[i]
            );
        }
    }
}

pub fn verify_rowdist_f32<F>(f: F)
where
    F: Fn(&[f32], ArrayView2<'_, f32>, usize, &mut [f32], usize),
{
    for n in [10, 20] {
        for d in get_test_sizes() {
            let center: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
            let data: Vec<f32> = (0..n * d).map(|i| i as f32 * 0.01).collect();
            let points = Array2::from_shape_vec((n, d), data).unwrap();
            let mut out_got = vec![0.0; n];
            let mut out_exp = vec![0.0; n];
            f(&center, points.view(), d, &mut out_got, n);
            crate::math::scalar::rowdist(&center, points.view(), d, &mut out_exp, n);
            for i in 0..n {
                assert!(
                    approx_eq(out_got[i], out_exp[i], 1e-6),
                    "n={}, d={}, i={}: got {:?}, expected {:?}",
                    n,
                    d,
                    i,
                    out_got[i],
                    out_exp[i]
                );
            }
        }
    }
}

pub fn verify_pairwise_sqdist_f32<F>(f: F)
where
    F: Fn(ArrayView2<'_, f32>, ArrayView2<'_, f32>, usize, &mut [f32], usize, usize),
{
    for nrows in [10, 20] {
        for ncols in [10, 20] {
            for d in get_test_sizes() {
                let p1_data: Vec<f32> = (0..nrows * d).map(|i| i as f32 * 0.01).collect();
                let p2_data: Vec<f32> = (0..ncols * d).map(|i| i as f32 * -0.01).collect();
                let p1 = Array2::from_shape_vec((nrows, d), p1_data).unwrap();
                let p2 = Array2::from_shape_vec((ncols, d), p2_data).unwrap();
                let mut out_got = vec![0.0; nrows * ncols];
                let mut out_exp = vec![0.0; nrows * ncols];
                f(p1.view(), p2.view(), d, &mut out_got, nrows, ncols);
                crate::math::scalar::pairwise_sqdist_between(
                    p1.view(),
                    p2.view(),
                    d,
                    &mut out_exp,
                    nrows,
                    ncols,
                );
                for i in 0..nrows * ncols {
                    assert!(
                        approx_eq(out_got[i], out_exp[i], 1e-6),
                        "nrows={}, ncols={}, d={}, i={}: got {:?}, expected {:?}",
                        nrows,
                        ncols,
                        d,
                        i,
                        out_got[i],
                        out_exp[i]
                    );
                }
            }
        }
    }
}

// --- f64 verifiers ---

pub fn verify_sqdist_f64<F>(f: F)
where
    F: Fn(&[f64], &[f64], usize) -> f64,
{
    for d in get_test_sizes() {
        let v1: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let v2: Vec<f64> = (0..d).map(|i| i as f64 * -0.2).collect();
        let got = f(&v1, &v2, d);
        let expected = crate::math::scalar::sqdist(&v1, &v2, d);
        assert!(approx_eq(got, expected, 1e-12), "d={}: got {:?}, expected {:?}", d, got, expected);
    }
}

pub fn verify_l1dist_f64<F>(f: F)
where
    F: Fn(&[f64], &[f64], usize) -> f64,
{
    for d in get_test_sizes() {
        let v1: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let v2: Vec<f64> = (0..d).map(|i| i as f64 * -0.2).collect();
        let got = f(&v1, &v2, d);
        let expected = crate::math::scalar::l1dist(&v1, &v2, d);
        assert!(approx_eq(got, expected, 1e-12), "d={}: got {:?}, expected {:?}", d, got, expected);
    }
}

pub fn verify_mul_f64<F>(f: F)
where
    F: Fn(&mut [f64], &[f64], f64, usize),
{
    for d in get_test_sizes() {
        let v2: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let a = 2.5f64;
        let mut v1_got = vec![0.0; d];
        let mut v1_exp = vec![0.0; d];
        f(&mut v1_got, &v2, a, d);
        crate::math::scalar::mul(&mut v1_exp, &v2, a, d);
        for i in 0..d {
            assert!(
                approx_eq(v1_got[i], v1_exp[i], 1e-12),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v1_got[i],
                v1_exp[i]
            );
        }
    }
}

pub fn verify_mul_assign_f64<F>(f: F)
where
    F: Fn(&mut [f64], f64, usize),
{
    for d in get_test_sizes() {
        let v_init: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let f_val = 2.5f64;
        let mut v_got = v_init.clone();
        let mut v_exp = v_init.clone();
        f(&mut v_got, f_val, d);
        crate::math::scalar::mul_assign(&mut v_exp, f_val, d);
        for i in 0..d {
            assert!(
                approx_eq(v_got[i], v_exp[i], 1e-12),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v_got[i],
                v_exp[i]
            );
        }
    }
}

pub fn verify_add_assign_f64<F>(f: F)
where
    F: Fn(&mut [f64], &[f64], usize),
{
    for d in get_test_sizes() {
        let v1_init: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let v2: Vec<f64> = (0..d).map(|i| i as f64 * -0.2).collect();
        let mut v1_got = v1_init.clone();
        let mut v1_exp = v1_init.clone();
        f(&mut v1_got, &v2, d);
        crate::math::scalar::add_assign(&mut v1_exp, &v2, d);
        for i in 0..d {
            assert!(
                approx_eq(v1_got[i], v1_exp[i], 1e-12),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v1_got[i],
                v1_exp[i]
            );
        }
    }
}

pub fn verify_sub_assign_f64<F>(f: F)
where
    F: Fn(&mut [f64], &[f64], usize),
{
    for d in get_test_sizes() {
        let v1_init: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let v2: Vec<f64> = (0..d).map(|i| i as f64 * -0.2).collect();
        let mut v1_got = v1_init.clone();
        let mut v1_exp = v1_init.clone();
        f(&mut v1_got, &v2, d);
        crate::math::scalar::sub_assign(&mut v1_exp, &v2, d);
        for i in 0..d {
            assert!(
                approx_eq(v1_got[i], v1_exp[i], 1e-12),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v1_got[i],
                v1_exp[i]
            );
        }
    }
}

pub fn verify_fmamul_f64<F>(f: F)
where
    F: Fn(&mut [f64], f64, &[f64], f64, usize),
{
    for d in get_test_sizes() {
        let v1_init: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let v2: Vec<f64> = (0..d).map(|i| i as f64 * -0.2).collect();
        let a = 1.5f64;
        let b = 0.8f64;
        let mut v1_got = v1_init.clone();
        let mut v1_exp = v1_init.clone();
        f(&mut v1_got, a, &v2, b, d);
        crate::math::scalar::fmamul(&mut v1_exp, a, &v2, b, d);
        for i in 0..d {
            assert!(
                approx_eq(v1_got[i], v1_exp[i], 1e-12),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v1_got[i],
                v1_exp[i]
            );
        }
    }
}

pub fn verify_dot_f64<F>(f: F)
where
    F: Fn(&[f64], &[f64], usize) -> f64,
{
    for d in get_test_sizes() {
        let v1: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let v2: Vec<f64> = (0..d).map(|i| i as f64 * -0.2).collect();
        let got = f(&v1, &v2, d);
        let expected = crate::math::scalar::dot(&v1, &v2, d);
        assert!(approx_eq(got, expected, 1e-12), "d={}: got {:?}, expected {:?}", d, got, expected);
    }
}

pub fn verify_axpy_f64<F>(f: F)
where
    F: Fn(&mut [f64], f64, &[f64], usize),
{
    for d in get_test_sizes() {
        let v1_init: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let v2: Vec<f64> = (0..d).map(|i| i as f64 * -0.2).collect();
        let a = 1.5f64;
        let mut v1_got = v1_init.clone();
        let mut v1_exp = v1_init.clone();
        f(&mut v1_got, a, &v2, d);
        crate::math::scalar::axpy(&mut v1_exp, a, &v2, d);
        for i in 0..d {
            assert!(
                approx_eq(v1_got[i], v1_exp[i], 1e-12),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v1_got[i],
                v1_exp[i]
            );
        }
    }
}

pub fn verify_sum_f64<F>(f: F)
where
    F: Fn(&[f64], usize) -> f64,
{
    for d in get_test_sizes() {
        let v: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let got = f(&v, d);
        let expected = crate::math::scalar::sum(&v, d);
        assert!(approx_eq(got, expected, 1e-12), "d={}: got {:?}, expected {:?}", d, got, expected);
    }
}

pub fn verify_add_scalar_f64<F>(f: F)
where
    F: Fn(&mut [f64], f64, usize),
{
    for d in get_test_sizes() {
        let v_init: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
        let s = 2.5f64;
        let mut v_got = v_init.clone();
        let mut v_exp = v_init.clone();
        f(&mut v_got, s, d);
        crate::math::scalar::add_scalar(&mut v_exp, s, d);
        for i in 0..d {
            assert!(
                approx_eq(v_got[i], v_exp[i], 1e-12),
                "d={}, i={}: got {:?}, expected {:?}",
                d,
                i,
                v_got[i],
                v_exp[i]
            );
        }
    }
}

pub fn verify_rowdist_f64<F>(f: F)
where
    F: Fn(&[f64], ArrayView2<'_, f64>, usize, &mut [f64], usize),
{
    for n in [10, 20] {
        for d in get_test_sizes() {
            let center: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
            let data: Vec<f64> = (0..n * d).map(|i| i as f64 * 0.01).collect();
            let points = Array2::from_shape_vec((n, d), data).unwrap();
            let mut out_got = vec![0.0; n];
            let mut out_exp = vec![0.0; n];
            f(&center, points.view(), d, &mut out_got, n);
            crate::math::scalar::rowdist(&center, points.view(), d, &mut out_exp, n);
            for i in 0..n {
                assert!(
                    approx_eq(out_got[i], out_exp[i], 1e-12),
                    "n={}, d={}, i={}: got {:?}, expected {:?}",
                    n,
                    d,
                    i,
                    out_got[i],
                    out_exp[i]
                );
            }
        }
    }
}

pub fn verify_pairwise_sqdist_f64<F>(f: F)
where
    F: Fn(ArrayView2<'_, f64>, ArrayView2<'_, f64>, usize, &mut [f64], usize, usize),
{
    for nrows in [10, 20] {
        for ncols in [10, 20] {
            for d in get_test_sizes() {
                let p1_data: Vec<f64> = (0..nrows * d).map(|i| i as f64 * 0.01).collect();
                let p2_data: Vec<f64> = (0..ncols * d).map(|i| i as f64 * -0.01).collect();
                let p1 = Array2::from_shape_vec((nrows, d), p1_data).unwrap();
                let p2 = Array2::from_shape_vec((ncols, d), p2_data).unwrap();
                let mut out_got = vec![0.0; nrows * ncols];
                let mut out_exp = vec![0.0; nrows * ncols];
                f(p1.view(), p2.view(), d, &mut out_got, nrows, ncols);
                crate::math::scalar::pairwise_sqdist_between(
                    p1.view(),
                    p2.view(),
                    d,
                    &mut out_exp,
                    nrows,
                    ncols,
                );
                for i in 0..nrows * ncols {
                    assert!(
                        approx_eq(out_got[i], out_exp[i], 1e-12),
                        "nrows={}, ncols={}, d={}, i={}: got {:?}, expected {:?}",
                        nrows,
                        ncols,
                        d,
                        i,
                        out_got[i],
                        out_exp[i]
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod integration {
    use ndarray::ShapeBuilder;

    use super::*;
    use crate::math::pairwise_sqdist;

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

        let rel_eps = N::from_f64(1e-6).unwrap();
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
                err / scale < N::from_f64(1e-6).unwrap(),
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
