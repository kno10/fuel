use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::*;

use num_traits::Float; //FIXME: use crate::Float instead

use crate::math::Math as MathTrait;

/// Unrolled vector maths; `LANES` specifies the unroll factor.
///
/// This implementation is intended for use when you know the CPU has a
/// particular vector width and you want to process `LANES` elements in
/// parallel; the drawback is code size, hence it is kept separate.
pub struct UnrollMath<N, const LANES: usize> {
    phantom: PhantomData<N>,
}

impl<N, const LANES: usize> MathTrait<N> for UnrollMath<N, LANES>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
{
    #[inline(always)]
    fn sqdist(v1: &[N], v2: &[N], d: usize) -> N {
        assert!(LANES.count_ones() == 1);
        assert!(v1.len() == d && v2.len() == d);
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
        if d > sd {
            sum += (sd..d)
                .map(|i| unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) })
                .map(|x| x * x)
                .sum()
        }
        sum
    }

    #[inline(always)]
    fn l1dist(v1: &[N], v2: &[N], d: usize) -> N {
        // simple fallback
        assert!(v1.len() == d && v2.len() == d);
        let mut sum = N::zero();
        for i in 0..d {
            let diff = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
            sum += diff.abs();
        }
        sum
    }

    #[inline(always)]
    fn mul(v1: &mut [N], v2: &[N], a: N, d: usize) {
        assert!(LANES.count_ones() == 1);
        assert!(v1.len() == d && v2.len() == d);
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
    fn mul_assign(v: &mut [N], f: N, d: usize) {
        assert!(LANES.count_ones() == 1);
        assert!(v.len() == d);
        let sd = d & !(LANES - 1);
        for i in (0..sd).step_by(LANES) {
            let v2 = &mut v[i..(i + LANES)];
            for j in 0..LANES {
                unsafe {
                    *v2.get_unchecked_mut(j) *= f;
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
    fn add_assign(v1: &mut [N], v2: &[N], d: usize) {
        assert!(LANES.count_ones() == 1);
        assert!(v1.len() == d && v2.len() == d);
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
    fn sub_assign(v1: &mut [N], v2: &[N], d: usize) {
        assert!(LANES.count_ones() == 1);
        assert!(v1.len() == d && v2.len() == d);
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
    fn fmamul(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) {
        assert!(LANES.count_ones() == 1);
        assert!(v1.len() == d && v2.len() == d);
        let sd = d & !(LANES - 1);
        for i in (0..sd).step_by(LANES) {
            let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
            for j in 0..LANES {
                unsafe {
                    *b1.get_unchecked_mut(j) =
                        (*b1.get_unchecked(j) * a + *b2.get_unchecked(j)) * b;
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
    fn dot(v1: &[N], v2: &[N], d: usize) -> N {
        assert!(LANES.count_ones() == 1);
        assert!(v1.len() == d && v2.len() == d);
        let sd = d & !(LANES - 1);
        let mut vsum = [N::zero(); LANES];
        for i in (0..sd).step_by(LANES) {
            let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
            for j in 0..LANES {
                unsafe {
                    let (a, b) = (*vv.get_unchecked(j), *cc.get_unchecked(j));
                    *vsum.get_unchecked_mut(j) += a * b;
                }
            }
        }
        let mut sum = vsum.iter().copied().sum::<N>();
        if d > sd {
            sum += (sd..d).map(|i| unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) }).sum()
        }
        sum
    }

    // new helpers
    #[inline(always)]
    fn axpy(v1: &mut [N], a: N, v2: &[N], d: usize) {
        assert!(LANES.count_ones() == 1);
        assert!(v1.len() == d && v2.len() == d);
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
    fn axpby(v1: &mut [N], a: N, v2: &[N], b: N, d: usize)
    where
        N: Copy,
    {
        assert!(LANES.count_ones() == 1);
        assert!(v1.len() == d && v2.len() == d);
        let sd = d & !(LANES - 1);
        for i in (0..sd).step_by(LANES) {
            let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
            for j in 0..LANES {
                unsafe {
                    *b1.get_unchecked_mut(j) = *b1.get_unchecked(j) * a + *b2.get_unchecked(j) * b;
                }
            }
        }
        for i in sd..d {
            unsafe {
                *v1.get_unchecked_mut(i) = *v1.get_unchecked(i) * a + *v2.get_unchecked(i) * b;
            }
        }
    }

    #[inline(always)]
    fn sum(v: &[N], d: usize) -> N {
        assert!(LANES.count_ones() == 1);
        assert!(v.len() == d);
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
        let mut sum = vsum.iter().copied().sum::<N>();
        if d > sd {
            sum += (sd..d).map(|i| unsafe { *v.get_unchecked(i) }).sum();
        }
        sum
    }

    #[inline(always)]
    fn add_scalar(v: &mut [N], s: N, d: usize) {
        assert!(LANES.count_ones() == 1);
        assert!(v.len() == d);
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
}
