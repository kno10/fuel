use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::*;

use num_traits::Float; //FIXME: use crate::Float instead

use crate::math::Math;

/// Basic, un-optimised implementation of the `Math` trait.
///
/// This is the one used when nothing special is requested; it relies on plain
/// scalar loops and should compile efficiently with LLVM's optimisations.
pub struct DefaultMath<N> {
    phantom: PhantomData<N>,
}

impl<N> Math<N> for DefaultMath<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
{
    #[inline(always)]
    fn sqdist(v1: &[N], v2: &[N], d: usize) -> N {
        assert!(v1.len() == d && v2.len() == d); // bounds check
        (0..d).map(|i| unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) }).map(|x| x * x).sum()
    }

    #[inline(always)]
    fn l1dist(v1: &[N], v2: &[N], d: usize) -> N {
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
        assert!(v1.len() == d && v2.len() == d);
        for i in 0..d {
            unsafe {
                *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a;
            }
        }
    }

    #[inline(always)]
    fn mul_assign(v: &mut [N], f: N, d: usize) {
        assert!(v.len() == d);
        for i in 0..d {
            unsafe {
                *v.get_unchecked_mut(i) *= f;
            }
        }
    }

    #[inline(always)]
    fn add_assign(v1: &mut [N], v2: &[N], d: usize) {
        assert!(v1.len() == d && v2.len() == d);
        for i in 0..d {
            unsafe {
                *v1.get_unchecked_mut(i) += *v2.get_unchecked(i);
            }
        }
    }

    #[inline(always)]
    fn sub_assign(v1: &mut [N], v2: &[N], d: usize) {
        assert!(v1.len() == d && v2.len() == d);
        for i in 0..d {
            unsafe {
                *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i);
            }
        }
    }

    #[inline(always)]
    fn fmamul(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) {
        assert!(v1.len() == d && v2.len() == d);
        for i in 0..d {
            unsafe {
                *v1.get_unchecked_mut(i) = (*v1.get_unchecked(i) * a + *v2.get_unchecked(i)) * b;
            }
        }
    }

    #[inline(always)]
    fn dot(v1: &[N], v2: &[N], d: usize) -> N {
        assert!(v1.len() == d && v2.len() == d);
        (0..d).map(|i| unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) }).sum()
    }

    #[inline(always)]
    fn axpy(v1: &mut [N], a: N, v2: &[N], d: usize) {
        assert!(v1.len() == d && v2.len() == d);
        for i in 0..d {
            unsafe {
                *v1.get_unchecked_mut(i) += *v2.get_unchecked(i) * a;
            }
        }
    }

    #[inline(always)]
    fn sum(v: &[N], d: usize) -> N {
        assert!(v.len() == d);
        (0..d).map(|i| unsafe { *v.get_unchecked(i) }).sum()
    }

    #[inline(always)]
    fn add_scalar(v: &mut [N], s: N, d: usize) {
        assert!(v.len() == d);
        for i in 0..d {
            unsafe {
                *v.get_unchecked_mut(i) += s;
            }
        }
    }
}

// unit tests for the new helpers
#[cfg(test)]
mod tests {
    use super::DefaultMath;
    use crate::math::Math as MathTrait;

    #[test]
    fn basic_axpy_sum_add_scalar() {
        let mut a = vec![1.0f64, 2.0, 3.0];
        let b = vec![0.5f64, -1.0, 2.0];
        DefaultMath::<f64>::axpy(&mut a, 2.0, &b, 3);
        assert_eq!(a, vec![2.0, 0.0, 7.0]);

        let s = DefaultMath::<f64>::sum(&a, 3);
        assert_eq!(s, 9.0);

        DefaultMath::<f64>::add_scalar(&mut a, 1.0, 3);
        assert_eq!(a, vec![3.0, 1.0, 8.0]);
    }
}
