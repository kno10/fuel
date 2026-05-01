use ndarray::ArrayView2;

/// Compile-time-dispatched vector math operations.
///
/// Implemented for [`f32`] and [`f64`] with architecture-appropriate SIMD backends
/// (AVX2+FMA on x86-64, unrolled scalar on other architectures).  Dispatch is
/// fully resolved at monomorphization; no `TypeId` or runtime overhead.
///
/// This trait is a supertrait of [`Float`].  You normally do not need to name
/// it explicitly - using `N: Float` is sufficient.
pub trait VecOps: Copy + Sized + 'static {
    /// Squared Euclidean distance between two length-`d` vectors.
    fn vec_sqdist(v1: &[Self], v2: &[Self], d: usize) -> Self;
    /// Pairwise squared distances between two 2D point sets.
    fn vec_pairwise_sqdist<'a>(
        points1: ArrayView2<'a, Self>, points2: ArrayView2<'a, Self>, d: usize, out: &mut [Self],
        nrows: usize, ncols: usize,
    );
    /// Squared distances from one `center` to each of the `n` rows in `points`.
    fn vec_row_sqdist<'a>(
        center: &[Self], points: ArrayView2<'a, Self>, d: usize, out: &mut [Self], n: usize,
    );
    /// L1 (Manhattan) distance between two length-`d` vectors.
    fn vec_l1dist(v1: &[Self], v2: &[Self], d: usize) -> Self;
    /// Element-wise scaled copy: `v1[i] = v2[i] * a` for `i in 0..d`.
    fn vec_mul(v1: &mut [Self], v2: &[Self], a: Self, d: usize);
    /// In-place scale: `v[i] *= f` for `i in 0..d`.
    fn vec_mul_assign(v: &mut [Self], f: Self, d: usize);
    /// In-place addition: `v1[i] += v2[i]` for `i in 0..d`.
    fn vec_add_assign(v1: &mut [Self], v2: &[Self], d: usize);
    /// In-place subtraction: `v1[i] -= v2[i]` for `i in 0..d`.
    fn vec_sub_assign(v1: &mut [Self], v2: &[Self], d: usize);
    /// FMA then scale: `v1[i] = (v1[i] * a + v2[i]) * b` for `i in 0..d`.
    fn vec_fmamul(v1: &mut [Self], a: Self, v2: &[Self], b: Self, d: usize);
    /// Dot product of two length-`d` vectors.
    fn vec_dot(v1: &[Self], v2: &[Self], d: usize) -> Self;
    /// In-place AXPY: `v1[i] += a * v2[i]` for `i in 0..d`.
    fn vec_axpy(v1: &mut [Self], a: Self, v2: &[Self], d: usize);
    /// Sum of the first `d` elements.
    fn vec_sum(v: &[Self], d: usize) -> Self;
    /// Add a scalar to every element: `v[i] += s` for `i in 0..d`.
    fn vec_add_scalar(v: &mut [Self], s: Self, d: usize);
}

/// Common Float requirements to keep the source readable...
pub trait Float:
    VecOps
    + num_traits::Float
    + Default
    + Copy
    + 'static
    + num_traits::AsPrimitive<Self>
    + num_traits::FromPrimitive
    + num_traits::NumCast
    + std::ops::AddAssign<Self>
    + std::ops::MulAssign<Self>
    + std::ops::SubAssign<Self>
    + std::ops::DivAssign<Self>
    + for<'a> std::ops::AddAssign<&'a Self>
    + for<'a> std::ops::MulAssign<&'a Self>
    + for<'a> std::ops::SubAssign<&'a Self>
    + for<'a> std::ops::DivAssign<&'a Self>
    + num_traits::MulAdd<Output = Self>
    + std::iter::Sum
    + std::iter::Product
    + std::marker::Unpin
    + std::marker::Send
    + std::marker::Sync
{
    fn cast<T: num_traits::NumCast>(x: T) -> Self { num_traits::NumCast::from(x).unwrap() }

    /// Common constants used throughout the code base.
    fn two() -> Self { Self::from_f64(2.0).unwrap() }

    fn four() -> Self { Self::from_f64(4.0).unwrap() }

    fn half() -> Self { Self::from_f64(0.5).unwrap() }

    fn quarter() -> Self { Self::from_f64(0.25).unwrap() }

    /// Convert this value to another float type.
    fn to_float<T: Float>(self) -> T {
        num_traits::cast(self).unwrap_or_else(|| T::from_f64(self.to_f64().unwrap()).unwrap())
    }
}

impl<
    T: VecOps
        + num_traits::Float
        + Default
        + Copy
        + num_traits::AsPrimitive<T>
        + num_traits::FromPrimitive
        + num_traits::NumCast
        + std::ops::AddAssign<Self>
        + std::ops::MulAssign<Self>
        + std::ops::SubAssign<Self>
        + std::ops::DivAssign<Self>
        + for<'a> std::ops::AddAssign<&'a Self>
        + for<'a> std::ops::MulAssign<&'a Self>
        + for<'a> std::ops::SubAssign<&'a Self>
        + for<'a> std::ops::DivAssign<&'a Self>
        + num_traits::MulAdd<Output = Self>
        + std::iter::Sum
        + std::iter::Product
        + std::marker::Unpin
        + std::marker::Send
        + std::marker::Sync
        + 'static,
> Float for T
{
}
