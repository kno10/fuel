/// Common Float requirements to keep the source readable...
pub trait Float:
    num_traits::Float
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
    T: num_traits::Float
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
