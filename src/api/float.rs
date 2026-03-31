/// Common Float requirements to keep the source readable...
pub trait Float:
    num_traits::Float
    + Default
    + Copy
    + num_traits::AsPrimitive<Self>
    + num_traits::ToPrimitive
    + for<'a> std::ops::AddAssign<&'a Self>
    + for<'a> std::ops::MulAssign<&'a Self>
    + for<'a> std::ops::SubAssign<&'a Self>
    + for<'a> std::ops::DivAssign<&'a Self>
    + num_traits::MulAdd<Output = Self>
    + std::iter::Sum
    + num_traits::FromPrimitive
    + std::marker::Unpin
{
    fn cast<T: num_traits::NumCast>(x: T) -> Self { num_traits::NumCast::from(x).unwrap() }

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
        + num_traits::ToPrimitive
        + for<'a> std::ops::AddAssign<&'a Self>
        + for<'a> std::ops::MulAssign<&'a Self>
        + for<'a> std::ops::SubAssign<&'a Self>
        + for<'a> std::ops::DivAssign<&'a Self>
        + num_traits::MulAdd<Output = Self>
        + std::iter::Sum
        + num_traits::FromPrimitive
        + std::marker::Unpin,
> Float for T
{
}
