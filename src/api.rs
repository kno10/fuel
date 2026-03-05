/// Interface into a data set for distance calculations.
pub trait DataAccess {
    /// Calculate the distance between two points.
    /// The points are identified by their indices in the data set.
    /// The distance function should be metric symmetric.
    fn distance(&self, a: usize, b: usize) -> f64;

    /// Distance from the current query point.
    /// TODO: move this to a separate trait, `DataQuery` that extends `DataAccess`.
    fn query_distance(&self, b: usize) -> f64;

    /// Get the size of the data set.
    fn size(&self) -> usize;

    /// Allocate a (mutable) vector of indices for the data set.
    fn iter(&self) -> impl Iterator<Item = usize> {
        0..self.size()
    }
}
