/// Allows for different optimization for different input data types.
pub trait Dataset<N> {
    /// Number of input data points
    fn nrows(&self) -> usize;
    /// Number of variables
    fn ncols(&self) -> usize;
    /// Load a data vector into the scratch space
    fn load_into(&self, i: usize, vec: &mut [N], d: usize);
}
