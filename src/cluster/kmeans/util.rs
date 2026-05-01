use ndarray::{Array2, ArrayBase, ArrayView2, Data, Ix2, RawData};

use crate::{Float, VectorData as Dataset, math};

pub fn compute_loss<N, D, A>(
    data: &A,
    cent: &ArrayBase<D, Ix2>, // TODO: allow more
    assign: &[usize],
) -> N
where
    N: Float + std::fmt::Display,
    D: RawData<Elem = N> + Data,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    (0..n)
        .map(|i| {
            data.load_into(i, &mut scratch, d);
            math::sqdist(cent.row(assign[i]).as_slice().unwrap(), &scratch, d)
        })
        .sum()
}

/// Common return structure for k-means algorithm variants.
///
/// Historically the different routines returned heterogeneous tuples: some
/// provided the final loss (inertia), others returned only a bound and a few
/// merely produced the centres/assignments/iteration count.  In order to
/// unify the interface we introduce a single `KMeansResult` type where
/// optional fields encode information that is not available for every
/// algorithm.
#[derive(Clone, Debug)]
pub struct KMeansResult<N> {
    /// Final cluster centers (shape k x d).
    pub centers: Array2<N>,
    /// Assignment of each point to a cluster index (len = n_samples).
    pub assignments: Vec<usize>,
    /// Number of iterations executed (initialisation counts as iteration 1).
    pub iterations: usize,
    /// Exact inertia (sum of squared distances) if computed by the algorithm.
    pub inertia: Option<N>,
    /// A bound on the inertia (e.g. a cheap lower bound) when available and the
    /// exact value is not returned.  `None` if no such information is provided.
    pub inertia_bound: Option<N>,
}

impl<N> KMeansResult<N> {
    /// Construct a result when the algorithm returns an explicit loss value.
    pub fn with_inertia(
        centers: Array2<N>, assignments: Vec<usize>, iterations: usize, inertia: N,
    ) -> Self {
        Self { centers, assignments, iterations, inertia: Some(inertia), inertia_bound: None }
    }

    /// Construct a result when only a bound on the loss is known.
    pub fn with_bound(
        centers: Array2<N>, assignments: Vec<usize>, iterations: usize, bound: N,
    ) -> Self {
        Self { centers, assignments, iterations, inertia: None, inertia_bound: Some(bound) }
    }

    /// Construct a result when neither loss nor bound is available.
    pub fn without_inertia(centers: Array2<N>, assignments: Vec<usize>, iterations: usize) -> Self {
        Self { centers, assignments, iterations, inertia: None, inertia_bound: None }
    }
}

// Convenience conversions from existing tuple-based return values.  By
// implementing `From` we can write `let result: KMeansResult<_> = algo(...).into();`.

impl<N> From<(Array2<N>, Vec<usize>, usize)> for KMeansResult<N> {
    fn from(tuple: (Array2<N>, Vec<usize>, usize)) -> Self {
        let (centers, assignments, iterations) = tuple;
        KMeansResult::without_inertia(centers, assignments, iterations)
    }
}

impl<N> From<(Array2<N>, Vec<usize>, usize, N)> for KMeansResult<N> {
    fn from(tuple: (Array2<N>, Vec<usize>, usize, N)) -> Self {
        let (centers, assignments, iterations, inertia) = tuple;
        KMeansResult::with_inertia(centers, assignments, iterations, inertia)
    }
}

impl<N> From<(Array2<N>, Vec<usize>, usize, Option<N>)> for KMeansResult<N> {
    fn from(tuple: (Array2<N>, Vec<usize>, usize, Option<N>)) -> Self {
        let (centers, assignments, iterations, bound) = tuple;
        if let Some(i) = bound {
            KMeansResult::with_bound(centers, assignments, iterations, i)
        } else {
            KMeansResult::without_inertia(centers, assignments, iterations)
        }
    }
}

/// Helper for computing fuzzy loss with a specific `Math` implementation.
///
/// This is the same objective function used by the fuzzy Lloyd algorithm.
/// Compute the fuzzy k-means loss (objective) given a membership matrix `U`.
///
/// The returned value is \(\sum_i\sum_j u_{ij}^m \|x_i - c_j\|^2\), where
/// `m` is the fuzziness exponent.
pub fn compute_fuzzy_loss<N, D, A>(
    data: &A, cent: &ArrayBase<D, Ix2>, u: &ArrayBase<D, Ix2>, m: N,
) -> N
where
    N: Float + std::fmt::Display,
    D: RawData<Elem = N> + Data,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let k = cent.nrows();
    let mut scratch = vec![N::zero(); d];
    let mut loss = N::zero();
    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        for j in 0..k {
            let uval = u[[i, j]];
            let um = uval.powf(m);
            let dist = math::sqdist(cent.row(j).as_slice().unwrap(), &scratch, d);
            loss += um * dist;
        }
    }
    loss
}

/// Index into triangular matrix
#[inline]
pub(crate) fn triindex(a: usize, b: usize) -> usize {
    assert!(a != b);
    if a < b { a + ((b * (b - 1)) >> 1) } else { b + ((a * (a - 1)) >> 1) }
}

/// Generate a data set for unit testing
#[cfg(test)]
pub(crate) fn gen_test_data<R>(shape: (usize, usize), mut rng: R) -> Array2<f64>
where
    R: rand::Rng,
{
    let mut mat = Array2::<f64>::from_elem(shape, 0.0);
    // TODO: generate clustered data instead
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            mat[[i, j]] = rand_distr::Distribution::sample(&normal, &mut rng);
        }
    }
    mat
}

/// Centers storage
#[derive(Clone)]
pub struct Centers<N> {
    centers: Array2<N>,
}

impl<N> Centers<N>
where
    N: Float,
{
    #[inline]
    pub fn new(k: usize, d: usize) -> Self {
        Self { centers: Array2::from_elem((k, d), N::zero()) }
    }

    #[inline]
    pub fn as_ndarray(&self) -> ArrayView2<'_, N> { self.centers.view() }

    #[inline(always)]
    pub fn center(&self, i: usize) -> &[N] {
        let cols = self.centers.ncols();
        let slice = self.centers.as_slice_memory_order().unwrap();
        &slice[i * cols..i * cols + cols]
    }

    #[inline(always)]
    pub fn center_mut(&mut self, i: usize) -> &mut [N] {
        let cols = self.centers.ncols();
        let slice = self.centers.as_slice_memory_order_mut().unwrap();
        &mut slice[i * cols..i * cols + cols]
    }

    /// Make a dense matrix from the aligned rows
    #[inline]
    pub fn into_ndarray(self) -> Array2<N> { self.centers }
}

// additional helper methods used for tolerance checks
impl<N> Centers<N>
where
    N: Float,
{
    /// Frobenius norm of all centers treated as a matrix.
    #[inline]
    pub fn frobenius_norm(&self) -> N {
        let mut sum = N::zero();
        for i in 0..self.centers.nrows() {
            let row = self.center(i);
            // dot(row, row) yields squared norm of the row
            sum += math::dot(row, row, self.centers.ncols());
        }
        sum.sqrt()
    }

    /// Frobenius norm of the difference between `self` and `other`.
    #[inline]
    pub fn diff_frobenius_norm(&self, other: &Self) -> N {
        let mut sum = N::zero();
        for i in 0..self.centers.nrows() {
            sum += math::sqdist(self.center(i), other.center(i), self.centers.ncols());
        }
        sum.sqrt()
    }
}
