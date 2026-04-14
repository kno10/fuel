use ndarray::{ArrayBase, Data as NdData, Ix2, RawData};

use crate::VectorData as Dataset;

/// FIXME: this is old/dead code. A ndarray wrapper in /src/api/ may be a good idea though, if not yet present.
/// Wrapper to use ndarrays as data sets
pub struct NdArrayDataset<'a, N: 'a, A: 'a> {
    data: &'a A,
    marker: std::marker::PhantomData<&'a N>,
}
impl<'a, N, A> NdArrayDataset<'a, N, A> {
    /// Borrow array
    pub fn new(data: &'a A) -> Self {
        // TODO: log a warning if the data is not in standard layout?
        Self { data, marker: std::marker::PhantomData {} }
    }
}
impl<N, D> crate::Data for NdArrayDataset<'_, N, ArrayBase<D, Ix2>>
where
    N: Copy,
    D: RawData<Elem = N> + NdData,
{
    fn len(&self) -> usize { self.data.nrows() }
}

impl<N, D> Dataset<N> for NdArrayDataset<'_, N, ArrayBase<D, Ix2>>
where
    N: Copy,
    D: RawData<Elem = N> + NdData,
{
    fn dims(&self) -> usize { self.data.ncols() }

    fn point(&self, idx: usize) -> &[N] {
        let row = self.data.row(idx);
        unsafe { std::slice::from_raw_parts(row.as_ptr(), self.dims()) }
    }

    fn load_into(&self, i: usize, vec: &mut [N], d: usize) {
        debug_assert_eq!(d, self.dims(), "Dimensionality mismatch");
        let ndrow = self.data.row(i);
        if let Some(s) = ndrow.as_slice() {
            vec.copy_from_slice(s);
        } else {
            // Fallback, not contiguous
            for i in 0..d {
                unsafe {
                    *vec.get_unchecked_mut(i) = *ndrow.uget(i);
                }
            }
        }
    }
}
