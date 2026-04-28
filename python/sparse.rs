use numpy::{Element, PyReadonlyArray1};
use pyo3::conversion::FromPyObject;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::{Float, VectorData};

pub enum IndexBuffer<'py> {
    I32 { _owner: PyReadonlyArray1<'py, i32>, ptr: *const i32, len: usize },
    I64 { _owner: PyReadonlyArray1<'py, i64>, ptr: *const i64, len: usize },
}

impl IndexBuffer<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::I32 { len, .. } | Self::I64 { len, .. } => *len,
        }
    }

    pub fn try_usize_at(&self, i: usize) -> Option<usize> {
        match self {
            Self::I32 { ptr, .. } => usize::try_from(unsafe { *ptr.add(i) }).ok(),
            Self::I64 { ptr, .. } => usize::try_from(unsafe { *ptr.add(i) }).ok(),
        }
    }

    pub fn usize_at(&self, i: usize) -> usize {
        match self {
            Self::I32 { ptr, .. } => unsafe { *ptr.add(i) as usize },
            Self::I64 { ptr, .. } => unsafe { *ptr.add(i) as usize },
        }
    }
}

pub fn extract_index_buffer<'py>(arr: Bound<'py, PyAny>, name: &str) -> PyResult<IndexBuffer<'py>> {
    if let Ok(owner) = arr.extract::<PyReadonlyArray1<i32>>() {
        let (ptr, len) = {
            let slice = owner.as_slice()?;
            (slice.as_ptr(), slice.len())
        };
        return Ok(IndexBuffer::I32 { _owner: owner, ptr, len });
    }
    if let Ok(owner) = arr.extract::<PyReadonlyArray1<i64>>() {
        let (ptr, len) = {
            let slice = owner.as_slice()?;
            (slice.as_ptr(), slice.len())
        };
        return Ok(IndexBuffer::I64 { _owner: owner, ptr, len });
    }
    Err(PyValueError::new_err(format!("{name} must be a contiguous int32 or int64 numpy array",)))
}

pub struct CsrDataset<'py, N>
where
    N: Element + numpy::Element,
{
    _data_owner: PyReadonlyArray1<'py, N>,
    data_ptr: *const N,
    indices: IndexBuffer<'py>,
    indptr: IndexBuffer<'py>,
    nrows: usize,
    ncols: usize,
}

impl<N: Element> CsrDataset<'_, N> {
    #[inline(always)]
    fn row_bounds(&self, i: usize) -> (usize, usize) {
        (self.indptr.usize_at(i), self.indptr.usize_at(i + 1))
    }
}

impl<'py, N> crate::Data for CsrDataset<'py, N>
where
    N: Element + Copy + numpy::Element,
{
    fn len(&self) -> usize { self.nrows }
}

impl<'py, N> VectorData<N> for CsrDataset<'py, N>
where
    N: Element + Copy + numpy::Element + Float,
{
    fn dims(&self) -> usize { self.ncols }

    fn point(&self, idx: usize) -> &[N] {
        let (start, end) = self.row_bounds(idx);
        let len = end - start;
        unsafe { std::slice::from_raw_parts(self.data_ptr.add(start), len) }
    }

    fn load_into(&self, i: usize, vec: &mut [N], d: usize)
    where
        N: Copy,
    {
        vec[..d].fill(N::zero());
        let (start, end) = self.row_bounds(i);
        for p in start..end {
            let c = self.indices.usize_at(p);
            unsafe {
                *vec.get_unchecked_mut(c) = *self.data_ptr.add(p);
            }
        }
    }
}

pub fn parse_csr_dataset<'py, N>(py: Python<'py>, data: Py<PyAny>) -> PyResult<CsrDataset<'py, N>>
where
    N: Element + Copy,
{
    let data = data.bind(py);
    let format: String = data.getattr("format")?.extract()?;
    if format != "csr" {
        return Err(PyValueError::new_err("data must be a scipy.sparse.csr_matrix (format='csr')"));
    }
    let (nrows, ncols): (usize, usize) = data.getattr("shape")?.extract()?;
    let data_owner = PyReadonlyArray1::<N>::extract((&data.getattr("data")?).into())?;
    let (data_ptr, nnz) = {
        let values = data_owner.as_slice()?;
        (values.as_ptr(), values.len())
    };
    let indices = extract_index_buffer(data.getattr("indices")?, "indices")?;
    let indptr = extract_index_buffer(data.getattr("indptr")?, "indptr")?;

    if nnz != indices.len() {
        return Err(PyValueError::new_err("csr data and indices must have identical length"));
    }
    if indptr.len() != nrows + 1 {
        return Err(PyValueError::new_err("csr indptr length must be nrows + 1"));
    }
    let Some(first) = indptr.try_usize_at(0) else {
        return Err(PyValueError::new_err("csr indptr must be non-negative"));
    };
    if first != 0 {
        return Err(PyValueError::new_err("csr indptr must start at 0"));
    }
    let mut prev = first;
    for r in 0..nrows {
        let Some(start) = indptr.try_usize_at(r) else {
            return Err(PyValueError::new_err("csr indptr must be non-negative"));
        };
        let Some(end) = indptr.try_usize_at(r + 1) else {
            return Err(PyValueError::new_err("csr indptr must be non-negative"));
        };
        if start < prev || end < start || end > nnz {
            return Err(PyValueError::new_err("csr indptr must be monotonic and within nnz range"));
        }
        for p in start..end {
            let Some(c) = indices.try_usize_at(p) else {
                return Err(PyValueError::new_err("csr indices must be non-negative"));
            };
            if c >= ncols {
                return Err(PyValueError::new_err("csr indices out of bounds for shape"));
            }
        }
        prev = end;
    }

    Ok(CsrDataset { _data_owner: data_owner, data_ptr, indices, indptr, nrows, ncols })
}
