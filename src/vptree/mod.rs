mod construct;
mod knn_search;
mod priority_search;
mod range_search;

pub use priority_search::PrioritySearcher;

#[allow(non_camel_case_types)]
pub(crate) type vpsize = u32;

/// A pair of distance and index for use in a priority queue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistPair<F = f64> {
    pub(crate) distance: F,
    pub(crate) index: vpsize,
}

impl<F> DistPair<F> {
    pub(crate) fn new(distance: F, index: usize) -> Self {
        Self {
            distance,
            index: index as vpsize,
        }
    }

    /// Distance of this neighbor/candidate to the query point.
    #[must_use]
    pub fn distance(&self) -> F
    where
        F: Copy,
    {
        self.distance
    }

    /// Index of this neighbor/candidate in the backing data set.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index as usize
    }
}

impl<F: PartialEq> Eq for DistPair<F> {}

impl<F: PartialOrd + PartialEq> PartialOrd for DistPair<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: PartialOrd + PartialEq> Ord for DistPair<F> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse the comparison to make BinaryHeap a min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub(crate) struct Bounds<F = f64> {
    pub(crate) lower: F,
    pub(crate) upper: F,
}

impl<F> Bounds<F> {
    pub(crate) const fn new(lower: F, upper: F) -> Self {
        Self { lower, upper }
    }
}

pub struct VPTree<F = f64> {
    points: Vec<vpsize>,
    bounds: Vec<Bounds<F>>,
}

#[cfg(test)]
mod tests;
