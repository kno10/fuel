use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::api::data::DistanceData;
use crate::api::float::Float;
use crate::api::query::DistanceSearch;

/// Simple pair of (distance, index) returned by search operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistPair<F> {
    /// Distance from the query point.
    pub distance: F,
    /// Index of the point in the data set.
    pub index: usize,
}

impl<F: Float> Eq for DistPair<F> {}

impl<F: Float> PartialOrd for DistPair<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl<F: Float> Ord for DistPair<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl<F> DistPair<F> {
    /// Construct a new pair.
    pub fn new(distance: F, index: usize) -> Self { Self { distance, index } }
}

impl<F: Float> DistPair<F> {
    /// An undefined value representing an empty candidate.
    ///
    /// Used by algorithms that need a placeholder for "no neighbor yet".
    pub fn undefined() -> Self { Self { distance: F::infinity(), index: usize::MAX } }

    /// Returns `true` if this is the sentinel value.
    pub fn is_sentinel(&self) -> bool { self.index == usize::MAX }
}

/// Candidate queue ordered by increasing distance.
///
/// This is a thin wrapper around a min-heap of `DistPair`s and is used by
/// search algorithms that grow a frontier of candidate points or subtree
/// bounds.
#[derive(Debug, Clone, Default)]
pub struct CandidateHeap<F: Float> {
    heap: BinaryHeap<std::cmp::Reverse<DistPair<F>>>,
}

impl<F: Float> CandidateHeap<F> {
    pub fn new() -> Self { Self { heap: BinaryHeap::new() } }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { heap: BinaryHeap::with_capacity(capacity) }
    }

    pub fn len(&self) -> usize { self.heap.len() }

    pub fn is_empty(&self) -> bool { self.heap.is_empty() }

    pub fn clear(&mut self) { self.heap.clear(); }

    pub fn push(&mut self, pair: DistPair<F>) { self.heap.push(std::cmp::Reverse(pair)); }

    pub fn pop(&mut self) -> Option<DistPair<F>> { self.heap.pop().map(|pair| pair.0) }

    pub fn peek(&self) -> Option<DistPair<F>> { self.heap.peek().copied().map(|pair| pair.0) }
}

/// Helper that tracks the best `k` neighbors and includes all ties at the
/// current kth distance.
#[derive(Debug, Clone)]
pub struct KNNHeap<F: Float> {
    k: usize,
    heap: BinaryHeap<DistPair<F>>,
    ties: Vec<DistPair<F>>,
    k_distance: F,
}

impl<F: Float> KNNHeap<F> {
    /// Create a new heap for a fixed `k`.
    pub fn new(k: usize) -> Self {
        Self { k, heap: BinaryHeap::with_capacity(k), ties: Vec::new(), k_distance: F::infinity() }
    }

    /// Number of stored neighbors, including ties.
    pub fn len(&self) -> usize { self.heap.len() + self.ties.len() }

    /// Whether the heap currently stores no neighbors.
    pub fn is_empty(&self) -> bool { self.heap.is_empty() && self.ties.is_empty() }

    /// Current kth distance, or infinity while fewer than `k` neighbors are stored.
    pub fn k_distance(&self) -> F {
        if self.k == 0 || self.heap.len() < self.k { F::infinity() } else { self.k_distance }
    }

    /// Insert a neighbor and return the current kth distance.
    pub fn insert(&mut self, pair: DistPair<F>) -> F {
        if self.k == 0 {
            return F::infinity();
        }

        if self.heap.len() < self.k {
            self.heap.push(pair);
            if self.heap.len() == self.k {
                self.k_distance = self.heap.peek().unwrap().distance;
            }
            return self.k_distance();
        }

        match pair.distance.partial_cmp(&self.k_distance).unwrap_or(Ordering::Equal) {
            Ordering::Less => {
                let removed = self.heap.pop().expect("full knn heap must not be empty");
                self.heap.push(pair);
                let new_k_distance = self.heap.peek().unwrap().distance;
                if new_k_distance < self.k_distance {
                    self.ties.clear();
                } else if removed.distance == new_k_distance {
                    self.ties.push(removed);
                }
                self.k_distance = new_k_distance;
            }
            Ordering::Equal => self.ties.push(pair),
            Ordering::Greater => {}
        }

        self.k_distance()
    }

    /// Remove all stored neighbors.
    pub fn clear(&mut self) {
        self.heap.clear();
        self.ties.clear();
        self.k_distance = F::infinity();
    }

    /// Convert to an ascending vector including ties.
    pub fn into_vec(self) -> Vec<DistPair<F>> {
        let mut out: Vec<DistPair<F>> = self.heap.into_vec();
        out.extend(self.ties);
        out.sort_unstable();
        out
    }
}

/// Range search over any DistanceData via linear scan.
pub fn linear_scan_range<F, D, Q>(data: &D, query: &Q, radius: F) -> Vec<DistPair<F>>
where
    F: Float,
    D: DistanceData<F>,
    Q: DistanceSearch<F> + ?Sized,
{
    let mut result: Vec<DistPair<F>> = (0..data.len())
        .map(|i| DistPair::new(query.query_distance(i), i))
        .filter(|p| p.distance <= radius)
        .collect();
    result.sort_unstable();
    result
}

/// kNN search over any DistanceData via linear scan, using KNNHeap.
pub fn linear_scan_knn<F, D, Q>(data: &D, query: &Q, k: usize) -> Vec<DistPair<F>>
where
    F: Float,
    D: DistanceData<F>,
    Q: DistanceSearch<F> + ?Sized,
{
    let n = data.len();
    if k == 0 || n == 0 {
        return Vec::new();
    }
    let mut heap = KNNHeap::new(k);
    for i in 0..n {
        heap.insert(DistPair::new(query.query_distance(i), i));
    }
    heap.into_vec()
}

/// A view of a single index-node’s contents during a priority search.
pub struct NodePoints<'a> {
    points: &'a [u32],
}

impl<'a> NodePoints<'a> {
    pub const fn new(points: &'a [u32]) -> Self { Self { points } }

    #[must_use]
    pub fn indices(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.points.iter().map(|&point| point as usize)
    }

    #[must_use]
    pub const fn len(&self) -> usize { self.points.len() }

    #[must_use]
    pub const fn is_empty(&self) -> bool { self.points.is_empty() }

    #[must_use]
    pub fn first_index(&self) -> usize { self.points[0] as usize }
}

/// Filter consulted by priority searchers to prune results.
pub trait SearchFilter {
    fn skip_node(&mut self, _points: NodePoints<'_>) -> bool { false }
    fn skip_point(&mut self, index: usize) -> bool;
}

/// Priority searcher interface for incremental neighbor enumeration.
pub trait PrioritySearcher<F: Float, Q: DistanceSearch<F> + ?Sized> {
    fn reset(&mut self);

    fn reset_with_limits(&mut self, cutoff: F, skip: F);

    fn next(&mut self, query: &Q) -> Option<DistPair<F>>;

    fn next_with_filter<S>(&mut self, query: &Q, filter: &mut S) -> Option<DistPair<F>>
    where
        S: SearchFilter;

    fn all_lower_bound(&self) -> F;

    fn decrease_cutoff(&mut self, threshold: F);
}

/// Generic factory for creating a priority searcher.
pub trait PrioritySearcherFactory<F: Float, Q: DistanceSearch<F> + ?Sized> {
    type Searcher<'a>: PrioritySearcher<F, Q>
    where
        Self: 'a,
        Q: 'a,
        F: 'a;

    fn priority_searcher<'a>(&'a self) -> Self::Searcher<'a>
    where
        Q: 'a;
}

/// Generic kNN search capability.
pub trait KnnSearch<F: Float, Q: DistanceSearch<F> + ?Sized> {
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>>;
}

/// Generic approximate kNN search capability.
pub trait ApproxKnnSearch<F: Float, Q: DistanceSearch<F> + ?Sized> {
    fn search_aknn(&self, query: &Q, k: usize, rate: f32) -> Vec<DistPair<F>>;
}

/// Generic range search capability.
pub trait RangeSearch<F: Float, Q: DistanceSearch<F> + ?Sized> {
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>>;
}

/// Priority searcher performing linear scan on source data.
pub struct LinearScanPrioritySearcher<'a, F, D>
where
    F: Float,
    D: DistanceData<F> + ?Sized,
{
    data: &'a D,
    candidates: Vec<DistPair<F>>,
    position: usize,
    cutoff: F,
    skip: F,
    initialized: bool,
}

impl<'a, F, D> LinearScanPrioritySearcher<'a, F, D>
where
    F: Float,
    D: DistanceData<F> + ?Sized,
{
    pub fn new(data: &'a D) -> Self {
        Self {
            data,
            candidates: Vec::new(),
            position: 0,
            cutoff: F::infinity(),
            skip: F::zero(),
            initialized: false,
        }
    }

    fn reset_state(&mut self) {
        self.position = 0;
        self.cutoff = F::infinity();
        self.skip = F::zero();
        self.initialized = false;
    }

    fn ensure_initialized<QI>(&mut self, query: &QI)
    where
        QI: DistanceSearch<F> + ?Sized,
    {
        if self.initialized {
            return;
        }

        let n = self.data.len();
        let mut candidates = Vec::with_capacity(n);

        for i in 0..n {
            let d = query.query_distance(i);
            if d < self.skip || d > self.cutoff {
                continue;
            }
            candidates.push(DistPair::new(d, i));
        }

        candidates.sort();

        self.candidates = candidates;
        self.position = 0;
        self.initialized = true;
    }

    fn find_next<QI, S>(&mut self, query: &QI, filter: &mut S) -> Option<DistPair<F>>
    where
        QI: DistanceSearch<F> + ?Sized,
        S: SearchFilter,
    {
        self.ensure_initialized(query);

        while let Some(candidate) = self.candidates.get(self.position).copied() {
            self.position += 1;

            if candidate.distance > self.cutoff {
                return None;
            }
            if candidate.distance < self.skip {
                continue;
            }
            if filter.skip_point(candidate.index) {
                continue;
            }
            return Some(candidate);
        }

        None
    }

    fn all_lower_bound_internal(&self) -> F {
        if !self.initialized {
            return F::zero();
        }

        for i in self.position..self.candidates.len() {
            let d = self.candidates[i].distance;
            if d > self.cutoff {
                break;
            }
            if d >= self.skip {
                return d;
            }
        }

        F::infinity()
    }

    fn decrease_cutoff_internal(&mut self, threshold: F) {
        debug_assert!(threshold <= self.cutoff, "Thresholds must only decrease.");
        self.cutoff = threshold;
    }

    pub fn reset(&mut self) { self.reset_state(); }

    pub fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        debug_assert!(skip >= F::zero(), "Skip threshold must be non-negative.");
        debug_assert!(cutoff >= skip, "Cutoff must be >= skip threshold.");
        self.reset_state();
        self.cutoff = cutoff;
        self.skip = skip;
    }

    pub fn all_lower_bound(&self) -> F { self.all_lower_bound_internal() }

    pub fn decrease_cutoff(&mut self, threshold: F) { self.decrease_cutoff_internal(threshold); }
}

struct PointFilter<P> {
    skip_point: P,
}

impl<P> SearchFilter for PointFilter<P>
where
    P: FnMut(usize) -> bool,
{
    fn skip_point(&mut self, index: usize) -> bool { (self.skip_point)(index) }
}

impl<F, Q, D> PrioritySearcher<F, Q> for LinearScanPrioritySearcher<'_, F, D>
where
    F: Float,
    Q: DistanceSearch<F> + ?Sized,
    D: DistanceData<F> + ?Sized,
{
    fn reset(&mut self) { self.reset_state(); }

    fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        debug_assert!(skip >= F::zero(), "Skip threshold must be non-negative.");
        debug_assert!(cutoff >= skip, "Cutoff must be >= skip threshold.");
        self.reset_state();
        self.cutoff = cutoff;
        self.skip = skip;
    }

    fn next(&mut self, query: &Q) -> Option<DistPair<F>> {
        self.next_with_filter(query, &mut PointFilter { skip_point: |_| false })
    }

    fn next_with_filter<S>(&mut self, query: &Q, filter: &mut S) -> Option<DistPair<F>>
    where
        S: SearchFilter,
    {
        self.find_next(query, filter)
    }

    fn all_lower_bound(&self) -> F { self.all_lower_bound_internal() }

    fn decrease_cutoff(&mut self, threshold: F) { self.decrease_cutoff_internal(threshold); }
}
