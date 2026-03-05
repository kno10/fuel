use crate::{DataAccess, DistanceFunction, MatrixDataAccess};

use super::common::{Builder, MergeHistory};

#[derive(Clone, Copy)]
pub(super) struct IndexedQueryData<'m, 'd, T, DF> {
    pub(super) data: &'m MatrixDataAccess<'d, T, DF>,
    pub(super) query_index: usize,
}

impl<T, DF> DataAccess for IndexedQueryData<'_, '_, T, DF>
where
    DF: DistanceFunction<T>,
{
    fn distance(&self, a: usize, b: usize) -> f64 {
        self.data.distance(a, b)
    }

    fn query_distance(&self, b: usize) -> f64 {
        self.data.distance(self.query_index, b)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

pub(super) struct ClusterBuilder {
    uf_parent: Vec<usize>,
    uf_size: Vec<usize>,
    cluster_id: Vec<usize>,
    merge_builder: Builder<f64>,
    merge_count: usize,
}

impl ClusterBuilder {
    pub(super) fn new(n: usize) -> Self {
        Self {
            uf_parent: (0..n).collect(),
            uf_size: vec![1; n],
            cluster_id: (0..n).collect(),
            merge_builder: Builder::new(n),
            merge_count: 0,
        }
    }

    pub(super) fn merge_count(&self) -> usize {
        self.merge_count
    }

    pub(super) fn find(&mut self, x: usize) -> usize {
        let p = self.uf_parent[x];
        if p != x {
            let r = self.find(p);
            self.uf_parent[x] = r;
            r
        } else {
            x
        }
    }

    pub(super) fn same_set(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }

    pub(super) fn cluster_size_of_point(&mut self, a: usize) -> usize {
        let r = self.find(a);
        self.uf_size[r]
    }

    pub(super) fn merge_points(&mut self, a: usize, b: usize, distance: f64) -> Option<usize> {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return None;
        }

        let cida = self.cluster_id[ra];
        let cidb = self.cluster_id[rb];
        let (h1, h2) = if cida <= cidb {
            (cida, cidb)
        } else {
            (cidb, cida)
        };
        let new_cluster_id = self.merge_builder.add(h1, distance, h2);
        self.merge_count += 1;

        if self.uf_size[ra] < self.uf_size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.uf_parent[rb] = ra;
        self.uf_size[ra] += self.uf_size[rb];
        self.cluster_id[ra] = new_cluster_id;
        Some(ra)
    }

    pub(super) fn into_history(self) -> MergeHistory<f64> {
        self.merge_builder.into_merges()
    }
}
