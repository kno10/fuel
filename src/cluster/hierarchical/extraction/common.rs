use std::collections::HashMap;

/// Lightweight union-find with path compression and union-by-size.
#[derive(Debug)]
pub(crate) struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self {
        let parent = (0..n).collect();
        Self {
            parent,
            size: vec![1; n],
        }
    }

    pub(crate) fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut node = x;
        while self.parent[node] != root {
            let next = self.parent[node];
            self.parent[node] = root;
            node = next;
        }
        root
    }

    pub(crate) fn union(&mut self, a: usize, b: usize) -> usize {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return ra;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
        ra
    }
}

/// Compress root ids into `0..k-1` labels in first-occurrence order.
pub(crate) fn compress_labels(roots: &[usize]) -> Vec<usize> {
    let mut map = HashMap::with_capacity(roots.len());
    let mut next = 0usize;
    let mut labels = Vec::with_capacity(roots.len());

    for &root in roots {
        let label = *map.entry(root).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        labels.push(label);
    }

    labels
}
