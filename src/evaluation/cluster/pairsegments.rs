use std::collections::{BTreeMap, BTreeSet};

pub const UNCLUSTERED: isize = -1;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Segment {
    pub cluster_ids: Vec<isize>,
    pub pair_count: u64,
}

impl Segment {
    pub fn is_unpaired(&self) -> bool {
        self.cluster_ids.contains(&UNCLUSTERED)
    }

    pub fn is_none(&self) -> bool {
        self.cluster_ids.iter().all(|&x| x == UNCLUSTERED)
    }

    pub fn get_unpaired_clustering_index(&self) -> Option<usize> {
        self.cluster_ids.iter().position(|&x| x == UNCLUSTERED)
    }
}

#[derive(Clone, Debug)]
pub struct Segments {
    clusterings_count: usize,
    num_clusters: Vec<usize>,
    total_objects: usize,
    actual_pairs: u64,
    segments: BTreeMap<Vec<isize>, Segment>,
}

impl Segments {
    pub fn from_labelings(clusterings: &[Vec<isize>]) -> Self {
        assert!(!clusterings.is_empty(), "need at least one clustering");
        let n = clusterings[0].len();
        for labels in clusterings.iter().skip(1) {
            assert_eq!(labels.len(), n, "all labelings must have equal length");
        }

        let clusterings_count = clusterings.len();
        let mut num_clusters = Vec::with_capacity(clusterings_count);
        for labels in clusterings {
            let mut unique = BTreeSet::new();
            for &l in labels {
                unique.insert(l);
            }
            num_clusters.push(unique.len());
        }

        let mut segments: BTreeMap<Vec<isize>, Segment> = BTreeMap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let mut key = Vec::with_capacity(clusterings_count);
                for labels in clusterings {
                    if labels[i] == labels[j] {
                        key.push(labels[i]);
                    } else {
                        key.push(UNCLUSTERED);
                    }
                }
                let entry = segments.entry(key.clone()).or_insert(Segment {
                    cluster_ids: key,
                    pair_count: 0,
                });
                entry.pair_count += 1;
            }
        }

        let actual_pairs = segments.values().map(|s| s.pair_count).sum::<u64>();

        Self {
            clusterings_count,
            num_clusters,
            total_objects: n,
            actual_pairs,
            segments,
        }
    }

    pub fn size(&self) -> usize {
        self.segments.len()
    }

    pub fn segments(&self) -> impl Iterator<Item = &Segment> {
        self.segments.values()
    }

    pub fn get_pair_count(&self, with_unclustered_pairs: bool) -> u64 {
        if with_unclustered_pairs {
            (self.total_objects as u64 * self.total_objects.saturating_sub(1) as u64) >> 1
        } else {
            self.actual_pairs
        }
    }

    pub fn get_clusterings(&self) -> usize {
        self.clusterings_count
    }

    pub fn get_total_cluster_count(&self) -> usize {
        self.num_clusters.iter().sum()
    }

    pub fn get_highest_cluster_count(&self) -> usize {
        self.num_clusters.iter().copied().max().unwrap_or(0)
    }

    pub fn unify_segment(&self, temp: &Segment) -> Segment {
        self.segments
            .get(&temp.cluster_ids)
            .cloned()
            .unwrap_or_else(|| temp.clone())
    }

    pub fn get_paired_segments(&self, unpaired_segment: &Segment) -> Vec<&Segment> {
        self.segments
            .values()
            .filter(|segment| {
                segment
                    .cluster_ids
                    .iter()
                    .zip(unpaired_segment.cluster_ids.iter())
                    .all(|(&seg, &unp)| unp == UNCLUSTERED || (seg == unp && seg != UNCLUSTERED))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_builds_pair_patterns() {
        let c1 = vec![0, 0, 1, 1];
        let c2 = vec![0, 1, 0, 1];
        let s = Segments::from_labelings(&[c1, c2]);

        assert_eq!(s.get_clusterings(), 2);
        assert_eq!(s.get_pair_count(true), 6);
        assert_eq!(s.size(), 5);
        assert_eq!(s.segments().map(|x| x.pair_count).sum::<u64>(), 6);
    }
}
