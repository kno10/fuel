use std::collections::HashMap;

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
