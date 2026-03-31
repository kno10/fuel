use rand::{Rng, SeedableRng};

use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float, VectorData};

fn to_f64_point<F: Float>(point: &[F]) -> Vec<f64> {
    point.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect()
}

#[derive(Debug)]
struct Node {
    code: u64,
    center: Vec<f64>,
    count: usize,
    level: isize,
    children: Option<Vec<Node>>,
}

impl Node {
    fn get_square_sum(&self, levels: isize) -> u64 {
        let mut sum = 0u64;
        let mut stack: Vec<(&Node, isize)> = Vec::new();
        stack.push((self, levels));
        while let Some((node, lvl)) = stack.pop() {
            if lvl <= 0 || node.children.is_none() {
                let cnt = node.count as u64;
                sum += cnt * cnt;
            } else if let Some(children) = &node.children {
                for child in children {
                    stack.push((child, lvl - 1));
                }
            }
        }
        sum
    }

    fn get_cubic_sum(&self, levels: isize) -> u64 {
        let mut sum = 0u64;
        let mut stack: Vec<(&Node, isize)> = Vec::new();
        stack.push((self, levels));
        while let Some((node, lvl)) = stack.pop() {
            if lvl <= 0 || node.children.is_none() {
                let cnt = node.count as u64;
                sum += cnt * cnt * cnt;
            } else if let Some(children) = &node.children {
                for child in children {
                    stack.push((child, lvl - 1));
                }
            }
        }
        sum
    }
}

#[derive(Debug)]
struct ALOCIQuadTree {
    shift: Vec<f64>,
    min: Vec<f64>,
    width: Vec<f64>,
    root: Node,
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

impl ALOCIQuadTree {
    fn max_level(&self) -> isize {
        let mut max_level = self.root.level;
        let mut stack: Vec<&Node> = Vec::new();
        stack.push(&self.root);
        while let Some(node) = stack.pop() {
            max_level = max_level.max(node.level);
            if let Some(children) = &node.children {
                for child in children {
                    stack.push(child);
                }
            }
        }
        max_level
    }

    fn get_shifted_dim(&self, point: &[f64], dim: usize, level: isize) -> f64 {
        let mut pos = point[dim] + self.shift[dim];
        pos = (pos - self.min[dim]) / self.width[dim] * (1.0 + level as f64);
        pos - pos.floor()
    }

    fn build_tree(
        min: &[f64], max: &[f64], shift: &[f64], nmin: usize, points: &[Vec<f64>],
    ) -> Node {
        let dims = min.len();
        let width: Vec<f64> = max
            .iter()
            .zip(min.iter())
            .map(|(mx, mn)| {
                let w = mx - mn;
                if w <= 0.0 { 1.0 } else { w }
            })
            .collect();

        let mut center = vec![0.0; dims];
        for i in 0..dims {
            center[i] =
                min[i] + shift[i] + width[i] * if shift[i] < width[i] * 0.5 { 0.5 } else { -0.5 };
        }

        let mut ids: Vec<usize> = (0..points.len()).collect();
        let ids_len = ids.len();
        let children = ALOCIQuadTree::bulk_load(
            points,
            min.to_vec(),
            max.to_vec(),
            &mut ids,
            0,
            ids_len,
            0,
            0,
            0,
            nmin,
            shift.to_vec(),
            min.to_vec(),
            width.clone(),
        );

        Node { code: 0, center, count: points.len(), level: -1, children: Some(children) }
    }

    #[allow(clippy::too_many_arguments)]
    fn bulk_load(
        points: &[Vec<f64>], mut lmin: Vec<f64>, mut lmax: Vec<f64>, ids: &mut [usize],
        start: usize, end: usize, dim: usize, level: isize, code: u64, nmin: usize,
        global_shift: Vec<f64>, global_min: Vec<f64>, global_width: Vec<f64>,
    ) -> Vec<Node> {
        let dims = lmin.len();
        let mut children: Vec<Node> = Vec::new();

        if dim == 0 && start < end {
            let first = &points[ids[start]];
            let mut degenerate = true;
            'degenerate_check: for idx in (start + 1)..end {
                let point = &points[ids[idx]];
                for d in 0..dims {
                    if (first[d] - point[d]).abs() > 1e-15 {
                        degenerate = false;
                        break 'degenerate_check;
                    }
                }
            }
            if degenerate {
                let mut center = vec![0.0; dims];
                for d in 0..dims {
                    center[d] = lmin[d] * 0.5 + lmax[d] * 0.5 + global_shift[d];
                    if center[d] > global_min[d] + global_width[d] {
                        center[d] -= global_width[d];
                    }
                }
                children.push(Node { code, center, count: end - start, level, children: None });
                return children;
            }
        }

        if dim == dims {
            let mut center = vec![0.0; dims];
            for d in 0..dims {
                center[d] = lmin[d] * 0.5 + lmax[d] * 0.5 + global_shift[d];
                if center[d] > global_min[d] + global_width[d] {
                    center[d] -= global_width[d];
                }
            }
            if end - start <= nmin {
                children.push(Node { code, center, count: end - start, level, children: None });
                return children;
            }
            let subchildren = ALOCIQuadTree::bulk_load(
                points,
                lmin.clone(),
                lmax.clone(),
                ids,
                start,
                end,
                0,
                level + 1,
                0,
                nmin,
                global_shift.clone(),
                global_min.clone(),
                global_width.clone(),
            );
            let children_opt = if subchildren.is_empty() { None } else { Some(subchildren) };
            children.push(Node { code, center, count: end - start, level, children: children_opt });
            return children;
        }

        // Two-pointer partition: hi is inclusive upper bound (starts at end-1).
        if start >= end {
            return children;
        }
        let mut lo = start;
        let mut hi = end - 1; // inclusive upper bound
        while lo < hi {
            let point_lo = &points[ids[lo]];
            if ALOCIQuadTree::calculate_shifted_dim_for_build(
                &global_shift,
                &global_min,
                &global_width,
                point_lo,
                dim,
                level,
            ) <= 0.5
            {
                lo += 1;
                continue;
            }
            let point_hi = &points[ids[hi]];
            if ALOCIQuadTree::calculate_shifted_dim_for_build(
                &global_shift,
                &global_min,
                &global_width,
                point_hi,
                dim,
                level,
            ) > 0.5
            {
                if hi == 0 {
                    break;
                }
                hi -= 1;
                continue;
            }
            ids.swap(lo, hi);
            lo += 1;
            if hi == 0 {
                break;
            }
            hi -= 1;
        }

        let spos = lo;
        if start < spos {
            let tmp = lmax[dim];
            lmax[dim] = lmax[dim] * 0.5 + lmin[dim] * 0.5;
            let left_children = ALOCIQuadTree::bulk_load(
                points,
                lmin.clone(),
                lmax.clone(),
                ids,
                start,
                spos,
                dim + 1,
                level,
                code,
                nmin,
                global_shift.clone(),
                global_min.clone(),
                global_width.clone(),
            );
            children.extend(left_children);
            lmax[dim] = tmp;
        }
        if spos < end {
            let tmp = lmin[dim];
            lmin[dim] = lmax[dim] * 0.5 + lmin[dim] * 0.5;
            let right_children = ALOCIQuadTree::bulk_load(
                points,
                lmin.clone(),
                lmax.clone(),
                ids,
                spos,
                end,
                dim + 1,
                level,
                code | (1 << dim),
                nmin,
                global_shift.clone(),
                global_min.clone(),
                global_width.clone(),
            );
            children.extend(right_children);
            lmin[dim] = tmp;
        }

        children
    }

    fn calculate_shifted_dim_for_build(
        shift: &[f64], min: &[f64], width: &[f64], point: &[f64], dim: usize, level: isize,
    ) -> f64 {
        let mut pos = point[dim] + shift[dim];
        pos = (pos - min[dim]) / width[dim] * (1.0 + level as f64);
        pos - pos.floor()
    }

    fn find_closest_node<'a>(&'a self, point: &[f64], tlevel: isize) -> &'a Node {
        let mut cur = &self.root;
        if tlevel < 0 {
            return cur;
        }

        for level in 0..=tlevel {
            let children = match &cur.children {
                Some(ch) => ch,
                None => break,
            };
            let mut code = 0;
            for d in 0..self.min.len() {
                if self.get_shifted_dim(point, d, level) > 0.5 {
                    code |= 1 << d;
                }
            }
            let mut found = false;
            for child in children.iter() {
                if child.code == code {
                    cur = child;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
        cur
    }
}

fn calculate_mdef_norm(sn: &Node, cg: &Node) -> f64 {
    let sq = sn.get_square_sum(cg.level - sn.level);
    if sq == sn.count as u64 {
        return 0.0;
    }
    let cb = sn.get_cubic_sum(cg.level - sn.level);
    let n_hat = (sq as f64) / (sn.count as f64);
    let cb_i = cb as f64;
    let sq_i = sq as f64;
    let cnt_i = sn.count as f64;
    let sig_n_hat_num = cb_i * cnt_i - sq_i * sq_i;
    let sig_n_hat = sig_n_hat_num.sqrt() / (sn.count as f64);
    if sig_n_hat < f64::MIN_POSITIVE {
        return 0.0;
    }
    let mdef = n_hat - (cg.count as f64);
    mdef / sig_n_hat
}

/// aLOCI implementation ported from ELKI ALOCI.java
pub fn approximate_local_correlation_integral<'a, D, F>(
    data: &'a D, nmin: usize, alpha: usize, g: usize, seed: u64,
) -> OutlierResult<F>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + VectorData<F> + Sync + 'a,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "aLOCI",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let dims = data.dims();
    assert!(dims <= 32, "aLOCI quadtree supports up to 32 dimensions");
    let points: Vec<Vec<f64>> = (0..size).map(|i| to_f64_point(data.point(i))).collect();

    let mut min = vec![f64::INFINITY; dims];
    let mut max = vec![f64::NEG_INFINITY; dims];
    for point in &points {
        for d in 0..dims {
            min[d] = min[d].min(point[d]);
            max[d] = max[d].max(point[d]);
        }
    }

    let mut max_diff = 0.0f64;
    for d in 0..dims {
        max_diff = max_diff.max(max[d] - min[d]);
    }
    for d in 0..dims {
        let diff = (max_diff - (max[d] - min[d])) * 0.5;
        min[d] -= diff;
        max[d] += diff;
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut qts: Vec<ALOCIQuadTree> = Vec::with_capacity(g);

    for grid in 0..g {
        let shift: Vec<f64> = if grid == 0 {
            vec![0.0; dims]
        } else {
            (0..dims).map(|d| rng.gen_range(0.0..(max[d] - min[d]))).collect()
        };

        let root = ALOCIQuadTree::build_tree(&min, &max, &shift, nmin, &points);
        let qt = ALOCIQuadTree {
            shift: shift.clone(),
            min: min.clone(),
            width: max
                .iter()
                .zip(min.iter())
                .map(|(mx, mn)| {
                    let w = mx - mn;
                    if w <= 0.0 { 1.0 } else { w }
                })
                .collect(),
            root,
        };
        qts.push(qt);
    }

    let alpha_level = alpha as isize;
    let max_level = qts.iter().map(|qt| qt.max_level()).max().unwrap_or(-1);

    let mut outlier_scores = vec![F::zero(); size];
    for i in 0..size {
        let point = &points[i];
        let mut max_mdef_norm = 0.0;

        for l in 0..=max_level {
            let mut ci: Option<&Node> = None;
            for qt in &qts {
                let c = qt.find_closest_node(point, l);
                if c.level != l {
                    continue;
                }
                if ci.is_none()
                    || euclidean_distance(&ci.unwrap().center, point)
                        > euclidean_distance(&c.center, point)
                {
                    ci = Some(c);
                }
            }
            let ci = match ci {
                Some(c) => c,
                None => break,
            };

            let l_alpha = l - alpha_level;
            let mut cj: Option<&Node> = None;
            for qt in &qts {
                let c = qt.find_closest_node(&ci.center, l_alpha);
                if let Some(prev) = cj
                    && c.level < prev.level
                {
                    continue;
                }
                if cj.is_none()
                    || euclidean_distance(&c.center, &ci.center)
                        < euclidean_distance(&cj.unwrap().center, &ci.center)
                {
                    cj = Some(c);
                }
            }

            let cj = match cj {
                Some(c) => c,
                None => continue,
            };
            let mdef_norm = calculate_mdef_norm(cj, ci);
            if mdef_norm > max_mdef_norm {
                max_mdef_norm = mdef_norm;
            }
        }

        outlier_scores[i] = if max_mdef_norm.is_infinite() {
            F::infinity()
        } else {
            F::from_f64(max_mdef_norm).unwrap_or(F::zero())
        };
    }

    make_outlier_result(outlier_scores, "aLOCI", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::outlier::common::*;
    use crate::search::vptree::VPTree;

    #[test]
    fn aloci_remote_outlier_lowest() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![5.0, 5.0],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let _tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let results = approximate_local_correlation_integral(&data, 2, 4, 1, 42);
        assert!(results.scores.iter().all(|v| !v.is_nan() && *v >= 0.0));

        let outlier_idx = points.len() - 1;
        let min_score =
            results.scores.iter().cloned().filter(|v| v.is_finite()).fold(f64::INFINITY, f64::min);
        assert!(
            results.scores[outlier_idx].is_infinite()
                || results.scores[outlier_idx] <= min_score + 1e-12
        );
    }

    #[test]
    fn aloci_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let _tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let reference = load_reference_scores();
        let expected = reference.get("ALOCI-10").expect("No reference for ALOCI-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        let result = approximate_local_correlation_integral(&data, 10, 4, 1, 0);
        println!(
            "ALOCI-10 {}",
            result.scores.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
        );

        assert_outlier_auc_approx(
            "ALOCI-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("ALOCI-10", &result.scores, expected, 1e-6);
    }
}
