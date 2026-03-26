use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::api::RangeSearch;
use crate::cluster::dbscan::NOISE;
#[cfg(test)]
use crate::distance::Euclidean;
use crate::{DistanceData, Float, IndexQuery};

#[derive(Debug, Clone, Copy)]
struct ReachCandidate<F: Float> {
    reachability: F,
    index: usize,
}

impl<F: Float> PartialEq for ReachCandidate<F> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.reachability == other.reachability
    }
}

impl<F: Float> Eq for ReachCandidate<F> {}

impl<F: Float> PartialOrd for ReachCandidate<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl<F: Float> Ord for ReachCandidate<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .reachability
            .partial_cmp(&self.reachability)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.index.cmp(&self.index))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpticsResult<F: Float> {
    pub ordering: Vec<usize>,
    pub reachability: Vec<F>,
    pub core_distance: Vec<Option<F>>,
    pub predecessor: Vec<Option<usize>>,
    pub labels: Vec<isize>,
}

/// Run OPTICS clustering with VP-tree range-search neighborhoods.
///
/// The extraction step uses the same `eps` threshold (DBSCAN-style extraction):
/// points in reachability valleys below `eps` are assigned to clusters,
/// other points become noise (`-1`).
///
/// # Panics
///
/// Panics if `eps < 0.0` or if `min_points == 0`.
pub fn optics<'a, S, D, F>(tree: &S, data: &'a D, eps: F, min_points: usize) -> OpticsResult<F>
where
    F: Float,
    D: DistanceData<F> + 'a,
    S: RangeSearch<F, D::Query<'a>>,
{
    assert!(eps >= F::zero(), "eps must be non-negative");
    assert!(min_points > 0, "min_points must be greater than 0");

    let size = data.size();
    let mut processed = vec![false; size];
    let mut reachability = vec![F::infinity(); size];
    let mut core_distance = vec![None; size];
    let mut predecessor = vec![None; size];
    let mut ordering = Vec::with_capacity(size);
    let mut seeds = BinaryHeap::new();

    let mut query = data.query();
    for point_idx in 0..size {
        if processed[point_idx] {
            continue;
        }

        let mut neighbors = region_query(tree, &mut query, point_idx, eps);
        processed[point_idx] = true;
        ordering.push(point_idx);

        core_distance[point_idx] = compute_core_distance(&mut neighbors, min_points);

        if let Some(point_core_distance) = core_distance[point_idx] {
            seeds.clear();
            update(
                &neighbors,
                point_idx,
                point_core_distance,
                &processed,
                &mut reachability,
                &mut predecessor,
                &mut seeds,
            );

            while let Some(candidate) = seeds.pop() {
                let current_idx = candidate.index;
                if processed[current_idx] {
                    continue;
                }
                if candidate.reachability > reachability[current_idx] {
                    continue;
                }

                let mut current_neighbors = region_query(tree, &mut query, current_idx, eps);
                processed[current_idx] = true;
                ordering.push(current_idx);

                core_distance[current_idx] =
                    compute_core_distance(&mut current_neighbors, min_points);

                if let Some(current_core_distance) = core_distance[current_idx] {
                    update::<F>(
                        &current_neighbors,
                        current_idx,
                        current_core_distance,
                        &processed,
                        &mut reachability,
                        &mut predecessor,
                        &mut seeds,
                    );
                }
            }
        }
    }

    let labels = extract_dbscan_labels(&ordering, &reachability, &core_distance, eps);

    OpticsResult { ordering, reachability, core_distance, predecessor, labels }
}

/// Extract Xi-based cluster labels from an OPTICS run result.
///
/// This follows the original OPTICS Xi steep area extraction and applies the
/// predecessor correction to reduce common Xi artifacts.
///
/// Returns flat labels (`-1` for noise), where nested clusters keep ownership
/// of points over parent clusters.
///
/// # Panics
///
/// Panics if `xi <= 0.0`, `xi >= 1.0`, or if `min_points == 0`.
#[must_use]
pub fn extract_xi_labels<F: Float>(
    result: &OpticsResult<F>, xi: F, min_points: usize,
) -> Vec<isize> {
    assert!(xi > F::zero() && xi < F::one(), "xi must be in (0, 1)");
    assert!(min_points > 0, "min_points must be greater than 0");

    let ordering = &result.ordering;
    let size = ordering.len();
    if size == 0 {
        return Vec::new();
    }

    let ixi = F::one() - xi;

    let mut reachability_by_order = Vec::with_capacity(size);
    for &point_idx in ordering {
        reachability_by_order.push(result.reachability[point_idx]);
    }

    let mut point_to_pos = vec![usize::MAX; size];
    for (pos, &point_idx) in ordering.iter().enumerate() {
        point_to_pos[point_idx] = pos;
    }

    let mut mib: F = F::zero();
    let mut sdaset: Vec<SteepDownArea<F>> = Vec::new();
    let mut cluster_intervals: Vec<(usize, usize)> = Vec::new();

    let mut scan = 0usize;
    while scan < size {
        mib = mib.max(reachability_by_order[scan]);

        if steep_down(&reachability_by_order, scan, ixi) {
            update_filter_sdaset(mib, &mut sdaset, ixi);

            let startval = reachability_by_order[scan];
            mib = F::zero();
            let startsteep = scan;
            let mut endsteep = scan;

            scan += 1;
            while scan < size {
                if steep_down(&reachability_by_order, scan, ixi) {
                    endsteep = scan;
                } else if !steep_down(&reachability_by_order, scan, F::one())
                    || scan - endsteep > min_points
                {
                    break;
                }
                scan += 1;
            }

            sdaset.push(SteepDownArea { start: startsteep, maximum: startval, mib: F::zero() });
            continue;
        }

        if steep_up(&reachability_by_order, scan, ixi) {
            update_filter_sdaset(mib, &mut sdaset, ixi);

            let mut endsteep = scan;
            mib = reachability_by_order[scan];
            let mut esuccr = next_reachability(&reachability_by_order, scan);

            while esuccr.is_finite() && scan < size {
                scan += 1;
                if scan >= size {
                    break;
                }

                if steep_up(&reachability_by_order, scan, ixi) {
                    endsteep = scan;
                    mib = reachability_by_order[scan];
                    esuccr = next_reachability(&reachability_by_order, scan);
                } else if !steep_up(&reachability_by_order, scan, F::one())
                    || scan - endsteep > min_points
                {
                    break;
                }
            }

            if esuccr.is_infinite() && scan < size {
                scan += 1;
            }

            for sda in sdaset.iter().rev() {
                let mut cstart = sda.start;
                let mut cend = endsteep;

                cend = predecessor_filter(
                    result,
                    ordering,
                    &reachability_by_order,
                    &point_to_pos,
                    cstart,
                    cend,
                );

                let e_u =
                    if cend + 1 < size { reachability_by_order[cend + 1] } else { F::infinity() };

                if sda.mib > sda.maximum.min(e_u) * ixi {
                    continue;
                }

                if sda.maximum * ixi >= e_u {
                    while cstart < cend && reachability_by_order[cstart + 1] * ixi > e_u {
                        cstart += 1;
                    }
                } else if e_u * ixi >= sda.maximum {
                    while cend > cstart && reachability_by_order[cend] * ixi > sda.maximum {
                        cend -= 1;
                    }
                }

                cend = predecessor_filter(
                    result,
                    ordering,
                    &reachability_by_order,
                    &point_to_pos,
                    cstart,
                    cend,
                );

                if cend + 1 - cstart < min_points {
                    continue;
                }

                cluster_intervals.push((cstart, cend));
            }
            continue;
        }

        scan += 1;
    }

    cluster_intervals.sort_by_key(|(start, end)| (end - start, *start));

    let mut labels = vec![NOISE; size];
    let mut claimed = vec![false; size];
    let mut cluster_id: isize = 0;

    for (cstart, cend) in cluster_intervals {
        let mut assigned_any = false;
        for &point_idx in ordering.iter().take(cend + 1).skip(cstart) {
            if !claimed[point_idx] {
                claimed[point_idx] = true;
                labels[point_idx] = cluster_id;
                assigned_any = true;
            }
        }
        if assigned_any {
            cluster_id += 1;
        }
    }

    labels
}

fn region_query<S, Q, F>(tree: &S, query: &mut Q, point_idx: usize, eps: F) -> Vec<(usize, F)>
where
    F: Float,
    Q: IndexQuery<F>,
    S: RangeSearch<F, Q>,
{
    query.set_index(point_idx);
    tree.search_range(query, eps).into_iter().map(|pair| (pair.index, pair.distance)).collect()
}

fn compute_core_distance<F: Float>(neighbors: &mut [(usize, F)], min_points: usize) -> Option<F> {
    if neighbors.len() < min_points {
        return None;
    }
    let rank = min_points - 1;
    let (_, candidate, _) = neighbors
        .select_nth_unstable_by(rank, |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    Some(candidate.1)
}

fn update<F: Float>(
    neighbors: &[(usize, F)], point_idx: usize, core_distance: F, processed: &[bool],
    reachability: &mut [F], predecessor: &mut [Option<usize>],
    seeds: &mut BinaryHeap<ReachCandidate<F>>,
) {
    for (neighbor_idx, distance) in neighbors {
        if *neighbor_idx == point_idx || processed[*neighbor_idx] {
            continue;
        }

        let new_reachability = core_distance.max(*distance);
        if new_reachability < reachability[*neighbor_idx] {
            reachability[*neighbor_idx] = new_reachability;
            predecessor[*neighbor_idx] = Some(point_idx);
            seeds.push(ReachCandidate { reachability: new_reachability, index: *neighbor_idx });
        }
    }
}

fn extract_dbscan_labels<F: Float>(
    ordering: &[usize], reachability: &[F], core_distance: &[Option<F>], eps: F,
) -> Vec<isize> {
    let mut labels = vec![NOISE; reachability.len()];
    let mut cluster_id: isize = -1;

    for &point_idx in ordering {
        let reach = reachability[point_idx];
        let core = core_distance[point_idx].unwrap_or(F::infinity());

        labels[point_idx] = if reach > eps {
            if core <= eps {
                cluster_id += 1;
                cluster_id
            } else {
                NOISE
            }
        } else if cluster_id >= 0 {
            cluster_id
        } else {
            NOISE
        };
    }

    labels
}

#[derive(Debug, Clone, Copy)]
struct SteepDownArea<F: Float> {
    start: usize,
    maximum: F,
    mib: F,
}

fn update_filter_sdaset<F: Float>(mib: F, sdaset: &mut Vec<SteepDownArea<F>>, ixi: F) {
    sdaset.retain_mut(|sda| {
        if sda.maximum * ixi <= mib {
            false
        } else {
            if mib > sda.mib {
                sda.mib = mib;
            }
            true
        }
    });
}

fn steep_up<F: Float>(reachability_by_order: &[F], idx: usize, ixi: F) -> bool {
    let current = reachability_by_order[idx];
    current.is_finite()
        && (idx + 1 >= reachability_by_order.len()
            || current <= reachability_by_order[idx + 1] * ixi)
}

fn steep_down<F: Float>(reachability_by_order: &[F], idx: usize, ixi: F) -> bool {
    if idx + 1 >= reachability_by_order.len() {
        return false;
    }
    let next = reachability_by_order[idx + 1];
    next.is_finite() && next <= reachability_by_order[idx] * ixi
}

fn next_reachability<F: Float>(reachability_by_order: &[F], idx: usize) -> F {
    if idx + 1 < reachability_by_order.len() {
        reachability_by_order[idx + 1]
    } else {
        F::infinity()
    }
}

fn predecessor_filter<F: Float>(
    result: &OpticsResult<F>, ordering: &[usize], reachability_by_order: &[F],
    point_to_pos: &[usize], cstart: usize, mut cend: usize,
) -> usize {
    if cend >= ordering.len() {
        return ordering.len().saturating_sub(1);
    }

    let startval = reachability_by_order[cstart];
    while cend > cstart {
        let point_idx = ordering[cend];

        if reachability_by_order[cend] < startval {
            break;
        }

        if let Some(pred_idx) = result.predecessor[point_idx]
            && let Some(pred_pos) = point_to_pos.get(pred_idx).copied()
            && pred_pos >= cstart
            && pred_pos < cend
        {
            break;
        }

        cend -= 1;
    }
    cend
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::vptree::VPTree;

    #[test]
    fn optics_finds_two_clusters_and_noise() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![5.0, 5.0],
        ];

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(7);
        let tree = VPTree::new(&data, 2, &mut rng);

        let result = optics(&tree, &data, 0.25, 3);
        let labels = result.labels;

        assert_eq!(labels.len(), points.len());
        assert_eq!(labels[6], NOISE);

        let first_cluster = labels[0];
        let second_cluster = labels[3];
        assert!(first_cluster >= 0);
        assert!(second_cluster >= 0);
        assert_ne!(first_cluster, second_cluster);

        assert_eq!(labels[1], first_cluster);
        assert_eq!(labels[2], first_cluster);
        assert_eq!(labels[4], second_cluster);
        assert_eq!(labels[5], second_cluster);

        let clusters: HashSet<isize> = labels.iter().copied().filter(|&label| label >= 0).collect();
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn optics_returns_full_ordering() {
        let points = vec![vec![0.0, 0.0], vec![0.2, 0.0], vec![0.4, 0.0], vec![0.6, 0.0]];

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(99);
        let tree = VPTree::new(&data, 2, &mut rng);

        let result = optics(&tree, &data, 0.25, 2);

        assert_eq!(result.ordering.len(), points.len());

        let set: HashSet<usize> = result.ordering.iter().copied().collect();
        assert_eq!(set.len(), points.len());

        assert_eq!(result.reachability.len(), points.len());
        assert_eq!(result.core_distance.len(), points.len());
        assert_eq!(result.predecessor.len(), points.len());
    }

    #[test]
    fn optics_xi_extracts_two_valleys_from_reachability_plot() {
        let n = 16usize;
        let ordering: Vec<usize> = (0..n).collect();
        let reachability = vec![
            f64::INFINITY,
            10.0,
            8.0,
            6.0,
            4.0,
            6.0,
            8.0,
            10.0,
            f64::INFINITY,
            9.0,
            7.0,
            5.0,
            3.0,
            5.0,
            7.0,
            9.0,
        ];
        let predecessor = (0..n).map(|idx| if idx == 0 { None } else { Some(idx - 1) }).collect();

        let result = OpticsResult {
            ordering,
            reachability,
            core_distance: vec![None; n],
            predecessor,
            labels: vec![NOISE; n],
        };

        let labels = extract_xi_labels(&result, 0.2, 3);

        assert!(labels[4] >= 0);
        assert!(labels[12] >= 0);
        assert_ne!(labels[4], labels[12]);

        let clusters: HashSet<isize> = labels.iter().copied().filter(|&label| label >= 0).collect();
        assert!(clusters.len() >= 2);
    }

    #[test]
    fn optics_matches_sklearn_processing_order_reference() {
        let points = vec![vec![0.0], vec![10.0], vec![-10.0], vec![25.0]];

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(1234);
        let tree = VPTree::new(&data, 1, &mut rng);

        let result = optics(&tree, &data, 15.0, 3);

        assert_eq!(result.ordering, vec![0, 1, 2, 3]);

        let expected_reachability = [f64::INFINITY, 10.0, 10.0, 15.0];
        for (actual, expected) in result.reachability.iter().zip(expected_reachability.iter()) {
            let actual: f64 = *actual;
            let expected: f64 = *expected;
            if expected.is_infinite() {
                assert!(actual.is_infinite());
            } else {
                assert!((actual - expected).abs() < 1e-12);
            }
        }

        let expected_core = [Some(10.0), Some(15.0), None, None];
        for (actual, expected) in result.core_distance.iter().zip(expected_core.iter()) {
            match (actual, expected) {
                (Some(a), Some(e)) => assert!((a - e).abs() < 1e-12),
                (None, None) => {}
                _ => panic!("core-distance mismatch"),
            }
        }
    }

    #[test]
    #[allow(clippy::excessive_precision, clippy::too_many_lines, clippy::unreadable_literal)]
    fn optics_matches_elki_reference_ordering_and_reachability() {
        let points = vec![
            vec![-3.588758123225869, -1.6798742333062213],
            vec![-4.2170096127154082, -0.20728544063883358],
            vec![-3.5059536078800262, -2.7818223039011287],
            vec![-4.2399292659795282, -2.1210857666381582],
            vec![-5.0825750814348467, -1.6715211984493021],
            vec![-4.8847651430712977, -0.83658119442961998],
            vec![-4.3911698198824052, -1.9026599868057372],
            vec![-4.6449094138036591, -1.7330605381005866],
            vec![-3.8047367414739153, -2.1641266110126409],
            vec![-4.7495458386792793, -2.6832765914413796],
            vec![3.7447010184165923, -0.93463814045596394],
            vec![4.0864436198859506, -1.0742165020406442],
            vec![4.2269754623987605, -1.1454365674598765],
            vec![4.0045758517301442, -1.0187183850025834],
            vec![4.153277921435846, -0.85306412300997148],
            vec![4.0154947425696914, -0.96218374803978268],
            vec![3.9112214252369886, -1.1980796468223927],
            vec![3.9652087850673849, -0.98436510308960201],
            vec![4.1230290680727721, -0.87976201512155883],
            vec![3.9612673182592046, -1.0302302750575336],
            vec![0.79028940698658146, -2.2840035874357949],
            vec![0.65874596187499745, -1.6098449209536421],
            vec![0.89806956364966928, -2.0876148603222373],
            vec![0.74944092799001472, -1.844501928833618],
            vec![0.67722043048840974, -2.0425480560427935],
            vec![0.82090668776126485, -1.9226195004281477],
            vec![0.89783897248622535, -2.2361264368244824],
            vec![0.99436355433226897, -1.9143336258939165],
            vec![1.0133034444766336, -1.9395056204520438],
            vec![0.8731355812638073, -2.0725482331974274],
            vec![-2.2017381343327855, 2.8921340515378375],
            vec![-2.2439438846133362, 2.4821152193004972],
            vec![-1.9467721573238741, 2.8794657191375213],
            vec![-2.4890595040898136, 3.1388346766577322],
            vec![-2.2721895093149724, 3.0155836187388418],
            vec![-1.781272831346739, 3.038694873227223],
            vec![-1.6581797946370098, 2.6295522538939045],
            vec![-1.8792975076467353, 2.7945569727179063],
            vec![-2.2612391447545646, 2.8263451005706752],
            vec![-2.093465759638212, 3.0168496026689238],
            vec![1.1357602547466294, -0.55867762087330064],
            vec![3.7450599035687357, -4.4579898980435582],
            vec![5.3812035100729592, 1.0334226816489331],
            vec![4.8860473138554408, -2.2878797372997615],
            vec![1.2867958055831319, -0.31287723691018132],
            vec![2.3549168848429125, -0.044087887388116087],
            vec![3.3332399649229765, -0.43737754162605946],
            vec![3.570186235479043, -0.86948293089288287],
            vec![3.0168000331533129, 0.85739279024933657],
            vec![3.2030593483257919, -1.3568170184884774],
            vec![8.7663013941125083, 3.3044818777151073],
            vec![2.4590300030285328, 7.9387934163160221],
            vec![2.6537531897716802, 9.8872423712985853],
            vec![4.1727620384805055, 4.5050903771184849],
            vec![8.8458840529607698, 8.9610295828688482],
            vec![8.7351179208531402, 7.8120893165507708],
            vec![3.2775486298905951, 9.8201299061980674],
            vec![4.463993258097239, 7.6049127915927901],
            vec![6.8945039355474957, 5.689979813818332],
            vec![6.2281587406921606, 7.8444133431330538],
        ];

        let expected_ordering = vec![
            0, 3, 6, 4, 7, 8, 2, 9, 5, 1, 31, 30, 32, 34, 33, 38, 39, 35, 37, 36, 44, 21, 23, 24,
            22, 25, 27, 29, 26, 28, 20, 40, 45, 46, 10, 15, 11, 13, 17, 19, 18, 12, 16, 14, 47, 49,
            43, 48, 42, 41, 53, 57, 51, 52, 56, 59, 54, 55, 58, 50,
        ];

        let expected_reachability_by_order = vec![
            f64::INFINITY,
            1.0574896366427478,
            0.7587934993548423,
            0.7290174038973836,
            0.7290174038973836,
            0.7290174038973836,
            0.6861627576116127,
            0.7587934993548423,
            0.9280118450166668,
            1.1748022534146194,
            3.3355455741292257,
            0.49618389254482587,
            0.2552805046961355,
            0.2552805046961355,
            0.24944622248445714,
            0.24944622248445714,
            0.24944622248445714,
            0.2552805046961355,
            0.2552805046961355,
            0.3086779122185853,
            4.163024452756142,
            1.623152630340929,
            0.45315840475822655,
            0.25468325192031926,
            0.2254004358159971,
            0.18765711877083036,
            0.1821471333893275,
            0.1821471333893275,
            0.18765711877083036,
            0.18765711877083036,
            0.2240202988740153,
            1.154337614548715,
            1.342604473837069,
            1.323308536402633,
            0.8607514948648837,
            0.27219111215810565,
            0.13260875220533205,
            0.13260875220533205,
            0.09890587675958984,
            0.09890587675958984,
            0.13548790801634494,
            0.1575483940837384,
            0.17515137170530226,
            0.17575920159442388,
            0.27219111215810565,
            0.6101447895405373,
            1.3189208094864302,
            1.323308536402633,
            2.2509184159764577,
            2.4517810628594527,
            3.675977064404973,
            3.8264795626020365,
            2.9130735341510614,
            2.9130735341510614,
            2.9130735341510614,
            2.9130735341510614,
            2.8459300127258036,
            2.8459300127258036,
            2.8459300127258036,
            3.0321982337972537,
        ];

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(0);
        let tree = VPTree::new(&data, 4, &mut rng);

        let result = optics(&tree, &data, f64::INFINITY, 5);

        assert_eq!(result.ordering, expected_ordering);

        let actual_reachability_by_order: Vec<f64> =
            result.ordering.iter().map(|&idx| result.reachability[idx]).collect();

        for (actual, expected) in
            actual_reachability_by_order.iter().zip(expected_reachability_by_order.iter())
        {
            if expected.is_infinite() {
                assert!(actual.is_infinite());
            } else {
                assert!((actual - expected).abs() < 1e-9);
            }
        }
    }
}
