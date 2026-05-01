use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fuel::api::float::VecOps;
use fuel::math;
use ndarray::{Array2, ShapeBuilder};

const PAIRWISE_CONFIG: [(usize, usize, usize); 3] = [(64, 64, 16), (128, 128, 32), (256, 64, 32)];

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(1000);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

fn bench_pairwise_sqdist_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::pairwise_sqdist_vs_naive");
    configure_group(&mut group);

    for &(k, n, d) in &PAIRWISE_CONFIG {
        let centers: Array2<f64> = Array2::from_shape_vec(
            (k, d),
            (0..k).flat_map(|j| (0..d).map(move |i| (j * d + i) as f64 * 0.001)).collect(),
        )
        .unwrap();
        let points: Array2<f64> = Array2::from_shape_vec(
            (n, d),
            (0..n).flat_map(|j| (0..d).map(move |i| (j * d + i) as f64 * 0.002)).collect(),
        )
        .unwrap();
        let center_refs: Vec<Vec<f64>> =
            centers.rows().into_iter().map(|row| row.as_slice().unwrap().to_vec()).collect();
        let point_refs: Vec<Vec<f64>> =
            points.rows().into_iter().map(|row| row.as_slice().unwrap().to_vec()).collect();

        group.bench_with_input(
            format!("pairwise k={k} n={n} d={d}"),
            &(k, n, d),
            |b, &(_, _, _)| {
                b.iter(|| math::pairwise_sqdist(black_box(&centers), black_box(&points)))
            },
        );

        group.bench_with_input(format!("naive k={k} n={n} d={d}"), &(k, n, d), |b, &(_, _, d)| {
            b.iter(|| {
                let mut matrix = vec![0_f64; k * n];
                for j in 0..k {
                    let center = &center_refs[j];
                    for i in 0..n {
                        let point = &point_refs[i];
                        matrix[j * n + i] =
                            math::sqdist(black_box(center), black_box(point), black_box(d));
                    }
                }
                black_box(matrix)
            })
        });

        group.bench_with_input(format!("row k={k} n={n} d={d}"), &(k, n, d), |b, &(_, _, d)| {
            let points_view = points.view();
            let mut row_buf = vec![0_f64; n];
            b.iter(|| {
                let mut matrix = vec![0_f64; k * n];
                for j in 0..k {
                    let row = centers.row(j);
                    let center = row.as_slice().unwrap();
                    <f64 as VecOps>::vec_row_sqdist(center, points_view, d, &mut row_buf, n);
                    matrix[j * n..(j + 1) * n].copy_from_slice(&row_buf);
                }
                black_box(matrix)
            })
        });

        // Fortran-order (column-major) variant: no packing, NR-way SIMD direct from columns.
        let points_f =
            Array2::from_shape_vec((n, d).f(), points.iter().copied().collect::<Vec<_>>()).unwrap();
        group.bench_with_input(
            format!("row-fortran k={k} n={n} d={d}"),
            &(k, n, d),
            |b, &(_, _, d)| {
                let points_view_f = points_f.view();
                let mut row_buf = vec![0_f64; n];
                b.iter(|| {
                    let mut matrix = vec![0_f64; k * n];
                    for j in 0..k {
                        let row = centers.row(j);
                        let center = row.as_slice().unwrap();
                        <f64 as VecOps>::vec_row_sqdist(center, points_view_f, d, &mut row_buf, n);
                        matrix[j * n..(j + 1) * n].copy_from_slice(&row_buf);
                    }
                    black_box(matrix)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(pairwise_benches, bench_pairwise_sqdist_vs_naive);
criterion_main!(pairwise_benches);
