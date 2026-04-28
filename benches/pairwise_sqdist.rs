use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fuel::math;

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
        let centers: Vec<Vec<f64>> =
            (0..k).map(|j| (0..d).map(|i| (j * d + i) as f64 * 0.001).collect()).collect();
        let points: Vec<Vec<f64>> =
            (0..n).map(|j| (0..d).map(|i| (j * d + i) as f64 * 0.002).collect()).collect();

        let center_refs: Vec<&[f64]> = centers.iter().map(|row| row.as_slice()).collect();
        let point_refs: Vec<&[f64]> = points.iter().map(|row| row.as_slice()).collect();

        group.bench_with_input(
            format!("pairwise k={k} n={n} d={d}"),
            &(k, n, d),
            |b, &(_, _, d)| {
                b.iter(|| {
                    math::pairwise_sqdist(
                        black_box(&center_refs),
                        black_box(&point_refs),
                        black_box(d),
                    )
                })
            },
        );

        group.bench_with_input(format!("naive k={k} n={n} d={d}"), &(k, n, d), |b, &(_, _, d)| {
            b.iter(|| {
                let mut matrix = vec![0_f64; k * n];
                for j in 0..k {
                    let center = center_refs[j];
                    for i in 0..n {
                        matrix[j * n + i] =
                            math::sqdist(black_box(center), black_box(point_refs[i]), black_box(d));
                    }
                }
                black_box(matrix)
            })
        });
    }

    group.finish();
}

criterion_group!(pairwise_benches, bench_pairwise_sqdist_vs_naive);
criterion_main!(pairwise_benches);
