use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fuel::math;

const MATH_DIMS: [usize; 5] = [2, 3, 4, 5, 8];

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(10000);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
}

fn bench_math_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::dot_dispatch");
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v1 = vec![1.234567_f64; d];
        let v2 = vec![2.345678_f64; d];

        group.bench_with_input(format!("d={d}"), &d, |b, &d| {
            b.iter(|| math::dot(black_box(&v1), black_box(&v2), black_box(d)))
        });
    }
    group.finish();
}

fn bench_math_dot_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::dot_scalar");
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v1 = vec![1.234567_f64; d];
        let v2 = vec![2.345678_f64; d];

        group.bench_with_input(format!("d={d}"), &d, |b, &d| {
            b.iter(|| math::scalar::dot(black_box(&v1), black_box(&v2), black_box(d)))
        });
    }
    group.finish();
}

fn bench_math_dot_avx2(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        let mut group = c.benchmark_group("math::dot_avx2");
        configure_group(&mut group);
        for &d in &MATH_DIMS {
            let v1 = vec![1.234567_f64; d];
            let v2 = vec![2.345678_f64; d];

            group.bench_with_input(format!("d={d}"), &d, |b, &d| {
                b.iter(|| math::avx2::dot(black_box(&v1), black_box(&v2), black_box(d)))
            });
        }
        group.finish();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("avx2 benchmarks are skipped on non-x86_64");
    }
}

fn bench_math_sqdist(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::sqdist_dispatch");
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v1 = vec![1.234567_f64; d];
        let v2 = vec![2.345678_f64; d];

        group.bench_with_input(format!("d={d}"), &d, |b, &d| {
            b.iter(|| math::sqdist(black_box(&v1), black_box(&v2), black_box(d)))
        });
    }
    group.finish();
}

fn bench_math_sqdist_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::sqdist_scalar");
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v1 = vec![1.234567_f64; d];
        let v2 = vec![2.345678_f64; d];

        group.bench_with_input(format!("d={d}"), &d, |b, &d| {
            b.iter(|| math::scalar::sqdist(black_box(&v1), black_box(&v2), black_box(d)))
        });
    }
    group.finish();
}

fn bench_math_sqdist_avx2(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        let mut group = c.benchmark_group("math::sqdist_avx2");
        configure_group(&mut group);
        for &d in &MATH_DIMS {
            let v1 = vec![1.234567_f64; d];
            let v2 = vec![2.345678_f64; d];

            group.bench_with_input(format!("d={d}"), &d, |b, &d| {
                b.iter(|| math::avx2::sqdist(black_box(&v1), black_box(&v2), black_box(d)))
            });
        }
        group.finish();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("avx2 benchmarks are skipped on non-x86_64");
    }
}

fn bench_math_axpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::axpy_dispatch");
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v2 = vec![1.234567_f64; d];
        let v1_base = vec![2.0_f64; d];

        group.bench_with_input(format!("d={d}"), &d, |b, &d| {
            b.iter_batched(
                || v1_base.clone(),
                |mut v| {
                    math::axpy(&mut v, black_box(3.14_f64), black_box(&v2), black_box(d));
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_math_axpy_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::axpy_scalar");
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v2 = vec![1.234567_f64; d];
        let v1_base = vec![2.0_f64; d];

        group.bench_with_input(format!("d={d}"), &d, |b, &d| {
            b.iter_batched(
                || v1_base.clone(),
                |mut v| {
                    math::scalar::axpy(&mut v, black_box(3.14_f64), black_box(&v2), black_box(d));
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_math_axpy_avx2(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        let mut group = c.benchmark_group("math::axpy_avx2");
        configure_group(&mut group);
        for &d in &MATH_DIMS {
            let v2 = vec![1.234567_f64; d];
            let v1_base = vec![2.0_f64; d];

            group.bench_with_input(format!("d={d}"), &d, |b, &d| {
                b.iter_batched(
                    || v1_base.clone(),
                    |mut v| {
                        math::avx2::axpy(&mut v, black_box(3.14_f64), black_box(&v2), black_box(d));
                        black_box(v)
                    },
                    criterion::BatchSize::SmallInput,
                )
            });
        }
        group.finish();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("avx2 benchmarks are skipped on non-x86_64");
    }
}

fn bench_math_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::sum_dispatch");
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v = vec![1.234567_f64; d];

        group.bench_with_input(format!("d={d}"), &d, |b, &d| {
            b.iter(|| math::sum(black_box(&v), black_box(d)))
        });
    }
    group.finish();
}

fn bench_math_sum_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::sum_scalar");
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v = vec![1.234567_f64; d];

        group.bench_with_input(format!("d={d}"), &d, |b, &d| {
            b.iter(|| math::scalar::sum(black_box(&v), black_box(d)))
        });
    }
    group.finish();
}

fn bench_math_sum_avx2(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        let mut group = c.benchmark_group("math::sum_avx2");
        configure_group(&mut group);
        for &d in &MATH_DIMS {
            let v = vec![1.234567_f64; d];

            group.bench_with_input(format!("d={d}"), &d, |b, &d| {
                b.iter(|| math::avx2::sum(black_box(&v), black_box(d)))
            });
        }
        group.finish();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("avx2 benchmarks are skipped on non-x86_64");
    }
}

fn bench_math_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::norm_dispatch");
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v = vec![1.234567_f64; d];

        group.bench_with_input(format!("d={d}"), &d, |b, &d| {
            b.iter(|| math::norm(black_box(&v), black_box(d)))
        });
    }
    group.finish();
}

fn bench_math_norm_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::norm_scalar");
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v = vec![1.234567_f64; d];

        group.bench_with_input(format!("d={d}"), &d, |b, &d| {
            b.iter(|| math::scalar::norm(black_box(&v), black_box(d)))
        });
    }
    group.finish();
}

fn bench_math_norm_avx2(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        let mut group = c.benchmark_group("math::norm_avx2");
        configure_group(&mut group);
        for &d in &MATH_DIMS {
            let v = vec![1.234567_f64; d];

            group.bench_with_input(format!("d={d}"), &d, |b, &d| {
                b.iter(|| math::avx2::norm(black_box(&v), black_box(d)))
            });
        }
        group.finish();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("avx2 benchmarks are skipped on non-x86_64");
    }
}

criterion_group!(
    math_benches,
    bench_math_dot,
    bench_math_dot_scalar,
    bench_math_dot_avx2,
    bench_math_sqdist,
    bench_math_sqdist_scalar,
    bench_math_sqdist_avx2,
    bench_math_axpy,
    bench_math_axpy_scalar,
    bench_math_axpy_avx2,
    bench_math_sum,
    bench_math_sum_scalar,
    bench_math_sum_avx2,
    bench_math_norm,
    bench_math_norm_scalar,
    bench_math_norm_avx2,
);
criterion_main!(math_benches);
