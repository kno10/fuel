use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use fuel::math;

const MATH_DIMS: [usize; 3] = [2, 4, 8];

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(200);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
}

fn bench_math_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::dot"); // Consolidated group name
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v1 = vec![1.234567_f64; d];
        let v2 = vec![2.345678_f64; d];

        // Benchmark Dispatch implementation
        group.bench_function(&format!("dispatch/{d}"), |b| {
            b.iter(|| math::dot(black_box(&v1), black_box(&v2), black_box(d)));
        });

        // Benchmark Scalar implementation
        group.bench_function(&format!("scalar/{d}"), |b| {
            b.iter(|| math::scalar::dot(black_box(&v1), black_box(&v2), black_box(d)));
        });

        // Benchmark SIMD implementations (if applicable)
        #[cfg(target_arch = "x86_64")]
        group.bench_function(&format!("simd/{d}"), |b| {
            b.iter(|| math::avx2::dot_f64(black_box(&v1), black_box(&v2), black_box(d)));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_function(&format!("simd/{d}"), |b| {
            b.iter(|| math::neon::dot_f64(black_box(&v1), black_box(&v2), black_box(d)));
        });
    }
    group.finish();
}

fn bench_math_sqdist(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::sqdist"); // Renamed group for clarity
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v1 = vec![1.234567_f64; d];
        let v2 = vec![2.345678_f64; d];

        // Benchmark Dispatch implementation
        group.bench_function(&format!("dispatch/{d}"), |b| {
            b.iter(|| math::sqdist(black_box(&v1), black_box(&v2), black_box(d)));
        });

        // Benchmark Scalar implementation
        group.bench_function(&format!("scalar/{d}"), |b| {
            b.iter(|| math::scalar::sqdist(black_box(&v1), black_box(&v2), black_box(d)));
        });

        // Benchmark SIMD implementations (if applicable)
        #[cfg(target_arch = "x86_64")]
        group.bench_function(&format!("simd/{d}"), |b| {
            b.iter(|| math::avx2::sqdist_f64(black_box(&v1), black_box(&v2), black_box(d)));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_function(&format!("simd/{d}"), |b| {
            b.iter(|| math::neon::sqdist_f64(black_box(&v1), black_box(&v2), black_box(d)));
        });
    }
    group.finish();
}

fn bench_math_axpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::axpy"); // Consolidated group name
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v2 = vec![1.234567_f64; d];
        let v1_base = vec![2.0_f64; d];

        // Benchmark Dispatch implementation
        group.bench_with_input(format!("dispatch/{d}"), &d, |b, &d| {
            b.iter_batched(
                || v1_base.clone(),
                |mut v| {
                    math::axpy(&mut v, black_box(3.15_f64), black_box(&v2), black_box(d));
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            )
        });

        // Benchmark Scalar implementation
        group.bench_function(&format!("scalar/{d}"), |b| {
            b.iter_batched(
                || v1_base.clone(),
                |mut v| {
                    math::scalar::axpy(&mut v, black_box(3.15_f64), black_box(&v2), black_box(d));
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // Benchmark SIMD implementations (if applicable)
        #[cfg(target_arch = "x86_64")]
        group.bench_function(&format!("simd/{d}"), |b| {
            b.iter_batched(
                || v1_base.clone(),
                |mut v| {
                    math::avx2::axpy_f64(&mut v, black_box(3.15_f64), black_box(&v2), black_box(d));
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_function(&format!("simd/{d}"), |b| {
            b.iter_batched(
                || v1_base.clone(),
                |mut v| {
                    math::neon::axpy_f64(&mut v, black_box(3.15_f64), black_box(&v2), black_box(d));
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_math_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::sum"); // Consolidated group name
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v = vec![1.234567_f64; d];

        // Benchmark Dispatch implementation
        group.bench_function(&format!("dispatch/{d}"), |b| {
            b.iter(|| math::sum(black_box(&v), black_box(d)));
        });

        // Benchmark Scalar implementation
        group.bench_function(&format!("scalar/{d}"), |b| {
            b.iter(|| math::scalar::sum(black_box(&v), black_box(d)));
        });

        // Benchmark SIMD implementations (if applicable)
        #[cfg(target_arch = "x86_64")]
        group.bench_function(&format!("simd/{d}"), |b| {
            b.iter(|| math::avx2::sum_f64(black_box(&v), black_box(d)));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_function(&format!("simd/{d}"), |b| {
            b.iter(|| math::neon::sum_f64(black_box(&v), black_box(d)));
        });
    }
    group.finish();
}

fn bench_math_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("math::norm"); // Consolidated group name
    configure_group(&mut group);
    for &d in &MATH_DIMS {
        let v = vec![1.234567_f64; d];

        // Benchmark Dispatch implementation
        group.bench_function(&format!("dispatch/{d}"), |b| {
            b.iter(|| math::norm(black_box(&v), black_box(d)));
        });

        // Benchmark Scalar implementation
        group.bench_function(&format!("scalar/{d}"), |b| {
            b.iter(|| math::scalar::norm(black_box(&v), black_box(d)));
        });

        // Benchmark SIMD implementations (if applicable)
        #[cfg(target_arch = "x86_64")]
        group.bench_function(&format!("simd/{d}"), |b| {
            b.iter(|| math::avx2::dot_f64(black_box(&v), black_box(&v), black_box(d)).sqrt());
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_function(&format!("simd/{d}"), |b| {
            b.iter(|| math::neon::dot_f64(black_box(&v), black_box(&v), black_box(d)).sqrt());
        });
    }
    group.finish();
}

criterion_group!(
    math_benches,
    bench_math_dot,
    bench_math_sqdist,
    bench_math_axpy,
    bench_math_sum,
    bench_math_norm,
);
criterion_main!(math_benches);
