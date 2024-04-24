use std::hash::BuildHasherDefault;

use cardinality_estimator::CardinalityEstimator;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
use pprof::criterion::{Output, PProfProfiler};
use wyhash::WyHash;

/// Insert and estimate operations are benchmarked against cardinalities ranging from 0 to
/// `DEFAULT_MAX_CARDINALITY` or environment variable `N` (if defined) with cardinality doubled
/// with every iteration as [0, 1, 2, ..., N].
const DEFAULT_MAX_CARDINALITY: usize = 256;

fn get_max_cardinality() -> usize {
    std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CARDINALITY)
}

fn insert_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    let mut cardinality = 0;
    let max_cardinality = get_max_cardinality();
    while cardinality <= max_cardinality {
        group.throughput(Throughput::Elements(cardinality as u64));

        group.bench_with_input(
            BenchmarkId::new("cardinality-estimator", cardinality),
            &cardinality,
            |b, &cardinality| {
                b.iter(|| {
                    let mut estimator = CardinalityEstimator::<12, 6>::new();
                    for i in 0..black_box(cardinality) {
                        estimator.insert(black_box(&i));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hyperloglog", cardinality),
            &cardinality,
            |b, &cardinality| {
                b.iter(|| {
                    let mut estimator = hyperloglog::HyperLogLog::new(0.01625);
                    for i in 0..black_box(cardinality) {
                        estimator.insert(black_box(&i));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hyperloglogplus", cardinality),
            &cardinality,
            |b, &cardinality| {
                b.iter(|| {
                    let mut estimator: HyperLogLogPlus<&usize, _> =
                        HyperLogLogPlus::new(12, BuildHasherDefault::<WyHash>::default()).unwrap();
                    for i in 0..black_box(cardinality) {
                        estimator.insert(black_box(&i));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("probabilistic-collections", cardinality),
            &cardinality,
            |b, &cardinality| {
                b.iter(|| {
                    let mut estimator =
                        probabilistic_collections::hyperloglog::HyperLogLog::<usize>::new(0.01625);
                    for i in 0..black_box(cardinality) {
                        estimator.insert(black_box(&i));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("amadeus-streaming", cardinality),
            &cardinality,
            |b, &cardinality| {
                b.iter(|| {
                    let mut estimator = amadeus_streaming::HyperLogLog::new(0.01625);
                    for i in 0..black_box(cardinality) {
                        estimator.push(black_box(&i));
                    }
                });
            },
        );

        if cardinality == 0 {
            cardinality = 1;
        } else {
            cardinality *= 2;
        }
    }

    group.finish();
}

fn estimate_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate");
    group.throughput(Throughput::Elements(1));

    let mut cardinality = 0;
    let max_cardinality = get_max_cardinality();
    while cardinality <= max_cardinality {
        group.bench_with_input(
            BenchmarkId::new("cardinality-estimator", cardinality),
            &cardinality,
            |b, &cardinality| {
                let mut estimator = CardinalityEstimator::<12, 6>::new();
                for i in 0..black_box(cardinality) {
                    estimator.insert(black_box(&i));
                }
                b.iter(|| estimator.estimate());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hyperloglog", cardinality),
            &cardinality,
            |b, &cardinality| {
                let mut estimator = hyperloglog::HyperLogLog::new(0.01625);
                for i in 0..black_box(cardinality) {
                    estimator.insert(black_box(&i));
                }
                b.iter(|| estimator.len());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hyperloglogplus", cardinality),
            &cardinality,
            |b, &cardinality| {
                let mut estimator: HyperLogLogPlus<&usize, _> =
                    HyperLogLogPlus::new(12, BuildHasherDefault::<WyHash>::default()).unwrap();
                for i in 0..black_box(cardinality) {
                    estimator.insert(black_box(&i));
                }
                b.iter(|| estimator.count());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("probabilistic-collections", cardinality),
            &cardinality,
            |b, &cardinality| {
                let mut estimator =
                    probabilistic_collections::hyperloglog::HyperLogLog::<usize>::new(0.01625);
                for i in 0..black_box(cardinality) {
                    estimator.insert(black_box(&i));
                }
                b.iter(|| estimator.len());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("amadeus-streaming", cardinality),
            &cardinality,
            |b, &cardinality| {
                let mut estimator = amadeus_streaming::HyperLogLog::new(0.01625);
                for i in 0..black_box(cardinality) {
                    estimator.push(black_box(&i));
                }
                b.iter(|| estimator.len());
            },
        );

        if cardinality == 0 {
            cardinality = 1;
        } else {
            cardinality *= 2;
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Protobuf));
    targets = insert_benchmark, estimate_benchmark
}
criterion_main!(benches);
