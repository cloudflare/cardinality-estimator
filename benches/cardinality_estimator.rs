#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use std::hash::{BuildHasherDefault, Hash};

use cardinality_estimator::CardinalityEstimator;
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use hyperloglogplus::HyperLogLog as HyperLogLogTrait;
use pprof::criterion::{Output, PProfProfiler};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tabled::settings::{Settings, Style};
use tabled::{Table, Tabled};
use wyhash::WyHash;

/// Insert and estimate operations are benchmarked against cardinalities ranging from 0 to
/// `DEFAULT_MAX_CARDINALITY` or environment variable `N` (if defined) with cardinality doubled
/// with every iteration as [0, 1, 2, ..., N].
const DEFAULT_MAX_CARDINALITY: usize = 256;

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Protobuf));
    targets = benchmark
}
criterion_main!(benches);

fn benchmark(c: &mut Criterion) {
    let bench_results_path = std::env::var("BENCH_RESULTS_PATH").unwrap();
    let max_cardinality = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CARDINALITY);

    let cardinalities: Vec<usize> = std::iter::once(0)
        .chain((0..).map(|c| 1 << c))
        .take_while(|&c| c <= max_cardinality)
        .collect();

    let mut group = c.benchmark_group("insert");
    for &cardinality in &cardinalities {
        group.throughput(Throughput::Elements(cardinality.max(1) as u64));
        bench_insert::<CardinalityEstimatorMut>(&mut group, cardinality);
        bench_insert::<AmadeusStreamingEstimator>(&mut group, cardinality);
        bench_insert::<ProbabilisticCollections>(&mut group, cardinality);
        bench_insert::<HyperLogLog>(&mut group, cardinality);
        bench_insert::<HyperLogLogPlus>(&mut group, cardinality);
    }
    group.finish();

    let mut group = c.benchmark_group("estimate");
    group.throughput(Throughput::Elements(1));
    for &cardinality in &cardinalities {
        bench_estimate::<CardinalityEstimatorMut>(&mut group, cardinality);
        bench_estimate::<AmadeusStreamingEstimator>(&mut group, cardinality);
        bench_estimate::<ProbabilisticCollections>(&mut group, cardinality);
        bench_estimate::<HyperLogLog>(&mut group, cardinality);
        bench_estimate::<HyperLogLogPlus>(&mut group, cardinality);
    }
    group.finish();

    let results: Vec<StatRecord> = cardinalities
        .iter()
        .map(|&cardinality| StatRecord {
            cardinality,
            cardinality_estimator: measure_allocations::<CardinalityEstimatorMut>(cardinality),
            amadeus_streaming: measure_allocations::<AmadeusStreamingEstimator>(cardinality),
            probabilistic_collections: measure_allocations::<ProbabilisticCollections>(cardinality),
            hyperloglog: measure_allocations::<HyperLogLog>(cardinality),
            hyperloglogplus: measure_allocations::<HyperLogLogPlus>(cardinality),
        })
        .collect();

    let table_config = Settings::default().with(Style::markdown());
    std::fs::write(
        format!("{}/memory_usage.md", bench_results_path),
        Table::new(results).with(table_config).to_string(),
    )
    .unwrap();

    let results: Vec<StatRecord> = cardinalities
        .iter()
        .map(|&cardinality| StatRecord {
            cardinality,
            cardinality_estimator: measure_error::<CardinalityEstimatorMut>(cardinality),
            amadeus_streaming: measure_error::<AmadeusStreamingEstimator>(cardinality),
            probabilistic_collections: measure_error::<ProbabilisticCollections>(cardinality),
            hyperloglog: measure_error::<HyperLogLog>(cardinality),
            hyperloglogplus: measure_error::<HyperLogLogPlus>(cardinality),
        })
        .collect();

    let table_config = Settings::default().with(Style::markdown());
    std::fs::write(
        format!("{}/relative_error.md", bench_results_path),
        Table::new(results).with(table_config).to_string(),
    )
    .unwrap();
}

/// Cardinality estimator trait representing common estimator operations.
trait CardinalityEstimatorTrait<T: Hash + ?Sized> {
    fn new() -> Self;
    fn insert(&mut self, item: &T);
    fn estimate(&mut self) -> usize;
    fn merge(&mut self, rhs: &Self);
    fn name() -> String;
}

fn bench_insert<E: CardinalityEstimatorTrait<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    cardinality: usize,
) {
    group.bench_with_input(
        BenchmarkId::new(E::name(), cardinality),
        &cardinality,
        |b, &cardinality| {
            b.iter(|| {
                let mut estimator = E::new();
                for i in 0..black_box(cardinality) {
                    estimator.insert(black_box(&i));
                }
            });
        },
    );
}

fn bench_estimate<E: CardinalityEstimatorTrait<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    cardinality: usize,
) {
    group.bench_with_input(
        BenchmarkId::new(E::name(), cardinality),
        &cardinality,
        |b, &cardinality| {
            let mut estimator = E::new();
            for i in 0..black_box(cardinality) {
                estimator.insert(black_box(&i));
            }
            b.iter(|| estimator.estimate());
        },
    );
}

fn measure_allocations<E: CardinalityEstimatorTrait<usize>>(cardinality: usize) -> String {
    let _profiler = dhat::Profiler::builder().testing().build();
    let mut estimator = E::new();
    for i in 0..cardinality {
        estimator.insert(&i);
    }
    let stats = dhat::HeapStats::get();
    format!(
        "{} / {} / {}",
        std::mem::size_of::<E>(),
        stats.total_bytes,
        stats.total_blocks,
    )
}

fn measure_error<E: CardinalityEstimatorTrait<usize>>(cardinality: usize) -> String {
    let n = 100;
    let mut total_relative_error: f64 = 0.0;
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..n {
        let mut estimator = E::new();
        for _ in 0..cardinality {
            estimator.insert(&rng.gen());
        }
        let relative_error = if cardinality == 0 {
            0.0
        } else {
            (estimator.estimate() as f64 - cardinality as f64).abs() / cardinality as f64
        };
        total_relative_error += relative_error;
    }
    let avg_relative_error = total_relative_error / (n as f64);

    if avg_relative_error < 1.0 {
        format!("{:.4}", avg_relative_error)
    } else {
        format!("{:.2e}", avg_relative_error)
    }
}

#[derive(Tabled)]
struct StatRecord {
    cardinality: usize,
    cardinality_estimator: String,
    amadeus_streaming: String,
    probabilistic_collections: String,
    hyperloglog: String,
    hyperloglogplus: String,
}

struct CardinalityEstimatorMut(CardinalityEstimator<usize>);

impl CardinalityEstimatorTrait<usize> for CardinalityEstimatorMut {
    fn new() -> Self {
        Self(CardinalityEstimator::new())
    }

    fn insert(&mut self, item: &usize) {
        self.0.insert(item);
    }

    fn estimate(&mut self) -> usize {
        self.0.estimate()
    }

    fn merge(&mut self, rhs: &Self) {
        self.0.merge(&rhs.0);
    }

    fn name() -> String {
        "cardinality-estimator".to_string()
    }
}

struct AmadeusStreamingEstimator(amadeus_streaming::HyperLogLog<usize>);

impl CardinalityEstimatorTrait<usize> for AmadeusStreamingEstimator {
    fn new() -> Self {
        AmadeusStreamingEstimator(amadeus_streaming::HyperLogLog::new(0.01625))
    }

    fn insert(&mut self, item: &usize) {
        self.0.push(item)
    }

    fn estimate(&mut self) -> usize {
        self.0.len() as usize
    }

    fn merge(&mut self, rhs: &Self) {
        self.0.union(&rhs.0);
    }

    fn name() -> String {
        "amadeus-streaming".to_string()
    }
}

struct ProbabilisticCollections(probabilistic_collections::hyperloglog::HyperLogLog<usize>);

impl CardinalityEstimatorTrait<usize> for ProbabilisticCollections {
    fn new() -> Self {
        Self(probabilistic_collections::hyperloglog::HyperLogLog::new(
            0.004,
        ))
    }

    fn insert(&mut self, item: &usize) {
        self.0.insert(item);
    }

    fn estimate(&mut self) -> usize {
        self.0.len() as usize
    }

    fn merge(&mut self, rhs: &Self) {
        self.0.merge(&rhs.0);
    }

    fn name() -> String {
        "probabilistic-collections".to_string()
    }
}

struct HyperLogLog(hyperloglog::HyperLogLog);

impl CardinalityEstimatorTrait<usize> for HyperLogLog {
    fn new() -> Self {
        Self(hyperloglog::HyperLogLog::new(0.004))
    }

    fn insert(&mut self, item: &usize) {
        self.0.insert(item);
    }

    fn estimate(&mut self) -> usize {
        self.0.len() as usize
    }

    fn merge(&mut self, rhs: &Self) {
        self.0.merge(&rhs.0);
    }

    fn name() -> String {
        "hyperloglog".to_string()
    }
}

struct HyperLogLogPlus(hyperloglogplus::HyperLogLogPlus<usize, BuildHasherDefault<WyHash>>);

impl CardinalityEstimatorTrait<usize> for HyperLogLogPlus {
    fn new() -> Self {
        Self(
            hyperloglogplus::HyperLogLogPlus::new(12, BuildHasherDefault::<WyHash>::default())
                .unwrap(),
        )
    }

    fn insert(&mut self, item: &usize) {
        self.0.insert(item);
    }

    fn estimate(&mut self) -> usize {
        self.0.count() as usize
    }

    fn merge(&mut self, rhs: &Self) {
        self.0.merge(&rhs.0).unwrap();
    }

    fn name() -> String {
        "hyperloglogplus".to_string()
    }
}
