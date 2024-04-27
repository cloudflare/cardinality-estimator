#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use cardinality_estimator::CardinalityEstimator;
use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
use std::hash::BuildHasherDefault;
use tabled::{
    settings::{Settings, Style},
    Table, Tabled,
};
use wyhash::WyHash;

#[derive(Tabled)]
struct Record {
    cardinality: usize,
    cardinality_estimator: String,
    amadeus_streaming: String,
    probabilistic_collections: String,
    hyperloglog: String,
    hyperloglogplus: String,
}

fn measure_memory_usage<T>(
    cardinality: usize,
    create: impl Fn() -> T,
    insert: impl Fn(&mut T, &usize),
) -> String
where
    T: Sized,
{
    let _profiler = dhat::Profiler::builder().testing().build();
    let mut estimator = create();
    for i in 0..cardinality {
        insert(&mut estimator, &i);
    }
    let stats = dhat::HeapStats::get();
    format!(
        "{} / {} / {}",
        std::mem::size_of::<T>(),
        stats.total_bytes,
        stats.total_blocks
    )
}

#[test]
fn test_allocations() {
    let results: Vec<Record> = std::iter::once(0)
        .chain((0..).map(|c| 1 << c))
        .take_while(|&c| c <= 1 << 20)
        .map(|cardinality| Record {
            cardinality,
            cardinality_estimator: measure_memory_usage(
                cardinality,
                || CardinalityEstimator::<12, 6>::new(),
                |est, i| est.insert(i),
            ),
            amadeus_streaming: measure_memory_usage(
                cardinality,
                || amadeus_streaming::HyperLogLog::new(0.01625),
                |est, i| est.push(i),
            ),
            probabilistic_collections: measure_memory_usage(
                cardinality,
                || probabilistic_collections::hyperloglog::HyperLogLog::<usize>::new(0.004),
                |est, i| est.insert(i),
            ),
            hyperloglog: measure_memory_usage(
                cardinality,
                || hyperloglog::HyperLogLog::new(0.004),
                |est, i| est.insert(i),
            ),
            hyperloglogplus: measure_memory_usage(
                cardinality,
                || {
                    HyperLogLogPlus::<usize, _>::new(12, BuildHasherDefault::<WyHash>::default())
                        .unwrap()
                },
                |est, i| est.insert(i),
            ),
        })
        .collect();

    let table_config = Settings::default().with(Style::markdown());
    let markdown = Table::new(results).with(table_config).to_string();
    std::fs::write(format!("{}/target/memory_allocations.md", env!("CARGO_MANIFEST_DIR")), &markdown).unwrap();
    println!("{}", markdown);
}
