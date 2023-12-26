# cardinality-estimator
![build](https://img.shields.io/github/actions/workflow/status/cloudflare/cardinality-estimator/ci.yml?branch=main)
[![docs.rs](https://docs.rs/cardinality-estimator/badge.svg)](https://docs.rs/cardinality-estimator)
[![crates.io](https://img.shields.io/crates/v/cardinality-estimator.svg)](https://crates.io/crates/cardinality-estimator)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

`cardinality-estimator` is a Rust crate designed to estimate the number of distinct elements in a stream or dataset in an efficient manner.
This library uses HyperLogLog++ with an optimized low memory footprint and high accuracy approach, suitable for large-scale data analysis tasks.
We're using `cardinality-estimator` for large-scale machine learning, detailed in our blog post: ["Efficient Data Aggregation at Scale: Solving the Count-Distinct Problem with a New Cardinality Estimator"](http://blog.cloudflare.com/introducing-cardinality-estimator).

## Overview
Our cardinality estimator is highly efficient in terms of memory usage, latency, and accuracy.
This is achieved by leveraging a combination of unique data structure design, efficient algorithms, and HyperLogLog++ for high cardinality ranges.

## Getting Started
To use `cardinality-estimator`, add it to your `Cargo.toml` under `[dependencies]`:
```toml
[dependencies]
cardinality-estimator = "1.0.0"
```
Then, import `cardinality-estimator` in your Rust program:
```rust
use cardinality_estimator::CardinalityEstimator;

let estimator = CardinalityEstimator::<12, 6>::new();
estimator.insert_hash(123);
let estimate = estimator.estimate();

println!("estimate = {}", estimate);

// To merge two estimators
let estimator2 = CardinalityEstimator::<12, 6>::new();
estimator.merge(&estimator2);
let estimate = estimator2.estimate();

println!("estimate = {}", estimate);
```

Please refer to our [examples](examples) and [benchmarks](benches) in the repository for more complex scenarios.

## Low memory footprint
The `cardinality-estimator` achieves low memory footprint by leveraging an efficient data storage format.
The data is stored in three different representations - Small, Sparse, and Dense - depending on the cardinality range.
For instance, for a cardinality of 0 to 2, only 8 bytes of memory are used.

## Low latency
The crate offers low latency by using auto-vectorization for slice operations via compiler hints to use SIMD instructions.
The number of zero registers and registers' harmonic sum are stored and updated dynamically as more data is inserted, resulting in fast estimate operations.

## High accuracy
The cardinality-estimator achieves high accuracy by using precise counting for small cardinality ranges and HyperLogLog++ with LogLog-Beta bias correction for larger ranges.
This provides expected error rates as low as 0.02% for large cardinalities.

## Benchmarks

To run benchmarks you first need to install `cargo-criterion` binary:
```shell
cargo install cargo-criterion
```

Then benchmarks with output format JSON to save results for further analysis:
```shell
cargo criterion --bench cardinality_estimator --message-format json | tee benches/bench_results_$(date '+%Y%m%d_%H%M%S').json
```

To generate benchmark results charts and tables install and launch Jupyter lab:
```shell
cd benches
pip install -r requirements.txt
jupyter lab
```

We've benchmarked cardinality-estimator against several other crates in the ecosystem. Here are the results:

| Crate                                                                           | Features | Insert latency diff (%) | Estimate latency diff (%) | Memory footprint diff (%) |
|---------------------------------------------------------------------------------|----------|-------------------------|---------------------------|---------------------------|
| [cardinality-estimator](https://crates.io/crates/cardinality-estimator)         |          |                         |                           |                           |
| [hyperloglog](https://crates.io/crates/hyperloglog)                             |          |                         |                           |                           |
| [hyperloglog-rs](https://crates.io/crates/hyperloglog-rs)                       |          |                         |                           |                           |
| [hyperloglogplus](https://crates.io/crates/hyperloglogplus)                     |          |                         |                           |                           |
| [amadeus-streaming](https://crates.io/crates/amadeus-streaming)                 |          |                         |                           |                           |
| [streaming_algorithms](https://crates.io/crates/streaming_algorithms)           |          |                         |                           |                           |
| [probabilistic-collections](https://crates.io/crates/probabilistic-collections) |          |                         |                           |                           |

We're continuously working to make `cardinality-estimator` the fastest, lightest, and most accurate tool for cardinality estimation in Rust.