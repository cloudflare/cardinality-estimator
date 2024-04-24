# cardinality-estimator
![build](https://img.shields.io/github/actions/workflow/status/cloudflare/cardinality-estimator/ci.yml?branch=main)
[![docs.rs](https://docs.rs/cardinality-estimator/badge.svg)](https://docs.rs/cardinality-estimator)
[![crates.io](https://img.shields.io/crates/v/cardinality-estimator.svg)](https://crates.io/crates/cardinality-estimator)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

`cardinality-estimator` is a Rust crate designed to estimate the number of distinct elements in a stream or dataset in an efficient manner.
This library uses HyperLogLog++ with an optimized low memory footprint and high accuracy approach, suitable for large-scale data analysis tasks.
We're using `cardinality-estimator` for large-scale machine learning, computing cardinality features across multiple dimensions of the request.

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

let mut estimator = CardinalityEstimator::<12, 6>::new();
estimator.insert("test");
let estimate = estimator.estimate();

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
make bench-extended
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
| [hyperloglogplus](https://crates.io/crates/hyperloglogplus)                     |          |                         |                           |                           |
| [amadeus-streaming](https://crates.io/crates/amadeus-streaming)                 |          |                         |                           |                           |
| [probabilistic-collections](https://crates.io/crates/probabilistic-collections) |          |                         |                           |                           |

We're continuously working to make `cardinality-estimator` the fastest, lightest, and most accurate tool for cardinality estimation in Rust.