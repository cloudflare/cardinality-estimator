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
The data is stored in three different representations - `Small`, `Array`, and `HyperLogLog` - depending on the cardinality range.
For instance, for a cardinality of 0 to 2, only 8 bytes of stack memory and 0 bytes of heap memory are used.

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

We've benchmarked cardinality-estimator against several other crates in the ecosystem:
* [hyperloglog](https://crates.io/crates/hyperloglog)
* [hyperloglogplus](https://crates.io/crates/hyperloglogplus)
* [amadeus-streaming](https://crates.io/crates/amadeus-streaming)
* [probabilistic-collections](https://crates.io/crates/probabilistic-collections)

Please note, that `[hyperloglog](https://github.com/jedisct1/rust-hyperloglog/blob/1.0.2/src/lib.rs#L33)` and `[probabilistic-collections](https://gitlab.com/jeffrey-xiao/probabilistic-collections-rs/-/blob/da2a331e9679e4686bdcc772c369b639b9c33dee/src/hyperloglog.rs#L103)` crates have bug in calculation of precision `p` based on provided `probability`:
* incorrect formula: `p = (1.04 / error_probability).powi(2).ln().ceil() as usize;`
* corrected formula: `p = (1.04 / error_probability).powi(2).log2().ceil() as usize;`

We're continuously working to make `cardinality-estimator` the fastest, lightest, and most accurate tool for cardinality estimation in Rust.

### Memory usage
Table below compares memory usage of different cardinality estimators. The number in each cell represents `stack memory bytes / heap memory bytes / heap memory blocks` at each measured cardinality. 

| cardinality | cardinality_estimator | amadeus_streaming | probabilistic_collections | hyperloglog    | hyperloglogplus    |
|-------------|-----------------------|-------------------|---------------------------|----------------|--------------------|
| 0           | 8 / 0 / 0             | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4464 / 2 | 160 / 0 / 0        |
| 1           | 8 / 0 / 0             | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 36 / 1       |
| 2           | 8 / 0 / 0             | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 36 / 1       |
| 4           | 8 / 16 / 1            | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 92 / 2       |
| 8           | 8 / 48 / 2            | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 188 / 3      |
| 16          | 8 / 112 / 3           | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 364 / 4      |
| 32          | 8 / 240 / 4           | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 700 / 5      |
| 64          | 8 / 496 / 5           | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 1400 / 13    |
| 128         | 8 / 1008 / 6          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 3261 / 23    |
| 256         | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 10361 / 43   |
| 512         | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 38295 / 83   |
| 1024        | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 146816 / 163 |
| 2048        | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 207711 / 194 |
| 4096        | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 207711 / 194 |
| 8192        | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 207711 / 194 |
| 16384       | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 207711 / 194 |
| 32768       | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 207711 / 194 |
| 65536       | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 207711 / 194 |
| 131072      | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 207711 / 194 |
| 262144      | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 207711 / 194 |
| 524288      | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 207711 / 194 |
| 1048576     | 8 / 4092 / 7          | 48 / 4096 / 1     | 128 / 4096 / 1            | 120 / 4096 / 1 | 160 / 207711 / 194 |

### Insert performance
// TODO: add table + chart

### Estimate performance
// TODO: add table + chart
