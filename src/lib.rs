//! `cardinality-estimator` is a Rust crate designed to estimate the number of distinct elements in a stream or dataset in an efficient manner.
//!
//! This library uses HyperLogLog++ with an optimized low memory footprint and high accuracy approach, suitable for large-scale data analysis tasks.
mod array;
pub mod estimator;
mod hyperloglog;
mod representation;
#[cfg(feature = "with_serde")]
mod serde;
mod small;
