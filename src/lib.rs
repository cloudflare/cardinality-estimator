//! `cardinality-estimator` is a Rust crate designed to estimate the number of distinct elements in a stream or dataset in an efficient manner.
//!
//! This library uses HyperLogLog++ with an optimized low memory footprint and high accuracy approach, suitable for large-scale data analysis tasks.
//!
//! Cardinality estimator allows to estimate number of distinct elements in the stream or dataset and is defined with const `P` and `W` parameters:
//! - `P`: precision parameter in [4..18] range, which defines
//!   number of bits to use for HyperLogLog register indices.
//! - `W`: width parameter in [4..6] range, which defines
//!   number of bits to use for HyperLogLog register width.
//!
//! # Data-structure design rationale
//!
//! ## Low memory footprint
//!
//! For parameters P = 12, W = 6:
//! - Cardinality in [0..2] range - 8 bytes (small representation)
//! - Cardinality in [3..4] range - 24 bytes (array representation)
//! - Cardinality in [5..8] range - 40 bytes (array representation)
//! - Cardinality in [9..16] range - 72 bytes (array representation)
//! - ...
//! - Cardinality in [65..128] range - 520 bytes (array representation)
//! - Cardinality in [129..] range - 3092 bytes (hyperloglog representation)
//!
//! ## Low latency
//! - Auto-vectorization for slice operations via compiler hints
//!   to use SIMD instructions when using `chunks_exact`.
//! - Number of zero registers and registers' harmonic sum are
//!   stored and updated dynamically as more data being inserted,
//!   allowing to have truly constant `estimate` operations.
//! - Efficient polynomial computation using Horner's method.
//!
//! ## High accuracy
//! - For small cardinality range (<= 128 for P = 12, W = 6)
//!   cardinality counted very accurately (within hash collisions chance)
//! - For large cardinality range HyperLogLog++ is used with LogLog-Beta bias correction.
//!   - Expected error (1.04 / sqrt(2^P)):
//!     - P = 10, W = 5: 0.0325
//!     - P = 12, W = 6: 0.0162
//!     - P = 14, W = 6: 0.0081
//!     - P = 18, W = 6: 0.0020
//!
//! # Data storage format
//! Cardinality estimator stores data in one of the three representations:
//! - `Small` representation - see `small` module for more details.
//! - `Array` representation - see `array` module for more details.
//! - `HyperLogLog` representation - see `hyperloglog` module for more details
//!
//! # Data Storage Format
//! The cardinality estimator stores data in one of three formats: `Small`, `Array`, and `HyperLogLog`.
//! See corresponding modules (`small`, `array`, `hyperloglog`) for more details.
mod array;
pub mod estimator;
mod hyperloglog;
mod representation;
#[cfg(feature = "with_serde")]
mod serde;
mod small;

pub use estimator::*;
