//! Cardinality estimator allows to estimate number of distinct elements
//! in the stream or dataset and is defined with const `P` and `W` parameters:
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
//!     - P = 10, W = 5: 3.25%
//!     - P = 12, W = 6: 1.62%
//!     - P = 14, W = 6: 0.81%
//!     - P = 18, W = 6: 0.02%
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
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;

use wyhash::WyHash;

use crate::representation::{Representation, RepresentationTrait};

/// Ensure that only 64-bit architecture is being used.
#[cfg(target_pointer_width = "64")]
pub struct CardinalityEstimator<T, H = WyHash, const P: usize = 12, const W: usize = 6>
where
    T: Hash + ?Sized,
    H: Hasher + Default,
{
    /// Raw data format described above
    pub(crate) data: usize,
    /// Zero-sized build hasher
    build_hasher: BuildHasherDefault<H>,
    /// Zero-sized phantom data for type `T`
    _phantom_data: PhantomData<T>,
}

/// Cardinality estimator trait representing common estimator operations.
pub trait CardinalityEstimatorTrait<T: Hash + ?Sized> {
    fn new() -> Self;
    fn insert(&mut self, item: &T);
    fn estimate(&mut self) -> usize;
    fn merge(&mut self, rhs: &Self);
    fn name() -> String;
}

impl<T, H, const P: usize, const W: usize> CardinalityEstimatorTrait<T>
    for CardinalityEstimator<T, H, P, W>
where
    T: Hash + ?Sized,
    H: Hasher + Default,
{
    /// Creates new instance of `CardinalityEstimator`
    #[inline]
    fn new() -> Self {
        // compile time check of params
        #[allow(clippy::let_unit_value)]
        let _ = Self::VALID_PARAMS;

        Self {
            // Start with empty small representation
            data: 0,
            build_hasher: BuildHasherDefault::default(),
            _phantom_data: PhantomData,
        }
    }

    /// Insert a hashable item into `CardinalityEstimator`
    #[inline]
    fn insert(&mut self, item: &T) {
        let mut hasher = self.build_hasher.build_hasher();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        self.insert_hash(hash);
    }

    /// Return cardinality estimate
    #[inline]
    fn estimate(&mut self) -> usize {
        self.representation().estimate()
    }

    /// Merge cardinality estimators
    #[inline]
    fn merge(&mut self, rhs: &Self) {
        match (self.representation(), rhs.representation()) {
            (_, Representation::Small(rhs_small)) => {
                for h in rhs_small.items() {
                    if h != 0 {
                        self.insert_encoded_hash(h);
                    }
                }
            }
            (_, Representation::Array(rhs_arr)) => {
                for &h in rhs_arr.deref() {
                    self.insert_encoded_hash(h);
                }
            }
            (Representation::Small(lhs_small), Representation::Hll(rhs_hll)) => {
                let mut hll = rhs_hll.clone();
                for h in lhs_small.items() {
                    if h != 0 {
                        hll.insert_encoded_hash(h);
                    }
                }
                self.data = hll.to_data();
            }
            (Representation::Array(mut lhs_arr), Representation::Hll(rhs_hll)) => {
                let mut hll = rhs_hll.clone();
                for &h in lhs_arr.deref() {
                    hll.insert_encoded_hash(h);
                }
                lhs_arr.drop();
                self.data = hll.to_data();
            }
            (Representation::Hll(mut lhs_hll), Representation::Hll(rhs_hll)) => {
                lhs_hll.merge(&rhs_hll);
            }
        }
    }

    fn name() -> String {
        "cardinality-estimator".to_string()
    }
}

impl<T, H, const P: usize, const W: usize> CardinalityEstimator<T, H, P, W>
where
    T: Hash + ?Sized,
    H: Hasher + Default,
{
    /// Ensure that `P` and `W` are in correct range at compile time
    const VALID_PARAMS: () = assert!(P >= 4 && P <= 18 && W >= 4 && W <= 6);

    /// Returns the representation type of `CardinalityEstimator`.
    #[inline]
    pub(crate) fn representation(&self) -> Representation<P, W> {
        Representation::<P, W>::from_data(self.data)
    }

    /// Insert hash into `CardinalityEstimator`
    #[inline]
    pub fn insert_hash(&mut self, hash: u64) {
        self.insert_encoded_hash(Self::encode_hash(hash));
    }

    /// Insert encoded hash into `CardinalityEstimator`
    #[inline]
    fn insert_encoded_hash(&mut self, h: u32) {
        self.data = self.representation().insert_encoded_hash(h);
    }

    /// Compute the sparse encoding of the given hash
    #[inline]
    fn encode_hash(hash: u64) -> u32 {
        let idx = (hash as u32) & ((1 << (32 - W - 1)) - 1);
        let rank = (!hash >> P).trailing_zeros() + 1;
        (idx << W) | rank
    }

    /// Return memory size of `CardinalityEstimator`
    pub fn size_of(&self) -> usize {
        self.representation().size_of()
    }
}

impl<T, H, const P: usize, const W: usize> Default for CardinalityEstimator<T, H, P, W>
where
    T: Hash + ?Sized,
    H: Hasher + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, H, const P: usize, const W: usize> Clone for CardinalityEstimator<T, H, P, W>
where
    T: Hash + ?Sized,
    H: Hasher + Default,
{
    /// Clone `CardinalityEstimator`
    fn clone(&self) -> Self {
        let mut estimator = Self::new();
        estimator.merge(self);
        estimator
    }
}

impl<T, H, const P: usize, const W: usize> Drop for CardinalityEstimator<T, H, P, W>
where
    T: Hash + ?Sized,
    H: Hasher + Default,
{
    /// Free memory occupied by `CardinalityEstimator`
    #[inline]
    fn drop(&mut self) {
        self.representation().drop();
    }
}

impl<T, H, const P: usize, const W: usize> PartialEq for CardinalityEstimator<T, H, P, W>
where
    T: Hash + ?Sized,
    H: Hasher + Default,
{
    /// Compare cardinality estimators
    fn eq(&self, rhs: &Self) -> bool {
        self.representation() == rhs.representation()
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case(0 => "representation: Small(estimate: 0, size: 8), avg_err: 0.0000")]
    #[test_case(1 => "representation: Small(estimate: 1, size: 8), avg_err: 0.0000")]
    #[test_case(2 => "representation: Small(estimate: 2, size: 8), avg_err: 0.0000")]
    #[test_case(3 => "representation: Array(estimate: 3, size: 24), avg_err: 0.0000")]
    #[test_case(4 => "representation: Array(estimate: 4, size: 24), avg_err: 0.0000")]
    #[test_case(8 => "representation: Array(estimate: 8, size: 40), avg_err: 0.0000")]
    #[test_case(16 => "representation: Array(estimate: 16, size: 72), avg_err: 0.0000")]
    #[test_case(17 => "representation: Array(estimate: 17, size: 136), avg_err: 0.0000")]
    #[test_case(28 => "representation: Array(estimate: 28, size: 136), avg_err: 0.0000")]
    #[test_case(29 => "representation: Array(estimate: 29, size: 136), avg_err: 0.0000")]
    #[test_case(56 => "representation: Array(estimate: 56, size: 264), avg_err: 0.0000")]
    #[test_case(57 => "representation: Array(estimate: 57, size: 264), avg_err: 0.0000")]
    #[test_case(128 => "representation: Array(estimate: 128, size: 520), avg_err: 0.0000")]
    #[test_case(129 => "representation: Hll(estimate: 131, size: 660), avg_err: 0.0001")]
    #[test_case(256 => "representation: Hll(estimate: 264, size: 660), avg_err: 0.0119")]
    #[test_case(512 => "representation: Hll(estimate: 512, size: 660), avg_err: 0.0151")]
    #[test_case(1024 => "representation: Hll(estimate: 1033, size: 660), avg_err: 0.0172")]
    #[test_case(10_000 => "representation: Hll(estimate: 10417, size: 660), avg_err: 0.0281")]
    #[test_case(100_000 => "representation: Hll(estimate: 93099, size: 660), avg_err: 0.0351")]
    fn test_estimator_p10_w5(n: usize) -> String {
        evaluate_cardinality_estimator(CardinalityEstimator::<usize, WyHash, 10, 5>::new(), n)
    }

    #[test_case(0 => "representation: Small(estimate: 0, size: 8), avg_err: 0.0000")]
    #[test_case(1 => "representation: Small(estimate: 1, size: 8), avg_err: 0.0000")]
    #[test_case(2 => "representation: Small(estimate: 2, size: 8), avg_err: 0.0000")]
    #[test_case(3 => "representation: Array(estimate: 3, size: 24), avg_err: 0.0000")]
    #[test_case(4 => "representation: Array(estimate: 4, size: 24), avg_err: 0.0000")]
    #[test_case(8 => "representation: Array(estimate: 8, size: 40), avg_err: 0.0000")]
    #[test_case(16 => "representation: Array(estimate: 16, size: 72), avg_err: 0.0000")]
    #[test_case(32 => "representation: Array(estimate: 32, size: 136), avg_err: 0.0000")]
    #[test_case(64 => "representation: Array(estimate: 64, size: 264), avg_err: 0.0000")]
    #[test_case(128 => "representation: Array(estimate: 128, size: 520), avg_err: 0.0000")]
    #[test_case(129 => "representation: Hll(estimate: 130, size: 3092), avg_err: 0.0001")]
    #[test_case(256 => "representation: Hll(estimate: 254, size: 3092), avg_err: 0.0029")]
    #[test_case(512 => "representation: Hll(estimate: 498, size: 3092), avg_err: 0.0068")]
    #[test_case(1024 => "representation: Hll(estimate: 1012, size: 3092), avg_err: 0.0130")]
    #[test_case(4096 => "representation: Hll(estimate: 4105, size: 3092), avg_err: 0.0089")]
    #[test_case(10_000 => "representation: Hll(estimate: 10068, size: 3092), avg_err: 0.0087")]
    #[test_case(100_000 => "representation: Hll(estimate: 95628, size: 3092), avg_err: 0.0182")]
    fn test_estimator_p12_w6(n: usize) -> String {
        evaluate_cardinality_estimator(CardinalityEstimator::<usize, WyHash, 12, 6>::new(), n)
    }

    #[test_case(0 => "representation: Small(estimate: 0, size: 8), avg_err: 0.0000")]
    #[test_case(1 => "representation: Small(estimate: 1, size: 8), avg_err: 0.0000")]
    #[test_case(2 => "representation: Small(estimate: 2, size: 8), avg_err: 0.0000")]
    #[test_case(3 => "representation: Array(estimate: 3, size: 24), avg_err: 0.0000")]
    #[test_case(4 => "representation: Array(estimate: 4, size: 24), avg_err: 0.0000")]
    #[test_case(8 => "representation: Array(estimate: 8, size: 40), avg_err: 0.0000")]
    #[test_case(16 => "representation: Array(estimate: 16, size: 72), avg_err: 0.0000")]
    #[test_case(32 => "representation: Array(estimate: 32, size: 136), avg_err: 0.0000")]
    #[test_case(64 => "representation: Array(estimate: 64, size: 264), avg_err: 0.0000")]
    #[test_case(128 => "representation: Array(estimate: 128, size: 520), avg_err: 0.0000")]
    #[test_case(129 => "representation: Hll(estimate: 129, size: 196628), avg_err: 0.0000")]
    #[test_case(256 => "representation: Hll(estimate: 256, size: 196628), avg_err: 0.0000")]
    #[test_case(512 => "representation: Hll(estimate: 511, size: 196628), avg_err: 0.0004")]
    #[test_case(1024 => "representation: Hll(estimate: 1022, size: 196628), avg_err: 0.0014")]
    #[test_case(4096 => "representation: Hll(estimate: 4100, size: 196628), avg_err: 0.0009")]
    #[test_case(10_000 => "representation: Hll(estimate: 10007, size: 196628), avg_err: 0.0008")]
    #[test_case(100_000 => "representation: Hll(estimate: 100240, size: 196628), avg_err: 0.0011")]
    fn test_estimator_p18_w6(n: usize) -> String {
        evaluate_cardinality_estimator(CardinalityEstimator::<usize, WyHash, 18, 6>::new(), n)
    }

    fn evaluate_cardinality_estimator<const P: usize, const W: usize>(
        mut e: CardinalityEstimator<usize, WyHash, P, W>,
        n: usize,
    ) -> String {
        let mut total_relative_error: f64 = 0.0;
        for i in 0..n {
            e.insert(&i);
            let estimate = e.estimate() as f64;
            let actual = (i + 1) as f64;
            let error = estimate - actual;
            let relative_error = error.abs() / actual;
            total_relative_error += relative_error;
        }

        let avg_relative_error = total_relative_error / ((n + 1) as f64);

        // Compute the expected standard error for HyperLogLog based on the precision
        let standard_error = 1.04 / 2.0f64.powi(P as i32).sqrt();
        let tolerance = 1.2;

        assert!(
            avg_relative_error <= standard_error * tolerance,
            "Average relative error {} exceeds acceptable threshold {}",
            avg_relative_error,
            standard_error * tolerance
        );

        format!(
            "representation: {:?}, avg_err: {:.4}",
            e.representation(),
            avg_relative_error
        )
    }

    #[test_case(0, 0 => "Small(estimate: 0, size: 8)")]
    #[test_case(0, 1 => "Small(estimate: 1, size: 8)")]
    #[test_case(1, 0 => "Small(estimate: 1, size: 8)")]
    #[test_case(1, 1 => "Small(estimate: 2, size: 8)")]
    #[test_case(1, 2 => "Array(estimate: 3, size: 24)")]
    #[test_case(2, 1 => "Array(estimate: 3, size: 24)")]
    #[test_case(2, 2 => "Array(estimate: 4, size: 24)")]
    #[test_case(2, 3 => "Array(estimate: 5, size: 40)")]
    #[test_case(2, 4 => "Array(estimate: 6, size: 40)")]
    #[test_case(4, 2 => "Array(estimate: 6, size: 40)")]
    #[test_case(3, 2 => "Array(estimate: 5, size: 40)")]
    #[test_case(3, 3 => "Array(estimate: 6, size: 40)")]
    #[test_case(3, 4 => "Array(estimate: 7, size: 40)")]
    #[test_case(4, 3 => "Array(estimate: 7, size: 40)")]
    #[test_case(4, 4 => "Array(estimate: 8, size: 40)")]
    #[test_case(4, 8 => "Array(estimate: 12, size: 72)")]
    #[test_case(8, 4 => "Array(estimate: 12, size: 72)")]
    #[test_case(4, 12 => "Array(estimate: 16, size: 72)")]
    #[test_case(12, 4 => "Array(estimate: 16, size: 72)")]
    #[test_case(1, 127 => "Array(estimate: 128, size: 520)")]
    #[test_case(1, 128 => "Hll(estimate: 130, size: 3092)")]
    #[test_case(127, 1 => "Array(estimate: 128, size: 520)")]
    #[test_case(128, 1 => "Hll(estimate: 130, size: 3092)")]
    #[test_case(128, 128 => "Hll(estimate: 254, size: 3092)")]
    #[test_case(512, 512 => "Hll(estimate: 1012, size: 3092)")]
    #[test_case(10000, 0 => "Hll(estimate: 10068, size: 3092)")]
    #[test_case(0, 10000 => "Hll(estimate: 10068, size: 3092)")]
    #[test_case(4, 10000 => "Hll(estimate: 10068, size: 3092)")]
    #[test_case(10000, 4 => "Hll(estimate: 10068, size: 3092)")]
    #[test_case(17, 10000 => "Hll(estimate: 10073, size: 3092)")]
    #[test_case(10000, 17 => "Hll(estimate: 10073, size: 3092)")]
    #[test_case(10000, 10000 => "Hll(estimate: 19974, size: 3092)")]
    fn test_merge(lhs_n: usize, rhs_n: usize) -> String {
        let mut lhs = CardinalityEstimator::<usize, WyHash, 12, 6>::new();
        for i in 0..lhs_n {
            lhs.insert(&i);
        }

        let mut rhs = CardinalityEstimator::<usize, WyHash, 12, 6>::new();
        for i in lhs_n..lhs_n + rhs_n {
            rhs.insert(&i);
        }

        lhs.merge(&rhs);

        format!("{:?}", lhs.representation())
    }

    #[test]
    fn test_insert() {
        // Create a new CardinalityEstimator.
        let mut e = CardinalityEstimator::<str, WyHash, 12, 6>::new();

        // Ensure initial estimate is 0.
        assert_eq!(e.estimate(), 0);

        // Insert a test item and validate estimate.
        e.insert("test item 1");
        assert_eq!(e.estimate(), 1);

        // Re-insert the same item, estimate should remain the same.
        e.insert("test item 1");
        assert_eq!(e.estimate(), 1);

        // Insert a new distinct item, estimate should increase.
        e.insert("test item 2");
        assert_eq!(e.estimate(), 2);
    }
}
