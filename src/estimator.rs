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
//! - Cardinality in [449..] range - 3092 bytes (hyperloglog representation)
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
//! - For small cardinality range (<= 448 for P = 12, W = 6)
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
use std::fmt::{Debug, Formatter};
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::ops::Deref;

use wyhash::WyHash;

use crate::array::Array;
use crate::hyperloglog::HyperLogLog;
use crate::small::Small;

/// Mask used for storing and retrieving representation type stored in lowest 2 bits of `data` field.
const REPRESENTATION_MASK: usize = 0x0000_0000_0000_0003;

/// Ensure that only 64-bit architecture is being used.
#[cfg(target_pointer_width = "64")]
pub struct CardinalityEstimator<
    const P: usize = 12,
    const W: usize = 6,
    H: Hasher + Default = WyHash,
> {
    /// Raw data format described above
    pub(crate) data: usize,
    /// Zero-sized build hasher
    build_hasher: BuildHasherDefault<H>,
}

/// Four representation types supported by `CardinalityEstimator`
#[repr(u8)]
#[derive(Debug, PartialEq)]
pub enum Representation {
    Small = 0,
    Slice = 1,
    HashSet = 2,
    HyperLogLog = 3,
}

impl<const P: usize, const W: usize, H: Hasher + Default> CardinalityEstimator<P, W, H> {
    /// Ensure that `P` and `W` are in correct range at compile time
    const VALID_PARAMS: () = assert!(P >= 4 && P <= 18 && W >= 4 && W <= 6);
    /// Number of HyperLogLog registers
    const M: usize = 1 << P;
    /// HyperLogLog representation `u32` slice length based on #registers, stored zero registers, harmonic sum, and
    /// one extra element for branchless register updates (see `set_register` for more details).
    const HLL_SLICE_LEN: usize = Self::M * W / 32 + 3;

    /// Creates new instance of `CardinalityEstimator`
    #[inline]
    pub fn new() -> Self {
        // compile time check of params
        _ = Self::VALID_PARAMS;

        Self {
            // Start with empty small representation
            data: 0,
            build_hasher: BuildHasherDefault::default(),
        }
    }

    /// Return representation type of `CardinalityEstimator`
    #[inline]
    pub fn representation(&self) -> Representation {
        // SAFETY: representation is always one of four types stored in lowest 2 bits of `data` field.
        unsafe { std::mem::transmute((self.data & REPRESENTATION_MASK) as u8) }
    }

    /// Insert a hashable item into `CardinalityEstimator`
    #[inline]
    pub fn insert<T: Hash + ?Sized>(&mut self, item: &T) {
        let mut hasher = self.build_hasher.build_hasher();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        self.insert_hash(hash);
    }

    /// Insert hash into `CardinalityEstimator`
    #[inline]
    pub fn insert_hash(&mut self, hash: u64) {
        if self.representation() == HyperLogLog {
            let idx = (hash & ((1 << P) - 1)) as u32;
            let rank = (!hash >> P).trailing_zeros() + 1;
            Self::insert_into_hll(self.as_hll_slice_mut(), idx, rank);
        } else {
            self.insert_encoded_hash(Self::encode_hash(hash));
        }
    }

    /// Insert encoded hash into `CardinalityEstimator`
    #[inline]
    fn insert_encoded_hash(&mut self, h: u32) {
        match self.representation() {
            Small => self.insert_into_small(h),
            Slice => self.insert_into_slice(h),
            HashSet => self.insert_into_set(h),
            HyperLogLog => {
                let (idx, rank) = Self::decode_hash(h);
                Self::insert_into_hll(self.as_hll_slice_mut(), idx, rank);
            }
        }
    }

    /// Return cardinality estimate
    #[inline]
    pub fn estimate(&self) -> usize {
        match self.representation() {
            Small => self.estimate_small(),
            Slice => self.slice_len(),
            HashSet => self.as_hashset().len(),
            HyperLogLog => self.estimate_hll(),
        }
    }

    /// Merge cardinality estimators
    #[inline]
    pub fn merge(&mut self, rhs: &Self) {
        match (self.representation(), rhs.representation()) {
            (_, Small) => {
                let h1 = rhs.small_h1();
                let h2 = rhs.small_h2();
                if h1 != 0 {
                    self.insert_encoded_hash(h1);
                }
                if h2 != 0 {
                    self.insert_encoded_hash(h2);
                }
            }
            (_, Slice) => {
                for &h in &rhs.as_slice()[..rhs.slice_len()] {
                    self.insert_encoded_hash(h);
                }
            }
            (_, HashSet) => {
                for &h in rhs.as_hashset() {
                    self.insert_encoded_hash(h);
                }
            }
            (Small, HyperLogLog) => {
                let mut data = rhs.as_hll_slice().to_vec();
                let h1 = self.small_h1();
                let h2 = self.small_h2();
                if h1 != 0 {
                    let (idx, rank) = Self::decode_hash(h1);
                    Self::insert_into_hll(&mut data, idx, rank);
                }
                if h2 != 0 {
                    let (idx, rank) = Self::decode_hash(h2);
                    Self::insert_into_hll(&mut data, idx, rank);
                }
                self.set_hll_data(data);
            }
            (Slice, HyperLogLog) => {
                let slice_len = self.slice_len();
                let slice_data = self.as_slice_mut();
                let mut data = rhs.as_hll_slice().to_vec();
                for &h in &slice_data[..slice_len] {
                    let (idx, rank) = Self::decode_hash(h);
                    Self::insert_into_hll(&mut data, idx, rank);
                }
                drop(unsafe { Box::from_raw(slice_data) });
                self.set_hll_data(data);
            }
            (HashSet, HyperLogLog) => {
                let hashset_data = self.as_hashset_mut();
                let mut data = rhs.as_hll_slice().to_vec();
                for &h in hashset_data.iter() {
                    let (idx, rank) = Self::decode_hash(h);
                    Self::insert_into_hll(&mut data, idx, rank);
                }
                drop(unsafe { Box::from_raw(hashset_data) });
                self.set_hll_data(data);
            }
            (HyperLogLog, HyperLogLog) => {
                let lhs_data = self.as_hll_slice_mut();
                let rhs_data = rhs.as_hll_slice();
                for idx in 0..Self::M as u32 {
                    let lhs_rank = get_register::<W>(lhs_data, idx);
                    let rhs_rank = get_register::<W>(rhs_data, idx);
                    if rhs_rank > lhs_rank {
                        set_register::<W>(lhs_data, idx, lhs_rank, rhs_rank);
                    }
                }
            }
        }
    }

    /// Return 1-st encoded hash assuming small representation
    #[inline]
    fn small_h1(&self) -> u32 {
        ((self.data >> 2) & SMALL_MASK) as u32
    }

    /// Return 2-nd encoded hash assuming small representation
    #[inline]
    fn small_h2(&self) -> u32 {
        ((self.data >> 33) & SMALL_MASK) as u32
    }

    /// Insert encoded hash into small representation
    /// with potential upgrade to slice representation
    #[inline]
    fn insert_into_small(&mut self, h: u32) {
        // Retrieve 1-st encoded hash
        let h1 = self.small_h1();
        if h1 == 0 {
            self.data |= (h as usize) << 2;
            return;
        }
        if h1 == h {
            return;
        }
        // Retrieve 2-nd encoded hash
        let h2 = self.small_h2();
        if h2 == 0 {
            self.data |= (h as usize) << 33;
            return;
        }
        if h2 == h {
            return;
        }

        // both hashes occupied -> upgrade to slice representation
        self.set_slice_data(vec![h1, h2, h, 0], 3);
    }

    /// Insert encoded hash into slice representation
    #[inline]
    fn insert_into_slice(&mut self, h: u32) {
        let len = self.slice_len();
        let data = self.as_slice_mut();
        let cap = data.len();

        let found = if cap == 4 {
            contains_vectorized::<4>(&data, h)
        } else {
            // calculate rounded up slice length for efficient look up in batches
            let rlen = 8 * len.div_ceil(8);
            contains_vectorized::<8>(&data[..rlen], h)
        };

        if found {
            return;
        }

        if len < cap {
            // if there are available slots in current slice - append to it
            *unsafe { data.get_unchecked_mut(len) } = h;
            self.data = ((len + 1) << 56) | (PTR_MASK & data.as_ptr() as usize) | (Slice as usize);
            return;
        }

        if cap < MAX_SLICE_CAPACITY {
            let mut new_data = vec![0; cap * 2];
            new_data[..len].copy_from_slice(data);
            new_data[len] = h;
            drop(unsafe { Box::from_raw(data) });
            self.set_slice_data(new_data, len + 1);
        } else {
            let mut set = Box::new(HashSet::with_capacity(cap + 1));
            for h in data {
                set.insert(*h);
            }
            set.insert(h);

            let ptr = Box::into_raw(set);
            self.data = (ptr as usize) | (HashSet as usize);
        }
    }

    /// Insert encoded hash into hashset representation
    #[inline]
    fn insert_into_set(&mut self, h: u32) {
        let set = self.as_hashset_mut();

        if set.capacity() == set.len() {
            let (_, layout) = set.raw_table().allocation_info();
            // if doubling hashset capacity exceeds HyperLogLog representation size - migrate to it
            if 2 * layout.size() > Self::HLL_SLICE_LEN * size_of::<u32>() {
                let mut data = vec![0; Self::HLL_SLICE_LEN];
                data[0] = Self::M as u32;
                data[1] = (Self::M as f32).to_bits();

                for &h in set.iter() {
                    let (idx, new_rank) = Self::decode_hash(h);
                    Self::insert_into_hll(&mut data, idx, new_rank);
                }

                drop(unsafe { Box::from_raw(set) });
                self.set_hll_data(data);
                self.insert_encoded_hash(h);

                return;
            }
        }

        set.insert(h);
    }

    /// Set slice representation with new data
    #[inline]
    pub(crate) fn set_slice_data(&mut self, data: Vec<u32>, len: usize) {
        self.data = (len << 56) | (PTR_MASK & data.as_ptr() as usize) | (Slice as usize);
        std::mem::forget(data);
    }

    /// Set HyperLogLog representation with new data
    #[inline]
    fn set_hll_data(&mut self, data: Vec<u32>) {
        self.data = (PTR_MASK & (data.as_ptr() as usize)) | (HyperLogLog as usize);
        std::mem::forget(data);
    }

    /// Compute the sparse encoding of the given hash
    #[inline]
    fn encode_hash(hash: u64) -> u32 {
        let idx = (hash as u32) & ((1 << (32 - W - 1)) - 1);
        let rank = (!hash >> P).trailing_zeros() + 1;
        (idx << W) | rank
    }

    /// Return normal index and rank from encoded sparse hash
    #[inline]
    fn decode_hash(h: u32) -> (u32, u32) {
        let rank = h & ((1 << W) - 1);
        let idx = (h >> W) & ((1 << P) - 1);
        (idx, rank)
    }

    /// Return cardinality estimate of small representation
    #[inline]
    fn estimate_small(&self) -> usize {
        match (self.small_h1(), self.small_h2()) {
            (0, 0) => 0,
            (_, 0) => 1,
            (_, _) => 2,
        }
    }

    /// Return cardinality estimate of slice representation
    #[inline]
    fn slice_len(&self) -> usize {
        self.data >> 56
    }

    /// Return underlying slice of `u32` for slice representation
    #[inline]
    fn as_slice(&self) -> &[u32] {
        let ptr = (self.data & PTR_MASK) as *const u32;
        let cap = self.slice_len().next_power_of_two();
        unsafe { slice::from_raw_parts(ptr, cap) }
    }

    /// Return mutable underlying slice of `u32` for slice representation
    #[inline]
    fn as_slice_mut(&mut self) -> &mut [u32] {
        let ptr = (self.data & PTR_MASK) as *mut u32;
        let cap = self.slice_len().next_power_of_two();
        unsafe { slice::from_raw_parts_mut(ptr, cap) }
    }

    /// Return underlying `HashSet` of `u32` for hashset representation
    #[inline]
    fn as_hashset(&self) -> &HashSet<u32> {
        unsafe { &*((self.data & PTR_MASK) as *const HashSet<u32>) }
    }

    /// Return mutable underlying `HashSet` of `u32` for hashset representation
    #[inline]
    fn as_hashset_mut(&mut self) -> &mut HashSet<u32> {
        unsafe { &mut *((self.data & PTR_MASK) as *mut HashSet<u32>) }
    }

    /// Return underlying slice of `u32` for HyperLogLog representation
    #[inline]
    fn as_hll_slice(&self) -> &[u32] {
        let ptr = (self.data & PTR_MASK) as *const u32;
        unsafe { slice::from_raw_parts(ptr, Self::HLL_SLICE_LEN) }
    }

    /// Return mutable underlying slice of `u32` for HyperLogLog representation
    #[inline]
    fn as_hll_slice_mut(&mut self) -> &mut [u32] {
        let ptr = (self.data & PTR_MASK) as *mut u32;
        unsafe { slice::from_raw_parts_mut(ptr, Self::HLL_SLICE_LEN) }
    }

    /// Insert encoded hash into HyperLogLog representation
    #[inline]
    fn insert_into_hll(data: &mut [u32], idx: u32, new_rank: u32) {
        let old_rank = get_register::<W>(data, idx);
        if new_rank > old_rank {
            set_register::<W>(data, idx, old_rank, new_rank);
        }
    }

    /// Return cardinality estimate of HyperLogLog representation
    #[inline]
    fn estimate_hll(&self) -> usize {
        let data = self.as_hll_slice();
        let zeros = unsafe { *data.get_unchecked(0) };
        let sum = f32::from_bits(unsafe { *data.get_unchecked(1) }) as f64;
        let estimate = alpha(Self::M) * ((Self::M * (Self::M - zeros as usize)) as f64)
            / (sum + beta_horner(zeros as f64, P));
        (estimate + 0.5) as usize
    }

    /// Return memory size of `CardinalityEstimator`
    pub fn size_of(&self) -> usize {
        size_of::<Self>()
            + match self.representation() {
                Small => 0,
                Slice => size_of_val(self.as_slice()),
                HashSet => {
                    let (_, layout) = self.as_hashset().raw_table().allocation_info();
                    layout.size()
                }
                HyperLogLog => size_of_val(self.as_hll_slice()),
            }
    }
}

impl<const P: usize, const W: usize, H: Hasher + Default> Default
    for CardinalityEstimator<P, W, H>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const P: usize, const W: usize, H: Hasher + Default> Clone for CardinalityEstimator<P, W, H> {
    /// Clone `CardinalityEstimator`
    fn clone(&self) -> Self {
        let mut estimator = Self::new();
        estimator.merge(self);
        estimator
    }
}

impl<const P: usize, const W: usize, H: Hasher + Default> Drop for CardinalityEstimator<P, W, H> {
    /// Free memory occupied by `CardinalityEstimator`
    fn drop(&mut self) {
        match self.representation() {
            Small => {}
            Slice => {
                drop(unsafe { Box::from_raw(self.as_slice_mut()) });
            }
            HashSet => {
                drop(unsafe { Box::from_raw(self.as_hashset_mut()) });
            }
            HyperLogLog => {
                drop(unsafe { Box::from_raw(self.as_hll_slice_mut()) });
            }
        }
    }
}

impl<const P: usize, const W: usize, H: Hasher + Default> PartialEq
    for CardinalityEstimator<P, W, H>
{
    /// Compare cardinality estimators
    fn eq(&self, rhs: &Self) -> bool {
        if self.representation() != rhs.representation() {
            return false;
        }

        match self.representation() {
            Small => self.data == rhs.data,
            Slice => self.as_slice() == rhs.as_slice(),
            HashSet => self.as_hashset() == rhs.as_hashset(),
            HyperLogLog => self.as_hll_slice() == rhs.as_hll_slice(),
        }
    }
}

impl<const P: usize, const W: usize, H: Hasher + Default> Debug for CardinalityEstimator<P, W, H> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{ representation: {:?}, estimate: {}, size: {} }}",
            self.representation(),
            self.estimate(),
            self.size_of()
        )
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case(0 => "estimator = { representation: Small, estimate: 0, size: 8 } avg_err = 0.0000")]
    #[test_case(1 => "estimator = { representation: Small, estimate: 1, size: 8 } avg_err = 0.0000")]
    #[test_case(2 => "estimator = { representation: Small, estimate: 2, size: 8 } avg_err = 0.0000")]
    #[test_case(3 => "estimator = { representation: Array, estimate: 3, size: 24 } avg_err = 0.0000")]
    #[test_case(4 => "estimator = { representation: Array, estimate: 4, size: 24 } avg_err = 0.0000")]
    #[test_case(8 => "estimator = { representation: Array, estimate: 8, size: 40 } avg_err = 0.0000")]
    #[test_case(16 => "estimator = { representation: Array, estimate: 16, size: 72 } avg_err = 0.0000")]
    #[test_case(17 => "estimator = { representation: Array, estimate: 17, size: 136 } avg_err = 0.0000")]
    #[test_case(28 => "estimator = { representation: Array, estimate: 28, size: 136 } avg_err = 0.0000")]
    #[test_case(29 => "estimator = { representation: Array, estimate: 29, size: 136 } avg_err = 0.0000")]
    #[test_case(56 => "estimator = { representation: Array, estimate: 56, size: 264 } avg_err = 0.0000")]
    #[test_case(57 => "estimator = { representation: Array, estimate: 57, size: 264 } avg_err = 0.0000")]
    #[test_case(128 => "estimator = { representation: Array, estimate: 128, size: 520 } avg_err = 0.0000")]
    #[test_case(129 => "estimator = { representation: HLL, estimate: 131, size: 660 } avg_err = 0.0001")]
    #[test_case(256 => "estimator = { representation: HLL, estimate: 264, size: 660 } avg_err = 0.0119")]
    #[test_case(512 => "estimator = { representation: HLL, estimate: 512, size: 660 } avg_err = 0.0151")]
    #[test_case(1024 => "estimator = { representation: HLL, estimate: 1033, size: 660 } avg_err = 0.0172")]
    #[test_case(10_000 => "estimator = { representation: HLL, estimate: 10417, size: 660 } avg_err = 0.0281")]
    #[test_case(100_000 => "estimator = { representation: HLL, estimate: 93099, size: 660 } avg_err = 0.0351")]
    fn test_estimator_p10_w5(n: usize) -> String {
        evaluate_cardinality_estimator(CardinalityEstimator::<10, 5>::new(), n)
    }

    #[test_case(0 => "estimator = { representation: Small, estimate: 0, size: 8 } avg_err = 0.0000")]
    #[test_case(1 => "estimator = { representation: Small, estimate: 1, size: 8 } avg_err = 0.0000")]
    #[test_case(2 => "estimator = { representation: Small, estimate: 2, size: 8 } avg_err = 0.0000")]
    #[test_case(3 => "estimator = { representation: Array, estimate: 3, size: 24 } avg_err = 0.0000")]
    #[test_case(4 => "estimator = { representation: Array, estimate: 4, size: 24 } avg_err = 0.0000")]
    #[test_case(8 => "estimator = { representation: Array, estimate: 8, size: 40 } avg_err = 0.0000")]
    #[test_case(16 => "estimator = { representation: Array, estimate: 16, size: 72 } avg_err = 0.0000")]
    #[test_case(32 => "estimator = { representation: Array, estimate: 32, size: 136 } avg_err = 0.0000")]
    #[test_case(64 => "estimator = { representation: Array, estimate: 64, size: 264 } avg_err = 0.0000")]
    #[test_case(128 => "estimator = { representation: Array, estimate: 128, size: 520 } avg_err = 0.0000")]
    #[test_case(129 => "estimator = { representation: HLL, estimate: 130, size: 3092 } avg_err = 0.0001")]
    #[test_case(256 => "estimator = { representation: HLL, estimate: 254, size: 3092 } avg_err = 0.0029")]
    #[test_case(512 => "estimator = { representation: HLL, estimate: 498, size: 3092 } avg_err = 0.0068")]
    #[test_case(1024 => "estimator = { representation: HLL, estimate: 1012, size: 3092 } avg_err = 0.0130")]
    #[test_case(4096 => "estimator = { representation: HLL, estimate: 4105, size: 3092 } avg_err = 0.0089")]
    #[test_case(10_000 => "estimator = { representation: HLL, estimate: 10068, size: 3092 } avg_err = 0.0087")]
    #[test_case(100_000 => "estimator = { representation: HLL, estimate: 95628, size: 3092 } avg_err = 0.0182")]
    fn test_estimator_p12_w6(n: usize) -> String {
        evaluate_cardinality_estimator(CardinalityEstimator::<12, 6>::new(), n)
    }

    #[test_case(0 => "estimator = { representation: Small, estimate: 0, size: 8 } avg_err = 0.0000")]
    #[test_case(1 => "estimator = { representation: Small, estimate: 1, size: 8 } avg_err = 0.0000")]
    #[test_case(2 => "estimator = { representation: Small, estimate: 2, size: 8 } avg_err = 0.0000")]
    #[test_case(3 => "estimator = { representation: Array, estimate: 3, size: 24 } avg_err = 0.0000")]
    #[test_case(4 => "estimator = { representation: Array, estimate: 4, size: 24 } avg_err = 0.0000")]
    #[test_case(8 => "estimator = { representation: Array, estimate: 8, size: 40 } avg_err = 0.0000")]
    #[test_case(16 => "estimator = { representation: Array, estimate: 16, size: 72 } avg_err = 0.0000")]
    #[test_case(32 => "estimator = { representation: Array, estimate: 32, size: 136 } avg_err = 0.0000")]
    #[test_case(64 => "estimator = { representation: Array, estimate: 64, size: 264 } avg_err = 0.0000")]
    #[test_case(128 => "estimator = { representation: Array, estimate: 128, size: 520 } avg_err = 0.0000")]
    #[test_case(129 => "estimator = { representation: HLL, estimate: 129, size: 196628 } avg_err = 0.0000")]
    #[test_case(256 => "estimator = { representation: HLL, estimate: 256, size: 196628 } avg_err = 0.0000")]
    #[test_case(512 => "estimator = { representation: HLL, estimate: 511, size: 196628 } avg_err = 0.0004")]
    #[test_case(1024 => "estimator = { representation: HLL, estimate: 1022, size: 196628 } avg_err = 0.0014")]
    #[test_case(4096 => "estimator = { representation: HLL, estimate: 4100, size: 196628 } avg_err = 0.0009")]
    #[test_case(10_000 => "estimator = { representation: HLL, estimate: 10007, size: 196628 } avg_err = 0.0008")]
    #[test_case(100_000 => "estimator = { representation: HLL, estimate: 100240, size: 196628 } avg_err = 0.0011")]
    fn test_estimator_p18_w6(n: usize) -> String {
        evaluate_cardinality_estimator(CardinalityEstimator::<18, 6>::new(), n)
    }

    fn evaluate_cardinality_estimator<const P: usize, const W: usize>(
        mut e: CardinalityEstimator<P, W>,
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

        format!("estimator = {:?} avg_err = {:.4}", e, avg_relative_error)
    }

    #[test_case(0, 0 => "{ representation: Small, estimate: 0, size: 8 }")]
    #[test_case(0, 1 => "{ representation: Small, estimate: 1, size: 8 }")]
    #[test_case(1, 0 => "{ representation: Small, estimate: 1, size: 8 }")]
    #[test_case(1, 1 => "{ representation: Small, estimate: 2, size: 8 }")]
    #[test_case(1, 2 => "{ representation: Array, estimate: 3, size: 24 }")]
    #[test_case(2, 1 => "{ representation: Array, estimate: 3, size: 24 }")]
    #[test_case(2, 2 => "{ representation: Array, estimate: 4, size: 24 }")]
    #[test_case(2, 3 => "{ representation: Array, estimate: 5, size: 40 }")]
    #[test_case(2, 4 => "{ representation: Array, estimate: 6, size: 40 }")]
    #[test_case(4, 2 => "{ representation: Array, estimate: 6, size: 40 }")]
    #[test_case(3, 2 => "{ representation: Array, estimate: 5, size: 40 }")]
    #[test_case(3, 3 => "{ representation: Array, estimate: 6, size: 40 }")]
    #[test_case(3, 4 => "{ representation: Array, estimate: 7, size: 40 }")]
    #[test_case(4, 3 => "{ representation: Array, estimate: 7, size: 40 }")]
    #[test_case(4, 4 => "{ representation: Array, estimate: 8, size: 40 }")]
    #[test_case(4, 8 => "{ representation: Array, estimate: 12, size: 72 }")]
    #[test_case(8, 4 => "{ representation: Array, estimate: 12, size: 72 }")]
    #[test_case(4, 12 => "{ representation: Array, estimate: 16, size: 72 }")]
    #[test_case(12, 4 => "{ representation: Array, estimate: 16, size: 72 }")]
    #[test_case(1, 127 => "{ representation: Array, estimate: 128, size: 520 }")]
    #[test_case(1, 128 => "{ representation: HLL, estimate: 126, size: 3092 }")]
    #[test_case(127, 1 => "{ representation: Array, estimate: 128, size: 520 }")]
    #[test_case(128, 1 => "{ representation: HLL, estimate: 130, size: 3092 }")]
    #[test_case(128, 128 => "{ representation: HLL, estimate: 256, size: 3092 }")]
    #[test_case(512, 512 => "{ representation: HLL, estimate: 991, size: 3092 }")]
    #[test_case(10000, 0 => "{ representation: HLL, estimate: 10068, size: 3092 }")]
    #[test_case(0, 10000 => "{ representation: HLL, estimate: 10035, size: 3092 }")]
    #[test_case(4, 10000 => "{ representation: HLL, estimate: 10045, size: 3092 }")]
    #[test_case(10000, 4 => "{ representation: HLL, estimate: 10070, size: 3092 }")]
    #[test_case(17, 10000 => "{ representation: HLL, estimate: 10052, size: 3092 }")]
    #[test_case(10000, 17 => "{ representation: HLL, estimate: 10084, size: 3092 }")]
    #[test_case(10000, 10000 => "{ representation: HLL, estimate: 19743, size: 3092 }")]
    fn test_merge(lhs_n: usize, rhs_n: usize) -> String {
        let mut lhs = CardinalityEstimator::<12, 6>::new();
        for i in 0..lhs_n {
            lhs.insert(&i);
        }

        let mut rhs = CardinalityEstimator::<12, 6>::new();
        for i in -(rhs_n as isize)..0 {
            rhs.insert(&i);
        }

        lhs.merge(&rhs);

        format!("{:?}", lhs)
    }

    #[test]
    fn test_insert() {
        // Create a new CardinalityEstimator.
        let mut e = CardinalityEstimator::<12, 6>::new();

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
