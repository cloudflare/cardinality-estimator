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
//! - Cardinality in [3..4] range - 28 bytes (sparse representation)
//! - Cardinality in [5..8] range - 44 bytes (sparse representation)
//! - Cardinality in [9..16] range - 76 bytes (sparse representation)
//! - ...
//! - Cardinality in [512..] range - 3092 bytes (dense representation)
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
//! - For small cardinality range (<= 512 for P = 12, W = 6)
//!   cardinality counted precisely (within hash collisions chance)
//! - For large cardinality range HyperLogLog++ is used with LogLog-Beta bias correction.
//!   - Expected error:
//!     P = 10, W = 5: 1.04 / sqrt(2^10) = 3.25%
//!     P = 12, W = 6: 1.04 / sqrt(2^12) = 1.62%
//!     P = 14, W = 6: 1.04 / sqrt(2^14) = 0.81%
//!     P = 18, W = 6: 1.04 / sqrt(2^18) = 0.02%
//!
//! # Data storage format
//! Cardinality estimator stores data in one of the three representations:
//!
//! ## Small representation
//! Allows to estimate cardinality in [0..2] range and uses only 8 bytes of memory.
//!
//! The `data` format of small representation:
//! - 0..31 bits    - store 32-bit encoded hash of the value.
//! - 32..62 bits   - store 32-bit encoded hash using 31 bits (only if its lowest bit set to 0).
//! - 63 bit        - bit set to 1 indicates whether small representation is used (see `SMALL_MASK`).
//!
//! ## Sparse representation
//! Allows to estimate medium cardinality in [3..N] range, where `N` is based on `P` and `W`.
//!
//! The `data` format of sparse representation:
//! - 0..58 bits    - store pointer to `u32` slice (on `x86_64 systems only 48-bits are needed).
//! - 59..62 bits   - store 4-bit slice capacity encoded as power of 2, e.g. 1 for 4, 2 for 8, etc.
//! - 63 bit        - bit set to 0 indicates whether sparse or dense representation is used.
//!
//! Slice encoding:
//! - data[0]       - stores actual number of hashes `N`
//! - data[1..N+1]  - store N `u32` encoded hashes
//! - data[N+1..]   - store zeros used for future hashes
//!
//! ## Dense representation
//! Allows to estimate large cardinality in `[N..]` range, where `N` is based on `P` and `W`.
//! Dense representation uses modified HyperLogLog++ with `M` registers of `W` width.
//!
//! Original HyperLogLog++ paper:
//! https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40671.pdf
//!
//! The `data` format of sparse representation:
//! - 0..58 bits    - store pointer to `u32` slice (on `x86_64 systems only 48-bits are needed).
//! - 59..62 bits   - store 4-bit 0 value indicating that dense representation is used.
//! - 63 bit        - bit set to 0 indicates whether sparse or dense representation is used.
//!
//! Slice encoding:
//! - data[0]       - stores number of HyperLogLog registers set to 0.
//! - data[1]       - stores harmonic sum of HyperLogLog registers (`f32` transmuted into `u32`).
//! - data[2..]     - stores register ranks using `W` bits per each register.

use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::slice;

use crate::beta::beta_horner;

use xxhash_rust::xxh3::Xxh3;

/// Mask used to check whether cardinality estimator uses small representation
const SMALL_MASK: usize = 0x8000_0000_0000_0000;
/// Mask used to store 1-st encoded hash in small representation
const SMALL_1_MASK: usize = 0x0000_0000_ffff_ffff;
/// Mask used to store 2-nd encoded hash in small representation
const SMALL_2_MASK: usize = 0x7fff_ffff_0000_0000;
/// Mask used to obtain slice pointer for sparse and dense representations
const SLICE_PTR_MASK: usize = 0x07ff_ffff_ffff_ffff;
/// Mask used to obtain slice length for sparse representation
const SLICE_LEN_MASK: usize = 0x7800_0000_0000_0000;
/// Offset used to store slice length for sparse representation
const SLICE_LEN_OFFSET: usize = 59;

/// Ensure that only 64-bit architecture is being used.
#[cfg(target_pointer_width = "64")]
#[derive(Debug)]
pub struct CardinalityEstimator<const P: usize = 12, const W: usize = 6, H: Hasher + Default = Xxh3>
{
    /// Raw data format described above
    pub(crate) data: usize,
    /// Phantom field for the hasher
    _phantom_hasher: PhantomData<H>,
}

impl<const P: usize, const W: usize, H: Hasher + Default> Default
    for CardinalityEstimator<P, W, H>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const P: usize, const W: usize, H: Hasher + Default> CardinalityEstimator<P, W, H> {
    /// Ensure that `P` and `W` are in correct range at compile time
    const VALID_PARAMS: () = assert!(P >= 4 && P <= 18 && W >= 4 && W <= 6);
    /// Number of HyperLogLog registers
    const M: usize = 1 << P;
    /// Sparse precision
    const SP: usize = 25;
    /// Dense representation slice length
    const DENSE_LEN: usize = Self::M * W / 32 + 3;

    /// Creates new instance of `CardinalityEstimator`
    #[inline]
    pub fn new() -> Self {
        // compile time check of params
        _ = Self::VALID_PARAMS;

        Self {
            // Start with empty small representation
            data: SMALL_MASK,
            _phantom_hasher: PhantomData,
        }
    }

    /// Creates new instance of `CardinalityEstimator` from given `hash`
    #[inline]
    pub fn from_hash(hash: u64) -> Self {
        let mut estimator = Self::new();
        estimator.insert_hash(hash);
        estimator
    }

    /// Insert hash into `CardinalityEstimator`
    #[inline]
    pub fn insert_hash(&mut self, hash: u64) {
        let h = Self::encode_hash(hash);
        self.insert_encoded_hash(h);
    }

    /// Insert a hashable item into `CardinalityEstimator`
    #[inline]
    pub fn insert<T: Hash>(&mut self, item: T) {
        let mut hasher = H::default();
        item.hash(&mut hasher);
        self.insert_hash(hasher.finish());
    }

    /// Insert encoded hash into `CardinalityEstimator`
    #[inline]
    fn insert_encoded_hash(&mut self, h: u32) {
        // Skip inserting zero hash (useful to simplify merges)
        if h == 0 {
            return;
        }
        match (self.is_small(), self.is_sparse()) {
            (true, false) => self.insert_into_small(h),
            (false, true) => self.insert_into_sparse(h),
            (_, _) => Self::insert_into_dense(self.as_mut_slice(), h),
        }
    }

    /// Return cardinality estimate
    #[inline]
    pub fn estimate(&self) -> usize {
        match (self.is_small(), self.is_sparse()) {
            (true, false) => self.estimate_small(),
            (false, true) => self.estimate_sparse(),
            (_, _) => self.estimate_dense(),
        }
    }

    /// Merge cardinality estimators
    #[inline]
    pub fn merge(&mut self, rhs: &Self) {
        match (self.is_small(), rhs.is_small()) {
            (_, true) => {
                // when `rhs` has small representation - just insert 2 hashes into `self`
                self.insert_encoded_hash(rhs.small_h1());
                self.insert_encoded_hash(rhs.small_h2());
            }
            (true, false) => {
                // when `self` has small representation - save 2 hashes into variables,
                // make copy of `rhs` slice into `self` and then insert saved hashes
                let h1 = self.small_h1();
                let h2 = self.small_h2();
                let rhs_data = Vec::from(rhs.as_slice());
                self.replace_data(rhs_data);
                self.insert_encoded_hash(h1);
                self.insert_encoded_hash(h2);
            }
            (false, false) => match (self.is_sparse(), rhs.is_sparse()) {
                (_, true) => {
                    // when `rhs` has sparse representation - just insert its hashes into `self`
                    let rhs_data = rhs.as_slice();
                    let rhs_len = rhs_data[0] as usize;
                    rhs_data[1..rhs_len + 1]
                        .iter()
                        .for_each(|h| self.insert_encoded_hash(*h));
                }
                (true, false) => {
                    // when `self` has sparse representation - save its hashes into vector,
                    // make copy of `rhs` slice into `self` and then insert saved hashes
                    let lhs_data = self.as_slice();
                    let lhs_len = lhs_data[0] as usize;
                    let lhs_hashes = Vec::from(&lhs_data[1..lhs_len + 1]);
                    let rhs_data = rhs.as_slice();
                    self.replace_data(Vec::from(rhs_data));
                    lhs_hashes.iter().for_each(|h| self.insert_encoded_hash(*h));
                }
                (false, false) => {
                    // when both estimators have dense HyperLogLog representation
                    let lhs_data = self.as_mut_slice();
                    let rhs_data = rhs.as_slice();
                    Self::merge_dense(lhs_data, rhs_data);
                }
            },
        }
    }

    /// Return whether small representation is used
    #[inline]
    pub(crate) fn is_small(&self) -> bool {
        self.data & SMALL_MASK != 0
    }

    /// Return whether sparse representation is used
    #[inline]
    fn is_sparse(&self) -> bool {
        self.data & SLICE_LEN_MASK != 0
    }

    /// Return 1-st encoded hash assuming small representation
    #[inline]
    fn small_h1(&self) -> u32 {
        (self.data & SMALL_1_MASK) as u32
    }

    /// Return 2-nd encoded hash assuming small representation
    #[inline]
    fn small_h2(&self) -> u32 {
        ((self.data & SMALL_2_MASK) >> 31) as u32
    }

    /// Insert encoded hash into small representation
    /// with potential upgrade to sparse representation
    #[inline]
    fn insert_into_small(&mut self, h: u32) {
        // Retrieve 1-st encoded hash
        let h1 = self.small_h1();
        if h1 == 0 {
            self.data |= h as usize;
            return;
        }
        if h1 == h {
            return;
        }
        // Retrieve 2-nd encoded hash
        let h2 = self.small_h2();
        if h2 != 0 {
            if h2 == h {
                return;
            }
            // 2-nd hash occupied -> upgrade to sparse representation
            self.replace_data(vec![3, h1, h2, h, 0]);
            return;
        }
        if h & 1 == 0 {
            // 2-nd hash fits into 31 bits (99.9%+ cases)
            self.data |= ((h as usize) >> 1) << 32;
            return;
        }
        if h1 & 1 == 0 {
            // 1-st and 2-nd hash can be swapped as 1-st hash fits into 31 bits
            self.data = SMALL_MASK | (((h1 as usize) >> 1) << 32) | (h as usize);
            return;
        }
        // neither 1-st or 2-nd hash fit into 31 bits -> upgrade to sparse representation.
        // this is very rare scenario with probability of 1 in 2^(SP-P)
        self.replace_data(vec![2, h1, h, 0, 0]);
    }

    /// Insert encoded hash into sparse representation
    #[inline]
    fn insert_into_sparse(&mut self, h: u32) {
        let data = self.as_mut_slice();

        let len = data[0] as usize;
        let found = if data.len() == 5 {
            contains_fixed_vectorized::<4>(data[1..5].try_into().unwrap(), h)
        } else {
            // calculate rounded up slice length for efficient look up in batches
            let clen = 8 * (len / 8 + usize::from(len % 8 != 0));
            contains_vectorized::<8>(&data[1..clen + 1], h)
        };

        if found {
            return;
        }

        if len < data.len() - 1 {
            data[len + 1] = h;
            data[0] += 1;
            return;
        }

        let new_data = Self::grow_data(&data[1..]);
        self.replace_data(new_data);
        self.insert_encoded_hash(h);
    }

    /// Grow data by either doubling sparse representation or switching to dense representation
    #[inline]
    fn grow_data(old_data: &[u32]) -> Vec<u32> {
        let old_len = old_data.len();
        let new_len = old_len * 2 + 1;
        if new_len >= Self::DENSE_LEN {
            return Self::sparse_to_dense(old_data);
        }

        let mut new_data = vec![0u32; new_len];
        new_data[0] = old_len as u32;
        new_data[1..old_len + 1].copy_from_slice(old_data);

        new_data
    }

    /// Replace any representation with new data while dropping old data
    #[inline]
    pub(crate) fn replace_data(&mut self, data: Vec<u32>) {
        if !self.is_small() {
            drop(unsafe { Box::from_raw(self.as_mut_slice()) });
        }
        self.data = (SLICE_PTR_MASK & (data.as_ptr() as usize)) | self.encoded_slice_len(&data);
        std::mem::forget(data);
    }

    /// Convert sparse representation into dense representation
    #[inline]
    fn sparse_to_dense(sparse_data: &[u32]) -> Vec<u32> {
        let mut data = vec![0u32; Self::DENSE_LEN];
        data[0] = Self::M as u32;
        data[1] = (Self::M as f32).to_bits();
        sparse_data[1..]
            .iter()
            .for_each(|h| Self::insert_into_dense(data.as_mut_slice(), *h));
        data
    }

    /// Compute the sparse encoding of the given hash
    #[inline]
    fn encode_hash(hash: u64) -> u32 {
        let idx = bextr64(hash, 64 - Self::SP, Self::SP) as u32;
        if bextr64(hash, 64 - Self::SP, Self::SP - P) == 0 {
            let tmp = (bextr64(hash, 0, 64 - Self::SP) << Self::SP) | ((1 << Self::SP) - 1);
            let zeros = tmp.leading_zeros() + 1;
            return (idx << 7) | (zeros << 1) | 1;
        }
        idx << 1
    }

    /// Return normal index and rank from encoded sparse hash
    #[inline]
    fn decode_hash(h: u32) -> (u32, u32) {
        if h & 1 == 1 {
            let idx = bextr32(h, 32 - P, P);
            let rank = bextr32(h, 1, W) + (Self::SP - P) as u32;
            (idx, rank)
        } else {
            let idx = bextr32(h, Self::SP - P + 1, P);
            let rank = ((h << (32 - Self::SP + P - 1)) as u64).leading_zeros() - 31;
            (idx, rank)
        }
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

    /// Return size of underlying slice of sparse or dense representation
    /// based on encoded length stored in `self.data`
    #[inline]
    fn slice_len(&self) -> usize {
        if self.is_sparse() {
            (1 << (((self.data & SLICE_LEN_MASK) >> SLICE_LEN_OFFSET) + 1)) + 1
        } else {
            Self::DENSE_LEN
        }
    }

    /// Return encoded slice len based on specified slice length
    #[inline]
    fn encoded_slice_len(&self, data: &[u32]) -> usize {
        if data.len() < Self::DENSE_LEN {
            ((data.len() - 1).trailing_zeros() as usize - 1) << SLICE_LEN_OFFSET
        } else {
            0
        }
    }

    /// Return mutable underlying slice of sparse or dense representation
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u32] {
        let ptr = (self.data & SLICE_PTR_MASK) as *mut u32;
        let len = self.slice_len();
        unsafe { slice::from_raw_parts_mut(ptr, len) }
    }

    /// Return underlying slice of sparse or dense representation
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u32] {
        let ptr = (self.data & SLICE_PTR_MASK) as *const u32;
        let len = self.slice_len();
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    /// Return cardinality estimate of sparse representation
    #[inline]
    fn estimate_sparse(&self) -> usize {
        let data = self.as_slice();
        data[0] as usize
    }

    /// Insert encoded hash into dense representation
    #[inline]
    fn insert_into_dense(data: &mut [u32], h: u32) {
        let (idx, new_rank) = Self::decode_hash(h);
        let old_rank = get_register::<W>(data, idx);
        if new_rank > old_rank {
            set_register::<W>(data, idx, old_rank, new_rank);
        }
    }

    /// Merge two dense HyperLogLog representations
    #[inline]
    fn merge_dense(lhs: &mut [u32], rhs: &[u32]) {
        for idx in 0..Self::M as u32 {
            let lhs_rank = get_register::<W>(lhs, idx);
            let rhs_rank = get_register::<W>(rhs, idx);
            if rhs_rank > lhs_rank {
                set_register::<W>(lhs, idx, lhs_rank, rhs_rank);
            }
        }
    }

    /// Return cardinality estimate of dense representation
    #[inline]
    fn estimate_dense(&self) -> usize {
        let data = self.as_slice();
        let zeros = data[0];
        let sum = f32::from_bits(data[1]) as f64;
        let estimate = alpha(Self::M) * ((Self::M * (Self::M - zeros as usize)) as f64)
            / (sum + beta_horner(zeros as f64, P));
        (estimate + 0.5) as usize
    }

    /// Return memory size of `CardinalityEstimator`
    pub fn size_of(&self) -> usize {
        match (self.is_small(), self.is_sparse()) {
            (true, false) => std::mem::size_of::<Self>(),
            (_, _) => {
                std::mem::size_of::<Self>() + std::mem::size_of::<u32>() * self.as_slice().len()
            }
        }
    }
}

impl<const P: usize, const W: usize> Clone for CardinalityEstimator<P, W> {
    /// Clone `CardinalityEstimator`
    fn clone(&self) -> Self {
        match self.is_small() {
            true => Self {
                data: self.data,
                _phantom_hasher: PhantomData,
            },
            false => {
                let mut estimator = Self::new();
                estimator.merge(self);
                estimator
            }
        }
    }
}

impl<const P: usize, const W: usize, H: Default + Hasher> Drop for CardinalityEstimator<P, W, H> {
    /// Free memory occupied by `CardinalityEstimator`
    fn drop(&mut self) {
        if !self.is_small() {
            drop(unsafe { Box::from_raw(self.as_mut_slice()) });
        }
    }
}

impl<const P: usize, const W: usize, H: Default + Hasher> PartialEq
    for CardinalityEstimator<P, W, H>
{
    /// Compare cardinality estimators
    fn eq(&self, rhs: &Self) -> bool {
        if self.is_small() {
            self.data == rhs.data
        } else {
            self.as_slice() == rhs.as_slice()
        }
    }
}

/// Vectorized linear array search benefiting from SIMD instructions (e.g. AVX2).
///
/// Input slice length assumed to be divisible by `N` to perform efficient
/// batch comparisons of slice elements to provided value `v`.
///
/// Assembly output: https://godbolt.org/z/eb8Kob9fa
/// Background reading: https://tinyurl.com/2e4srh2d
fn contains_vectorized<const N: usize>(a: &[u32], v: u32) -> bool {
    a.chunks_exact(N)
        .any(|chunk| contains_fixed_vectorized::<N>(chunk.try_into().unwrap(), v))
}

/// Vectorized linear fixed array search
#[inline]
fn contains_fixed_vectorized<const N: usize>(a: [u32; N], v: u32) -> bool {
    let mut res = false;
    for x in a {
        res |= x == v
    }
    res
}

/// Parameter for bias correction
#[inline]
fn alpha(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / (m as f64)),
    }
}

/// Extract `length` bits from a bit starting at `start` from `u64` integer
#[inline]
fn bextr64(v: u64, start: usize, length: usize) -> u64 {
    (v >> start) & ((1 << length) - 1)
}

/// Extract `length` bits from a bit starting at `start` from `u32` integer
#[inline]
fn bextr32(v: u32, start: usize, length: usize) -> u32 {
    (v >> start) & ((1 << length) - 1)
}

/// Get HyperLogLog `idx` register
#[inline]
fn get_register<const W: usize>(data: &[u32], idx: u32) -> u32 {
    let bit_idx = (idx as usize) * W;
    let u32_idx = (bit_idx >> 5) + 2;
    let bit_offset = bit_idx & 31;
    let bits_left = 32 - bit_offset;
    if bits_left >= W {
        // register rank fits into single `u32`
        bextr32(data[u32_idx], bit_offset, W)
    } else {
        // register rank spread across two `u32`
        bextr32(data[u32_idx], bit_offset, bits_left)
            | bextr32(data[u32_idx + 1], 0, W - bits_left) << bits_left
    }
}

/// Set HyperLogLog `idx` register to new value `rank`
#[inline]
fn set_register<const W: usize>(data: &mut [u32], idx: u32, old_rank: u32, new_rank: u32) {
    let bit_idx = (idx as usize) * W;
    let u32_idx = (bit_idx / 32) + 2;
    let bit_pos = bit_idx % 32;

    let bits = unsafe { data.get_unchecked_mut(u32_idx..u32_idx + 2) };
    let bits_1 = W.min(32 - bit_pos);
    let bits_2 = W - bits_1;
    let mask_1 = u32::MAX >> (32 - bits_1);
    let mask_2 = (1u32 << bits_2) - 1;

    // Unconditionally update two `u32` elements based on `new_rank` bits and masks
    bits[0] &= !(mask_1 << bit_pos);
    bits[0] |= (new_rank & mask_1) << bit_pos;
    bits[1] &= !mask_2;
    bits[1] |= (new_rank >> bits_1) & mask_2;

    // Update HyperLogLog's number of zero registers and harmonic sum
    data[0] -= (old_rank == 0) as u32 & (data[0] > 0) as u32;

    let mut sum = f32::from_bits(data[1]);
    sum -= 1.0 / ((1u64 << (old_rank as u64)) as f32);
    sum += 1.0 / ((1u64 << (new_rank as u64)) as f32);
    data[1] = sum.to_bits();
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case(0 => (8, 0.0))]
    #[test_case(1 => (8, 0.0))]
    #[test_case(2 => (8, 0.0))]
    #[test_case(3 => (28, 0.0))]
    #[test_case(4 => (28, 0.0))]
    #[test_case(8 => (44, 0.0))]
    #[test_case(16 => (76, 0.0))]
    #[test_case(128 => (524, 0.0))]
    #[test_case(256 => (660, 0.01428608717942597))]
    #[test_case(512 => (660, 0.013023788120268947))]
    #[test_case(1024 => (660, 0.014517720556272953))]
    #[test_case(10_000 => (660, 0.02614431358700056))]
    #[test_case(100_000 => (660, 0.016867961360547964))]
    fn test_estimator_p10_w5(n: usize) -> (usize, f64) {
        evaluate_cardinality_estimator(CardinalityEstimator::<10, 5>::new(), n)
    }

    #[test_case(0 => (8, 0.0))]
    #[test_case(1 => (8, 0.0))]
    #[test_case(2 => (8, 0.0))]
    #[test_case(3 => (28, 0.0))]
    #[test_case(4 => (28, 0.0))]
    #[test_case(8 => (44, 0.0))]
    #[test_case(16 => (76, 0.0))]
    #[test_case(256 => (1036, 0.0))]
    #[test_case(512 => (2060, 0.0))]
    #[test_case(1024 => (3092, 0.002375373334168125))]
    #[test_case(4096 => (3092, 0.003105140671097913))]
    #[test_case(10_000 => (3092, 0.0051820599485679466))]
    #[test_case(100_000 => (3092, 0.011020222347468332))]
    fn test_estimator_p12_w6(n: usize) -> (usize, f64) {
        evaluate_cardinality_estimator(CardinalityEstimator::<12, 6>::new(), n)
    }

    #[test_case(0 => (8, 0.0))]
    #[test_case(1 => (8, 0.0))]
    #[test_case(2 => (8, 0.0))]
    #[test_case(3 => (28, 0.0))]
    #[test_case(4 => (28, 0.0))]
    #[test_case(8 => (44, 0.0))]
    #[test_case(16 => (76, 0.0))]
    #[test_case(256 => (1036, 0.0))]
    #[test_case(512 => (2060, 0.0))]
    #[test_case(1024 => (4108, 0.0))]
    #[test_case(4096 => (16396, 0.0))]
    #[test_case(8192 => (32780, 5.3602457823617544e-6))]
    #[test_case(10_000 => (65548, 2.4330815282868074e-5))]
    fn test_estimator_p18_w6(n: usize) -> (usize, f64) {
        evaluate_cardinality_estimator(CardinalityEstimator::<18, 6>::new(), n)
    }

    fn evaluate_cardinality_estimator<const P: usize, const W: usize>(
        mut e: CardinalityEstimator<P, W>,
        n: usize,
    ) -> (usize, f64) {
        let mut total_relative_error: f64 = 0.0;
        for i in 0..n {
            e.insert(i);
            let estimate = e.estimate() as f64;
            let actual = (i + 1) as f64;
            let error = estimate - actual;
            let relative_error = error.abs() / actual;
            total_relative_error += relative_error;
        }

        let avg_relative_error = total_relative_error / ((n + 1) as f64);
        let size = e.size_of();

        (size, avg_relative_error)
    }

    // cases with error = 0%
    #[test_case(0, 0 => 0)]
    #[test_case(0, 1 => 1)]
    #[test_case(1, 0 => 1)]
    #[test_case(1, 1 => 2)]
    #[test_case(1, 2 => 3)]
    #[test_case(2, 1 => 3)]
    #[test_case(2, 2 => 4)]
    #[test_case(2, 3 => 5)]
    #[test_case(2, 4 => 6)]
    #[test_case(4, 2 => 6)]
    #[test_case(3, 2 => 5)]
    #[test_case(3, 3 => 6)]
    #[test_case(3, 4 => 7)]
    #[test_case(4, 3 => 7)]
    #[test_case(4, 4 => 8)]
    #[test_case(4, 8 => 12)]
    #[test_case(8, 4 => 12)]
    #[test_case(4, 508 => 512)]
    #[test_case(508, 4 => 512)]
    #[test_case(511, 1 => 512)]
    #[test_case(1, 511 => 512)]
    #[test_case(512, 0 => 512)]
    #[test_case(0, 512 => 512)]
    // cases with error > 0%
    #[test_case(512, 1 => 508)]
    #[test_case(1, 512 => 520)]
    #[test_case(512, 512 => 1031)]
    #[test_case(513, 513 => 1032)]
    #[test_case(10000, 0 => 10183)]
    #[test_case(0, 10000 => 9882)]
    #[test_case(4, 10000 => 9892)]
    #[test_case(512, 10000 => 10455)]
    #[test_case(10000, 10000 => 19569)]
    fn test_merge(lhs_n: usize, rhs_n: usize) -> usize {
        let mut lhs = CardinalityEstimator::<12, 6>::new();
        let mut buf = [0, 0, 0, 0, 0, 0, 0, 0, 1];
        for i in 0..lhs_n {
            buf[..8].copy_from_slice(&i.to_le_bytes());
            lhs.insert(buf);
        }

        let mut rhs = CardinalityEstimator::<12, 6>::new();
        let mut buf = [0, 0, 0, 0, 0, 0, 0, 0, 2];
        for i in 0..rhs_n {
            buf[..8].copy_from_slice(&i.to_le_bytes());
            rhs.insert(buf);
        }

        lhs.merge(&rhs);
        lhs.estimate()
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
