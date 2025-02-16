//! ## Array representation
//! Allows to estimate medium cardinality in [3..MAX_CAPACITY] range.
//!
//! The `data` format of array representation:
//! - 0..1 bits     - store representation type (bits are set to `01`)
//! - 2..55 bits    - store pointer to `u32` slice (on `x86_64` systems only 48-bits are needed).
//! - 56..63 bits   - store number of items `N` stored in array
//!
//! Slice encoding:
//! - data[0..N]    - store `N` encoded hashes
//! - data[N..]     - store zeros used for future hashes

use std::fmt::{Debug, Formatter};
use std::mem::{size_of, size_of_val};
use std::ops::Deref;
use std::slice;

use crate::hyperloglog::HyperLogLog;
use crate::representation::{RepresentationTrait, REPRESENTATION_ARRAY};

/// Maximum number of elements stored in array representation
pub(crate) const MAX_CAPACITY: usize = 128;
/// Bit offset of the array's length
const LEN_OFFSET: usize = 56;
/// Mask used for accessing heap allocated data stored at the pointer in `data` field.
const PTR_MASK: usize = ((1 << LEN_OFFSET) - 1) & !3;

/// Array representation container
pub(crate) struct Array<'a, const P: usize, const W: usize> {
    /// Number of items stored in the array
    len: usize,
    /// Array of items. Not all items are used, only first `len` of them.
    /// The length of the slice is actually the current capacity.
    arr: &'a mut [u32],
}

impl<'a, const P: usize, const W: usize> Array<'a, P, W> {
    /// Insert encoded hash into `Array` representation
    /// Returns true on success, false otherwise.
    #[inline]
    pub(crate) fn insert(&mut self, h: u32) -> bool {
        let cap = self.arr.len();
        let found = if cap == 4 {
            contains_fixed_vectorized::<4>(self.arr.try_into().unwrap(), h)
        } else if cap == 8 {
            contains_fixed_vectorized::<8>(self.arr.try_into().unwrap(), h)
        } else {
            // calculate rounded up slice length for efficient look up in batches
            let rlen = 16 * self.len.div_ceil(16);
            // SAFETY: `rlen` guaranteed to be within `self.arr` boundaries
            contains_vectorized::<16>(unsafe { self.arr.get_unchecked(..rlen) }, h)
        };

        if found {
            return true;
        }

        if self.len < cap {
            // if there are available slots in current array - append to it
            self.arr[self.len] = h;
            self.len += 1;
            return true;
        }

        if cap < MAX_CAPACITY {
            // double array capacity up to `MAX_CAPACITY`
            let new_arr = Self::from_vec(vec![0; cap * 2], self.len + 1);
            new_arr.arr[..self.len].copy_from_slice(self.arr);
            new_arr.arr[self.len] = h;
            unsafe { self.drop() };
            *self = new_arr;
            return true;
        };

        false
    }

    /// Create new instance of `Array` representation from vector
    #[inline]
    pub(crate) fn from_vec(mut arr: Vec<u32>, len: usize) -> Array<'a, P, W> {
        let cap = arr.len();
        let ptr = arr.as_mut_ptr();
        std::mem::forget(arr);
        // SAFETY: valid pointer from vector being used to create slice reference
        let arr = unsafe { slice::from_raw_parts_mut(ptr, cap) };
        Self { len, arr }
    }
}

impl<const P: usize, const W: usize> RepresentationTrait for Array<'_, P, W> {
    /// Insert encoded hash into `HyperLogLog` representation.
    #[inline]
    fn insert_encoded_hash(&mut self, h: u32) -> usize {
        if self.insert(h) {
            self.to_data()
        } else {
            // upgrade from `Array` to `HyperLogLog` representation
            let mut hll = HyperLogLog::<P, W>::new(self);
            unsafe { self.drop() };
            hll.insert_encoded_hash(h)
        }
    }

    /// Return cardinality estimate of `Array` representation
    #[inline]
    fn estimate(&self) -> usize {
        self.len
    }

    /// Return memory size of `Array` representation
    #[inline]
    fn size_of(&self) -> usize {
        size_of::<usize>() + size_of_val(self.arr)
    }

    /// Free memory occupied by the `Array` representation
    /// SAFETY: caller of this method must ensure that `self.arr` holds valid slice elements.
    #[inline]
    unsafe fn drop(&mut self) {
        drop(Box::from_raw(self.arr));
    }

    /// Convert `Array` representation to `data`
    #[inline]
    fn to_data(&self) -> usize {
        (self.len << LEN_OFFSET) | (PTR_MASK & self.arr.as_ptr() as usize) | REPRESENTATION_ARRAY
    }
}

impl<const P: usize, const W: usize> Debug for Array<'_, P, W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string())
    }
}

impl<const P: usize, const W: usize> PartialEq for Array<'_, P, W> {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<const P: usize, const W: usize> From<usize> for Array<'_, P, W> {
    /// Create new instance of `Array` from given `data`
    #[inline]
    fn from(data: usize) -> Self {
        let ptr = (data & PTR_MASK) as *mut u32;
        let len = data >> LEN_OFFSET;
        let cap = len.next_power_of_two();
        let arr = unsafe { slice::from_raw_parts_mut(ptr, cap) };
        Self { len, arr }
    }
}

impl<const P: usize, const W: usize> Deref for Array<'_, P, W> {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        &self.arr[..self.len]
    }
}

/// Vectorized linear array search benefiting from SIMD instructions (e.g. AVX2).
///
/// Input slice length assumed to be divisible by `N` to perform efficient
/// batch comparisons of slice elements to provided value `v`.
///
/// Assembly output: https://godbolt.org/z/eb8Kob9fa
/// Background reading: https://tinyurl.com/2e4srh2d
#[inline]
fn contains_vectorized<const N: usize>(a: &[u32], v: u32) -> bool {
    debug_assert_eq!(a.len() % N, 0);
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
