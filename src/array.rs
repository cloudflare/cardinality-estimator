//! ## Array representation
//! Allows to estimate medium cardinality in [3..MAX_CAPACITY] range.
//!
//! The `data` format of array representation:
//! - 0..1 bits     - store representation type (bits are set to `01`)
//! - 2..55 bits    - store pointer to `u32` slice (on `x86_64 systems only 48-bits are needed).
//! - 56..63 bits   - store number of items `N` stored in array
//!
//! Slice encoding:
//! - data[0..N]    - store `N` encoded hashes
//! - data[N..]     - store zeros used for future hashes

use std::mem::{size_of, size_of_val};
use std::slice;

/// Maximum number of elements stored in array representation
const MAX_CAPACITY: usize = 128;
/// Bit offset of the array's length
const LEN_OFFSET: usize = 56;
/// Mask used for accessing heap allocated data stored at the pointer in `data` field.
const PTR_MASK: usize = ((1 << LEN_OFFSET) - 1) & !3;

/// Array representation container
#[derive(Debug)]
pub(crate) struct Array<'a> {
    /// Number of items stored in the array
    len: usize,
    /// Capacity of the array
    cap: usize,
    /// Array of items
    arr: &'a mut [u32],
}

impl<'a> Array<'a> {
    /// Return cardinality estimate of `Array` representation
    #[inline]
    pub(crate) fn estimate(&self) -> usize {
        self.len
    }

    /// Insert encoded hash into `Array` representation
    /// Returns true on success, false otherwise.
    #[inline]
    pub(crate) fn insert(&mut self, h: u32) -> bool {
        let found = if self.cap == 4 {
            contains_fixed_vectorized::<4>(self.arr.try_into().unwrap(), h)
        } else if self.cap == 8 {
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

        if self.len < self.arr.len() {
            // if there are available slots in current array - append to it
            self.arr[self.len] = h;
            self.len += 1;
            return true;
        }

        if self.cap < MAX_CAPACITY {
            // double array capacity up to `MAX_CAPACITY`
            let new_arr = Self::from_vec(vec![0; self.cap * 2], self.len + 1);
            new_arr.arr[..self.len].copy_from_slice(self.arr);
            new_arr.arr[self.len] = h;
            self.destroy();
            *self = new_arr;
            return true;
        };

        false
    }

    /// Return items stored within `Array` representation
    #[inline]
    pub(crate) fn items(&self) -> &[u32] {
        &self.arr[..self.len]
    }

    /// Return memory size of `Array` representation
    #[inline]
    pub(crate) fn size_of(&self) -> usize {
        size_of::<usize>() + size_of_val(self.arr)
    }

    /// Create new instance of `Array` representation from vector
    #[inline]
    pub(crate) fn from_vec(mut arr: Vec<u32>, len: usize) -> Array<'a> {
        let cap = arr.len();
        let ptr = arr.as_mut_ptr();
        std::mem::forget(arr);
        // SAFETY: valid pointer from vector being used to create slice reference
        let arr = unsafe { slice::from_raw_parts_mut(ptr, cap) };
        Self { len, cap, arr }
    }

    /// Free memory occupied by the `Array` representation
    #[inline]
    pub(crate) fn destroy(&mut self) {
        // SAFETY: caller of this method must ensure that `self.arr` holds valid slice elements.
        drop(unsafe { Box::from_raw(self.arr) });
    }
}

impl From<usize> for Array<'_> {
    /// Create new instance of `Array` from given `data`
    #[inline]
    fn from(data: usize) -> Self {
        let ptr = (data & PTR_MASK) as *mut u32;
        let len = data >> LEN_OFFSET;
        let cap = len.next_power_of_two();
        // SAFETY: caller of this method must ensure that `data` contains valid slice pointer.
        let arr = unsafe { slice::from_raw_parts_mut(ptr, cap) };
        Self { len, cap, arr }
    }
}

impl From<Array<'_>> for usize {
    /// Convert instance of `Array` back to usize
    #[inline]
    fn from(v: Array) -> Self {
        (v.len << LEN_OFFSET) | (PTR_MASK & v.arr.as_ptr() as usize) | 1
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
