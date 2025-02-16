//! ## Small representation
//! Allows to estimate cardinality in [0..2] range and uses only 8 bytes of memory.
//!
//! The `data` format of small representation:
//! - 0..1 bits     - store representation type (bits are set to `00`)
//! - 2..33 bits    - store 31-bit encoded hash
//! - 34..63 bits   - store 31-bit encoded hash

use std::fmt::{Debug, Formatter};

use crate::array::Array;
use crate::representation::{RepresentationTrait, REPRESENTATION_SMALL};

/// Mask used for extracting hashes stored in small representation (31 bits)
const SMALL_MASK: usize = 0x0000_0000_7fff_ffff;

/// Small representation container
#[derive(PartialEq)]
pub(crate) struct Small<const P: usize, const W: usize>(usize);

impl<const P: usize, const W: usize> Small<P, W> {
    /// Insert encoded hash into `Small` representation.
    /// Returns true on success, false otherwise.
    #[inline]
    pub(crate) fn insert(&mut self, h: u32) -> bool {
        let h1 = self.h1();
        if h1 == 0 {
            self.0 |= (h as usize) << 2;
            return true;
        } else if h1 == h {
            return true;
        }

        let h2 = self.h2();
        if h2 == 0 {
            self.0 |= (h as usize) << 33;
            return true;
        } else if h2 == h {
            return true;
        }

        false
    }

    /// Return 1-st encoded hash
    #[inline]
    fn h1(&self) -> u32 {
        ((self.0 >> 2) & SMALL_MASK) as u32
    }

    /// Return 2-nd encoded hash
    #[inline]
    fn h2(&self) -> u32 {
        ((self.0 >> 33) & SMALL_MASK) as u32
    }

    /// Return items stored within `Small` representation
    #[inline]
    pub(crate) fn items(&self) -> [u32; 2] {
        [self.h1(), self.h2()]
    }
}

impl<const P: usize, const W: usize> RepresentationTrait for Small<P, W> {
    /// Insert encoded hash into `Small` representation.
    fn insert_encoded_hash(&mut self, h: u32) -> usize {
        if self.insert(h) {
            self.to_data()
        } else {
            // upgrade from `Small` to `Array` representation
            let items = self.items();
            let arr = Array::<P, W>::from_vec(vec![items[0], items[1], h, 0], 3);
            arr.to_data()
        }
    }

    /// Return cardinality estimate of `Small` representation
    #[inline]
    fn estimate(&self) -> usize {
        match (self.h1(), self.h2()) {
            (0, 0) => 0,
            (_, 0) => 1,
            (_, _) => 2,
        }
    }

    /// Return memory size of `Small` representation
    fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    /// Free memory occupied by the `Small` representation
    #[inline]
    unsafe fn drop(&mut self) {}

    /// Convert `Small` representation to `data`
    #[inline]
    fn to_data(&self) -> usize {
        self.0 | REPRESENTATION_SMALL
    }
}

impl<const P: usize, const W: usize> Debug for Small<P, W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string())
    }
}

impl<const P: usize, const W: usize> From<usize> for Small<P, W> {
    /// Create new instance of `Small` from given `data`
    fn from(data: usize) -> Self {
        Self(data)
    }
}
