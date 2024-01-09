//! ## Small representation
//! Allows to estimate cardinality in [0..2] range and uses only 8 bytes of memory.
//!
//! The `data` format of small representation:
//! - 0..1 bits     - store representation type (bits are set to `00`)
//! - 2..33 bits    - store 31-bit encoded hash
//! - 34..63 bits   - store 31-bit encoded hash

/// Mask used for extracting hashes stored in small representation (31 bits)
const SMALL_MASK: usize = 0x0000_0000_7fff_ffff;

/// Small representation container
pub(crate) struct Small(usize);

impl Small {
    /// Return cardinality estimate of `Small` representation
    #[inline]
    pub(crate) fn estimate(&self) -> usize {
        match (self.h1(), self.h2()) {
            (0, 0) => 0,
            (_, 0) => 1,
            (_, _) => 2,
        }
    }

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

    /// Return memory size of `Small` representation
    pub(crate) fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl From<usize> for Small {
    /// Create new instance of `Small` from given `data`
    #[inline]
    fn from(data: usize) -> Self {
        Self(data)
    }
}

impl From<Small> for usize {
    /// Convert instance of `Small` back to usize
    #[inline]
    fn from(value: Small) -> Self {
        value.0
    }
}
