use std::hash::{Hash, Hasher};

use enum_dispatch::enum_dispatch;

use crate::array::{Array, MAX_CAPACITY};
use crate::hyperloglog::HyperLogLog;
use crate::representation::RepresentationError::*;
use crate::small::Small;
use crate::CardinalityEstimator;

/// Masks used for storing and retrieving representation type stored in lowest 2 bits of `data` field.
const REPRESENTATION_MASK: usize = 0x0000_0000_0000_0003;
const REPRESENTATION_SMALL: usize = 0x0000_0000_0000_0000;
const REPRESENTATION_ARRAY: usize = 0x0000_0000_0000_0001;
const REPRESENTATION_HLL: usize = 0x0000_0000_0000_0003;

/// Representation types supported by `CardinalityEstimator`
// #[repr(u8)]
#[derive(PartialEq)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg, mem_dbg::MemSize))]
#[enum_dispatch]
pub(crate) enum Representation<'a, const P: usize, const W: usize> {
    Small(Small<P, W>),
    Array(Array<'a, P, W>),
    Hll(HyperLogLog<'a, P, W>),
}

/// Representation trait which must be implemented by all representations.
pub(crate) trait RepresentationTrait {
    fn insert_encoded_hash(&mut self, h: u32) -> usize;
    fn estimate(&self) -> usize;
    fn size_of(&self) -> usize;
    unsafe fn drop(&mut self);
    fn to_data(&self) -> usize;
    fn to_string(&self) -> String {
        format!("estimate: {}, size: {}", self.estimate(), self.size_of())
    }
}

impl<'a, const P: usize, const W: usize> core::fmt::Debug for Representation<'a, P, W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl<'a, const P: usize, const W: usize> RepresentationTrait for Representation<'a, P, W> {
    #[inline]
    fn insert_encoded_hash(&mut self, h: u32) -> usize {
        match self {
            Representation::Small(s) => s.insert_encoded_hash(h) as usize,
            Representation::Array(a) => a.insert_encoded_hash(h) as usize,
            Representation::Hll(hll) => hll.insert_encoded_hash(h) as usize,
        }
    }

    #[inline]
    fn estimate(&self) -> usize {
        match self {
            Representation::Small(s) => s.estimate(),
            Representation::Array(a) => a.estimate(),
            Representation::Hll(hll) => hll.estimate(),
        }
    }

    #[inline]
    fn size_of(&self) -> usize {
        // A priori, we do not know which of the child structs is the largest and therefore
        // composing the bulk of the memory usage by the enum. Therefore, we subtract the size of
        // the variants, and then we sum the size of the enum itself.
        let size_of_variant = match self {
            Representation::Small(s) => s.size_of() - size_of::<Small<P, W>>(),
            Representation::Array(a) => a.size_of() - size_of::<Array<'a, P, W>>(),
            Representation::Hll(hll) => hll.size_of() - size_of::<HyperLogLog<'a, P, W>>(),
        };

        size_of_variant + size_of::<Self>()
    }

    #[inline]
    unsafe fn drop(&mut self) {
        match self {
            Representation::Small(s) => s.drop(),
            Representation::Array(a) => a.drop(),
            Representation::Hll(hll) => hll.drop(),
        }
    }

    #[inline]
    fn to_data(&self) -> usize {
        match self {
            Representation::Small(s) => s.to_data(),
            Representation::Array(a) => a.to_data(),
            Representation::Hll(hll) => hll.to_data(),
        }
    }
}

/// Representation error
#[derive(Debug)]
pub enum RepresentationError {
    InvalidRepresentation,
    SmallRepresentationInvalid,
    ArrayRepresentationInvalid,
    HllRepresentationInvalid,
}

impl<'a, const P: usize, const W: usize> Representation<'a, P, W> {
    /// Returns the representation type of `CardinalityEstimator`.
    ///
    /// This method extracts the representation based on the lowest 2 bits of `data`.
    ///
    /// Valid encodings:
    /// - `0` for `Small` representation
    /// - `1` for `Array` representation
    /// - `3` for `HLL` representation
    ///
    /// If `data` is not encoded as 0, 1, or 3, the function defaults to `Small` with value of 0
    /// as a safe fallback to handle unexpected conditions.
    #[inline]
    pub(crate) fn from_data(data: usize) -> Self {
        match data & REPRESENTATION_MASK {
            REPRESENTATION_SMALL => Representation::Small(Small::from(data)),
            REPRESENTATION_ARRAY => Representation::Array(Array::from(data)),
            REPRESENTATION_HLL => Representation::Hll(HyperLogLog::<P, W>::from(data)),
            _ => Representation::Small(Small::from(0)),
        }
    }

    /// Create new cardinality estimator from data and optional vector
    pub fn try_from<T, H>(
        data: usize,
        opt_vec: Option<Vec<u32>>,
    ) -> Result<CardinalityEstimator<T, H, P, W>, RepresentationError>
    where
        T: Hash + ?Sized,
        H: Hasher + Default,
    {
        let mut estimator = CardinalityEstimator::<T, H, P, W>::new();
        estimator.data = match data & REPRESENTATION_MASK {
            REPRESENTATION_SMALL if opt_vec.is_some() => return Err(SmallRepresentationInvalid),
            REPRESENTATION_SMALL => Small::<P, W>::from(data).to_data(),
            REPRESENTATION_ARRAY => {
                let vec = opt_vec.ok_or(ArrayRepresentationInvalid)?;
                let len = vec.len();
                if len <= 2 || len > MAX_CAPACITY {
                    return Err(ArrayRepresentationInvalid);
                }
                Array::<P, W>::from_vec(vec, len).to_data()
            }
            REPRESENTATION_HLL => {
                let vec = opt_vec.ok_or(HllRepresentationInvalid)?;
                if vec.len() != HyperLogLog::<P, W>::HLL_SLICE_LEN {
                    return Err(HllRepresentationInvalid);
                }
                HyperLogLog::<P, W>::from(vec).to_data()
            }
            _ => return Err(InvalidRepresentation),
        };

        Ok(estimator)
    }
}
