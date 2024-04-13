use enum_dispatch::enum_dispatch;

use crate::array::Array;
use crate::hyperloglog::HyperLogLog;
use crate::small::Small;

/// Mask used for storing and retrieving representation type stored in lowest 2 bits of `data` field.
const REPRESENTATION_MASK: usize = 0x0000_0000_0000_0003;

/// Representation types supported by `CardinalityEstimator`
#[repr(u8)]
#[derive(Debug, PartialEq)]
#[enum_dispatch]
pub(crate) enum Representation<'a, const P: usize, const W: usize> {
    Small(Small<P, W>),
    Array(Array<'a, P, W>),
    HLL(HyperLogLog<'a, P, W>),
}

/// Representation trait which must be implemented by all representations.
#[enum_dispatch(Representation<P, W>)]
pub(crate) trait RepresentationTrait {
    fn insert_encoded_hash(&mut self, h: u32) -> usize;
    fn estimate(&self) -> usize;
    fn size_of(&self) -> usize;
    fn drop(&mut self);
    fn to_data(&self) -> usize;
    fn to_string(&self) -> String {
        format!("estimate: {}, size: {}", self.estimate(), self.size_of())
    }
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
            0 => Representation::Small(Small::from(data)),
            1 => Representation::Array(Array::from(data)),
            3 => Representation::HLL(HyperLogLog::<P, W>::from(data)),
            _ => Representation::Small(Small::from(0)),
        }
    }
}
