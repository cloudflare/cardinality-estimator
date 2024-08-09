//! ## HyperLogLog representation
//! Allows to estimate large cardinality in `[N..]` range, where `N` is based on `P` and `W`.
//! This representation uses modified HyperLogLog++ with `M` registers of `W` width.
//!
//! [Original HyperLogLog++ paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40671.pdf)
//!
//! The `data` format of HyperLogLog representation:
//! - 0..1 bits     - store representation type (bits are set to `11`)
//! - 2..63 bits    - store pointer to `u32` slice (on `x86_64 systems only 48-bits are needed).
//!
//! Slice encoding:
//! - data[0]       - stores number of HyperLogLog registers set to 0.
//! - data[1]       - stores harmonic sum of HyperLogLog registers (`f32` transmuted into `u32`).
//! - data[2..]     - stores register ranks using `W` bits per each register.

use std::fmt::{Debug, Formatter};
use std::mem::{size_of, size_of_val};
use std::slice;

use crate::representation::RepresentationTrait;

/// Mask used for accessing heap allocated data stored at the pointer in `data` field.
const PTR_MASK: usize = !3;

#[derive(PartialEq)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg, mem_dbg::MemSize))]
pub(crate) struct HyperLogLog<'a, const P: usize = 12, const W: usize = 6> {
    pub(crate) data: &'a mut [u32],
}

impl<'a, const P: usize, const W: usize> HyperLogLog<'a, P, W> {
    /// Number of HyperLogLog registers
    const M: usize = 1 << P;
    /// HyperLogLog representation `u32` slice length based on #registers, stored zero registers, harmonic sum, and
    /// one extra element for branchless register updates (see `set_register` for more details).
    pub(crate) const HLL_SLICE_LEN: usize = Self::M * W / 32 + 3;

    /// Create new instance of `HyperLogLog` representation from items
    #[inline]
    pub(crate) fn new(items: &[u32]) -> Self {
        let mut hll_data = vec![0u32; Self::HLL_SLICE_LEN];
        let data = (PTR_MASK & hll_data.as_mut_ptr() as usize) | 3;
        std::mem::forget(hll_data);
        let mut hll = Self::from(data);

        hll.data[0] = Self::M as u32;
        hll.data[1] = (Self::M as f32).to_bits();

        for &h in items.iter() {
            hll.insert_encoded_hash(h);
        }

        hll
    }

    /// Return normal index and rank from encoded sparse hash
    #[inline]
    fn decode_hash(h: u32) -> (u32, u32) {
        let rank = h & ((1 << W) - 1);
        let idx = (h >> W) & ((1 << P) - 1);
        (idx, rank)
    }

    /// Insert encoded hash into HyperLogLog representation
    #[inline]
    fn update_rank(&mut self, idx: u32, new_rank: u32) {
        let old_rank = self.get_register(idx);
        if new_rank > old_rank {
            self.set_register(idx, old_rank, new_rank);
        }
    }

    /// Get HyperLogLog `idx` register
    #[inline]
    fn get_register(&self, idx: u32) -> u32 {
        let bit_idx = (idx as usize) * W;
        let u32_idx = (bit_idx / 32) + 2;
        let bit_pos = bit_idx % 32;
        // SAFETY: `self.data` is always guaranteed to have these elements.
        let bits = unsafe { self.data.get_unchecked(u32_idx..u32_idx + 2) };
        let bits_1 = W.min(32 - bit_pos);
        let bits_2 = W - bits_1;
        let mask_1 = (1 << bits_1) - 1;
        let mask_2 = (1 << bits_2) - 1;

        ((bits[0] >> bit_pos) & mask_1) | ((bits[1] & mask_2) << bits_1)
    }

    /// Set HyperLogLog `idx` register to new value `rank`
    #[inline]
    fn set_register(&mut self, idx: u32, old_rank: u32, new_rank: u32) {
        let bit_idx = (idx as usize) * W;
        let u32_idx = (bit_idx / 32) + 2;
        let bit_pos = bit_idx % 32;
        // SAFETY: `self.data` is always guaranteed to have these elements.
        let bits = unsafe { self.data.get_unchecked_mut(u32_idx..u32_idx + 2) };
        let bits_1 = W.min(32 - bit_pos);
        let bits_2 = W - bits_1;
        let mask_1 = (1 << bits_1) - 1;
        let mask_2 = (1 << bits_2) - 1;

        // Unconditionally update two `u32` elements based on `new_rank` bits and masks
        bits[0] &= !(mask_1 << bit_pos);
        bits[0] |= (new_rank & mask_1) << bit_pos;
        bits[1] &= !mask_2;
        bits[1] |= (new_rank >> bits_1) & mask_2;

        // Update HyperLogLog's number of zero registers and harmonic sum
        // SAFETY: `self.data` is always guaranteed to have 0-th and 1-st elements.
        let zeros_and_sum = unsafe { self.data.get_unchecked_mut(0..2) };
        zeros_and_sum[0] -= (old_rank == 0) as u32 & (zeros_and_sum[0] > 0) as u32;

        let mut sum = f32::from_bits(zeros_and_sum[1]);
        sum -= 1.0 / ((1u64 << (old_rank as u64)) as f32);
        sum += 1.0 / ((1u64 << (new_rank as u64)) as f32);
        zeros_and_sum[1] = sum.to_bits();
    }

    /// Merge two `HyperLogLog` representations.
    #[inline]
    pub(crate) fn merge(&mut self, rhs: &HyperLogLog<P, W>) {
        for idx in 0..Self::M as u32 {
            let lhs_rank = self.get_register(idx);
            let rhs_rank = rhs.get_register(idx);
            if rhs_rank > lhs_rank {
                self.set_register(idx, lhs_rank, rhs_rank);
            }
        }
    }
}

impl<'a, const P: usize, const W: usize> RepresentationTrait for HyperLogLog<'a, P, W> {
    /// Insert encoded hash into `HyperLogLog` representation.
    #[inline]
    fn insert_encoded_hash(&mut self, h: u32) -> usize {
        let (idx, rank) = Self::decode_hash(h);
        self.update_rank(idx, rank);
        self.to_data()
    }

    /// Return cardinality estimate of `HyperLogLog` representation
    #[inline]
    fn estimate(&self) -> usize {
        // SAFETY: `self.data` is always guaranteed to have 0-th and 1-st elements.
        let zeros = unsafe { *self.data.get_unchecked(0) };
        let sum = f32::from_bits(unsafe { *self.data.get_unchecked(1) }) as f64;
        let estimate = alpha(Self::M) * ((Self::M * (Self::M - zeros as usize)) as f64)
            / (sum + beta_horner(zeros as f64, P));
        (estimate + 0.5) as usize
    }

    /// Return memory size of `HyperLogLog`
    #[inline]
    fn size_of(&self) -> usize {
        // The length of the slice, plus the pointer of the slice reference, and the size of the slice itself.
        size_of::<usize>() + size_of::<usize>() + size_of_val(self.data)
    }

    /// Free memory occupied by the `HyperLogLog` representation
    /// SAFETY: caller of this method must ensure that `self.data` holds valid slice elements.
    #[inline]
    unsafe fn drop(&mut self) {
        drop(Box::from_raw(self.data));
    }

    /// Convert `HyperLogLog` representation to `data`
    #[inline]
    fn to_data(&self) -> usize {
        (PTR_MASK & self.data.as_ptr() as usize) | 3
    }
}

impl<const P: usize, const W: usize> From<usize> for HyperLogLog<'_, P, W> {
    /// Create new instance of `HyperLogLog` from given `data`
    #[inline]
    fn from(data: usize) -> Self {
        let ptr = (data & PTR_MASK) as *mut u32;
        // SAFETY: caller of this method must ensure that `data` contains valid slice pointer.
        let data = unsafe { slice::from_raw_parts_mut(ptr, Self::HLL_SLICE_LEN) };
        Self { data }
    }
}

impl<'a, const P: usize, const W: usize> From<Vec<u32>> for HyperLogLog<'a, P, W> {
    /// Create new instance of `HyperLogLog` from given `hll_data`
    #[inline]
    fn from(mut hll_data: Vec<u32>) -> Self {
        let data = (PTR_MASK & hll_data.as_mut_ptr() as usize) | 3;
        std::mem::forget(hll_data);
        Self::from(data)
    }
}

impl<const P: usize, const W: usize> Clone for HyperLogLog<'_, P, W> {
    /// Clone `HyperLogLog` representation
    #[inline]
    fn clone(&self) -> Self {
        let mut hll_data = self.data.to_vec();
        let data = (PTR_MASK & hll_data.as_mut_ptr() as usize) | 3;
        std::mem::forget(hll_data);
        Self::from(data)
    }
}

impl<const P: usize, const W: usize> Debug for HyperLogLog<'_, P, W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string())
    }
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
/// Computes LogLog-Beta estimate bias correction using Horner's method.
///
/// Paper: https://arxiv.org/pdf/1612.02284.pdf
/// Wikipedia: https://en.wikipedia.org/wiki/Horner%27s_method
#[inline]
fn beta_horner(z: f64, precision: usize) -> f64 {
    let beta = BETA[precision - 4];
    let zl = (z + 1.0).ln();
    let mut res = 0.0;
    for i in (1..8).rev() {
        res = res * zl + beta[i];
    }
    res * zl + beta[0] * z
}

/// LogLog-Beta polynomial coefficients for precision in [4..18] range.
const BETA: [[f64; 8]; 15] = [
    // p = 4
    [
        -0.582581413904517,
        -1.93530035756005,
        11.079323758035073,
        -22.131357446444323,
        22.505391846630037,
        -12.000723834917984,
        3.220579408194167,
        -0.342225302271235,
    ],
    // p = 5
    [
        -0.7518999460733967,
        -0.959003007774876,
        5.59973713221416,
        -8.209763699976552,
        6.509125489447204,
        -2.683029373432373,
        0.5612891113138221,
        -0.0463331622196545,
    ],
    // p = 6
    [
        29.825790096961963,
        -31.328708333772592,
        -10.594252303658228,
        -11.572012568909962,
        3.818875437390749,
        -2.416013032853081,
        0.4542208940970826,
        -0.0575155452020420,
    ],
    // p = 7
    [
        2.810292129082006,
        -3.9780498518175995,
        1.3162680041351582,
        -3.92524863358059,
        2.008083575394647,
        -0.7527151937556955,
        0.1265569894242751,
        -0.0109946438726240,
    ],
    // p = 8
    [
        1.0063354488755052,
        -2.005806664051124,
        1.6436974936651412,
        -2.7056080994056617,
        1.392099802442226,
        -0.4647037427218319,
        0.07384282377269775,
        -0.00578554885254223,
    ],
    // p = 9
    [
        -0.09415657458167959,
        -0.7813097592455053,
        1.7151494675071246,
        -1.7371125040651634,
        0.8644150848904892,
        -0.23819027465047218,
        0.03343448400269076,
        -0.00207858528178157,
    ],
    // p = 10
    [
        -0.25935400670790054,
        -0.5259830199980581,
        1.4893303492587684,
        -1.2964271408499357,
        0.6228475621722162,
        -0.1567232677025104,
        0.02054415903878563,
        -0.00112488483925502,
    ],
    // p = 11
    [
        -4.32325553856025e-01,
        -1.08450736399632e-01,
        6.09156550741120e-01,
        -1.65687801845180e-02,
        -7.95829341087617e-02,
        4.71830602102918e-02,
        -7.81372902346934e-03,
        5.84268708489995e-04,
    ],
    // p = 12
    [
        -3.84979202588598e-01,
        1.83162233114364e-01,
        1.30396688841854e-01,
        7.04838927629266e-02,
        -8.95893971464453e-03,
        1.13010036741605e-02,
        -1.94285569591290e-03,
        2.25435774024964e-04,
    ],
    // p = 13
    [
        -0.41655270946462997,
        -0.22146677040685156,
        0.38862131236999947,
        0.4534097974606237,
        -0.36264738324476375,
        0.12304650053558529,
        -0.0170154038455551,
        0.00102750367080838,
    ],
    // p = 14
    [
        -3.71009760230692e-01,
        9.78811941207509e-03,
        1.85796293324165e-01,
        2.03015527328432e-01,
        -1.16710521803686e-01,
        4.31106699492820e-02,
        -5.99583540511831e-03,
        4.49704299509437e-04,
    ],
    // p = 15
    [
        -0.38215145543875273,
        -0.8906940053609084,
        0.3760233577467887,
        0.9933597744068238,
        -0.6557744163831896,
        0.1833234212970361,
        -0.02241529633062872,
        0.00121399789330194,
    ],
    // p = 16
    [
        -0.3733187664375306,
        -1.41704077448123,
        0.40729184796612533,
        1.5615203390658416,
        -0.9924223353428613,
        0.2606468139948309,
        -0.03053811369682807,
        0.00155770210179105,
    ],
    // p = 17
    [
        -0.36775502299404605,
        0.5383142235137797,
        0.7697028927876792,
        0.5500258358645056,
        -0.7457558826114694,
        0.2571183578582195,
        -0.03437902606864149,
        0.00185949146371616,
    ],
    // p = 18
    [
        -0.3647962332596054,
        0.9973041232863503,
        1.5535438623008122,
        1.2593267719802892,
        -1.5332594820911016,
        0.4780104220005659,
        -0.05951025172951174,
        0.00291076804642205,
    ],
];
