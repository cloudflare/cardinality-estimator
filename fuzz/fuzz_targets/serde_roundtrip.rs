#![no_main]

use cardinality_estimator::estimator::CardinalityEstimator;
use libfuzzer_sys::fuzz_target;
use postcard::{to_allocvec, from_bytes};

fuzz_target!(|data: &[u8]| {
    let mut estimator = CardinalityEstimator::<u8>::new();
    for d in data {
        estimator.insert(&d);
    }
    let serialized = to_allocvec(&estimator).unwrap();
    let mut roundtripped: CardinalityEstimator<u8> = from_bytes(&serialized).unwrap();
    roundtripped.insert(&1);
});
