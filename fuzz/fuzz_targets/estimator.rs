#![no_main]

use cardinality_estimator::estimator::{CardinalityEstimator, CardinalityEstimatorTrait};
use libfuzzer_sys::fuzz_target;
use wyhash::wyhash;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let split_index = wyhash(data, 0) as usize % data.len();
    let (first_half, second_half) = data.split_at(split_index);

    let mut estimator1 = CardinalityEstimator::<&[u8]>::new();
    for chunk in first_half.chunks(4) {
        estimator1.insert(&chunk);
        assert!(estimator1.estimate() > 0);
        assert!(estimator1.size_of() > 0);

    }

    let mut estimator2 = CardinalityEstimator::<&[u8]>::new();
    for chunk in second_half.chunks(4) {
        estimator2.insert(&chunk);
        assert!(estimator2.estimate() > 0);
        assert!(estimator2.size_of() > 0);
    }

    estimator1.merge(&estimator2);
});
