#![no_main]

use cardinality_estimator::estimator::CardinalityEstimator;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _: Result<CardinalityEstimator<str>, _> = serde_json::from_slice(data);
});
