#![no_main]

use serde_json::Value;
use cardinality_estimator::estimator::CardinalityEstimator;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // pretty naiive version, u8s directly into each number position
    let json: serde_json::Value = match data.len() {
        0 => Value::Array(vec![]),
        1 => Value::Array(vec![data[0].into()]),
        _ => Value::Array(vec![data[0].into(),
            Value::Array(data[1..].iter().map(|n| (*n).into()).collect())]),
    };
    if let Ok(mut estimator) = serde_json::from_value::<CardinalityEstimator<usize>>(json) {
        estimator.insert(&1);
        assert!(estimator.estimate() > 0);
    }
});
