//! # Serde module for CardinalityEstimator
//!
//! This module provides serde-based (serialization and deserialization) features for
//! `CardinalityEstimator`. It uses `serde`'s custom serialization and deserialization mechanisms.
//!
//! `CardinalityEstimator` has a usize field, `data`, and an optional `Vec<u32>` hidden behind a
//! pointer within `data`. During serialization, these fields are converted into a tuple:
//! `(data, Option<Vec<u32>>)`.
//!
//! During deserialization, the tuple is converted back into the `CardinalityEstimator` struct,
//! handling the case where the `Vec<u32>` may be `None` (indicating a "small" estimator).
//!
//! This allows `CardinalityEstimator` to be easily serialized/deserialized, for storage,
//! transmission, and reconstruction.
//!
//! Refer to the serde documentation for more details on custom serialization and deserialization:
//! - [Serialization](https://serde.rs/impl-serialize.html)
//! - [Deserialization](https://serde.rs/impl-deserialize.html)
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use serde::de::Error;
use serde::ser::SerializeTuple;
use serde::{Deserialize, Serialize};

use crate::estimator::CardinalityEstimator;
use crate::representation::Representation;

impl<T, H, const P: usize, const W: usize> Serialize for CardinalityEstimator<T, H, P, W>
where
    T: Hash + ?Sized,
    H: Hasher + Default,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Begin a new serialized tuple with two elements.
        let mut tup = serializer.serialize_tuple(2)?;

        // The first element is the data field of the estimator.
        tup.serialize_element(&self.data)?;
        match self.representation() {
            Representation::Small(_) => {
                // If the estimator is small, the second element is a None value. This indicates that
                // the estimator is using the small data optimization and has no separate slice data.
                tup.serialize_element(&None::<Vec<u32>>)?;
            }
            Representation::Array(arr) => {
                // If the estimator is slice, the second element is a option containing slice data.
                tup.serialize_element(&Some(arr.deref()))?;
            }
            Representation::Hll(hll) => {
                // If the estimator is HLL, the second element is a option containing HLL data.
                tup.serialize_element(&Some(hll.data))?;
            }
        }

        // Finalize the tuple.
        tup.end()
    }
}

impl<'de, T, H, const P: usize, const W: usize> Deserialize<'de>
    for CardinalityEstimator<T, H, P, W>
where
    T: Hash + ?Sized,
    H: Hasher + Default,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialize the tuple that was serialized by the serialize method. The first element
        // of the tuple is the data field of the estimator, and the second element is an Option
        // that contains the array data if the estimator is not small.
        let (data, opt_vec): (usize, Option<Vec<u32>>) = Deserialize::deserialize(deserializer)?;
        Representation::try_from(data, opt_vec).map_err(|e| Error::custom(format!("{:?}", e)))
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case(0; "empty set")]
    #[test_case(1; "single element")]
    #[test_case(2; "two distinct elements")]
    #[test_case(100; "hundred distinct elements")]
    #[test_case(10000; "ten thousand distinct elements")]
    fn test_serde(n: usize) {
        let mut original_estimator = CardinalityEstimator::<str>::new();

        for i in 0..n {
            let item = &format!("item{}", i);
            original_estimator.insert(&item);
        }

        let serialized = serde_json::to_string(&original_estimator).expect("serialization failed");
        assert!(
            !serialized.is_empty(),
            "serialized string should not be empty"
        );

        let deserialized_estimator: CardinalityEstimator<str> =
            serde_json::from_str(&serialized).expect("deserialization failed");

        assert_eq!(
            original_estimator.representation(),
            deserialized_estimator.representation()
        );

        // run each case with postcard serialization as well

        let postcard_serialized =
            postcard::to_allocvec(&original_estimator).expect("serialization failed");
        assert!(
            !postcard_serialized.is_empty(),
            "postcard_serialized bytes should not be empty"
        );

        let postcard_estimator: CardinalityEstimator<str> =
            postcard::from_bytes(&postcard_serialized).expect("deserialization failed");

        assert_eq!(
            original_estimator.representation(),
            postcard_estimator.representation()
        );
    }

    #[test]
    fn test_deserialize_invalid_json() {
        let invalid_json = "{ invalid_json_string }";
        let result: Result<CardinalityEstimator<str>, _> = serde_json::from_str(invalid_json);

        assert!(
            result.is_err(),
            "Deserialization should fail for invalid JSON"
        );
    }

    #[test_case("[12345,null]".as_bytes(); "case 1")]
    #[test_case(&[91, 49, 55, 44, 13, 10, 91, 13, 93, 93]; "case 2")]
    #[test_case(&[91, 51, 44, 10, 110, 117, 108, 108, 93, 122]; "case 3")]
    #[test_case(&[91, 51, 44, 10, 110, 117, 108, 108, 93]; "case 4")]
    fn test_failed_deserialization(input: &[u8]) {
        let result: Result<CardinalityEstimator<str>, _> = serde_json::from_slice(input);
        assert!(result.is_err());

        let result: Result<CardinalityEstimator<str>, _> = postcard::from_bytes(input);
        assert!(result.is_err());
    }
}
