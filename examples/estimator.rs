use cardinality_estimator::CardinalityEstimator;

fn main() {
    let mut estimator1 = CardinalityEstimator::<usize>::new();
    for i in 0..10 {
        estimator1.insert(&i);
    }
    println!("estimator1 estimate = {}", estimator1.estimate());

    let mut estimator2 = CardinalityEstimator::<usize>::new();
    for i in 10..15 {
        estimator2.insert(&i);
    }
    println!("estimator2 estimate = {}", estimator2.estimate());

    estimator1.merge(&estimator2);
    println!("merged estimate = {}", estimator1.estimate());
}
