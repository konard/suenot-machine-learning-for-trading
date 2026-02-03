//! Example: Train a trading classifier
//!
//! This example demonstrates how to train a simple neural network
//! classifier for trading signals.
//!
//! Run with: cargo run --example train_classifier

use rust_counterfactual::{
    data::get_sample_data,
    model::TradingClassifier,
};

fn main() {
    println!("Training Trading Classifier");
    println!("===========================\n");

    // Load sample data
    println!("1. Loading sample data...");
    let (features, labels, feature_names) = get_sample_data();
    println!("   Loaded {} samples with {} features", features.len(), feature_names.len());
    println!("   Features: {:?}", feature_names);

    // Count label distribution
    let sell_count = labels.iter().filter(|&&l| l == 0).count();
    let hold_count = labels.iter().filter(|&&l| l == 1).count();
    let buy_count = labels.iter().filter(|&&l| l == 2).count();
    println!("   Label distribution: SELL={}, HOLD={}, BUY={}\n",
             sell_count, hold_count, buy_count);

    // Split into train/test
    println!("2. Splitting data...");
    let split_idx = (features.len() as f64 * 0.8) as usize;
    let train_features = &features[..split_idx];
    let train_labels = &labels[..split_idx];
    let test_features = &features[split_idx..];
    let test_labels = &labels[split_idx..];
    println!("   Train: {} samples, Test: {} samples\n", train_features.len(), test_features.len());

    // Create and train model
    println!("3. Training model...");
    let mut model = TradingClassifier::new(
        feature_names.len(),
        64,  // hidden dimension
        3,   // number of classes
    );

    model.train(train_features, train_labels, 50, 0.01);

    // Evaluate on test set
    println!("\n4. Evaluating on test set...");
    let mut correct = 0;
    let mut predictions = Vec::new();

    for (x, &y) in test_features.iter().zip(test_labels.iter()) {
        let pred = model.predict(x);
        predictions.push(pred);
        if pred == y {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / test_labels.len() as f64;
    println!("   Test accuracy: {:.2}%\n", accuracy * 100.0);

    // Show confusion matrix
    println!("5. Confusion matrix:");
    let mut matrix = [[0usize; 3]; 3];
    for (pred, &actual) in predictions.iter().zip(test_labels.iter()) {
        matrix[actual][*pred] += 1;
    }

    println!("                Predicted");
    println!("           SELL  HOLD  BUY");
    println!("   Actual");
    for (i, row) in matrix.iter().enumerate() {
        let label = match i { 0 => "SELL", 1 => "HOLD", 2 => "BUY", _ => "?" };
        println!("   {:>5}  {:>5} {:>5} {:>5}", label, row[0], row[1], row[2]);
    }

    // Show some predictions
    println!("\n6. Sample predictions:");
    for i in 0..5.min(test_features.len()) {
        let x = &test_features[i];
        let probs = model.predict_proba(x);
        let pred = model.predict(x);
        let actual = test_labels[i];

        println!("   Sample {}: Predicted {} ({:.1}% conf), Actual {}",
                 i + 1,
                 model.get_class_name(pred),
                 probs[pred] * 100.0,
                 model.get_class_name(actual));
    }

    println!("\nDone!");
}
