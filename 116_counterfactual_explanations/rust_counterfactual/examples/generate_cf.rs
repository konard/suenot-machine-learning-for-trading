//! Example: Generate counterfactual explanations
//!
//! This example demonstrates how to generate counterfactual explanations
//! for trading model predictions.
//!
//! Run with: cargo run --example generate_cf

use rust_counterfactual::{
    counterfactual::CounterfactualOptimizer,
    data::get_sample_data,
    model::TradingClassifier,
};

fn main() {
    println!("Counterfactual Explanation Generation");
    println!("=====================================\n");

    // Load sample data
    println!("1. Loading sample data and training model...");
    let (features, labels, feature_names) = get_sample_data();

    let mut model = TradingClassifier::new(feature_names.len(), 64, 3);
    model.train(&features, &labels, 30, 0.01);
    println!("   Model trained.\n");

    // Create optimizer
    let optimizer = CounterfactualOptimizer::new(&model)
        .with_proximity_weight(1.0)
        .with_validity_weight(1.0)
        .with_sparsity_weight(0.1);

    // Generate counterfactuals for a few samples
    println!("2. Generating counterfactual explanations...\n");

    for i in 0..3 {
        let x = &features[i];
        let pred = model.predict(x);
        let probs = model.predict_proba(x);

        println!("Sample {}", i + 1);
        println!("---------");
        println!("Input features:");
        for (j, name) in feature_names.iter().enumerate() {
            println!("  {}: {:.4}", name, x[j]);
        }
        println!("\nOriginal prediction: {} ({:.1}% confidence)",
                 model.get_class_name(pred),
                 probs[pred] * 100.0);

        // Find counterfactual to opposite class
        let target_class = if pred == 2 { 0 } else if pred == 0 { 2 } else { 2 };
        println!("Target class: {}", model.get_class_name(target_class));

        let result = optimizer.generate(x, target_class, 100, 0.05);
        println!("\n{}", result);

        // Show with feature names
        if !result.changed_features.is_empty() {
            println!("Changed features (named):");
            for (idx, orig, cf) in &result.changed_features {
                let name = &feature_names[*idx];
                let direction = if cf > orig { "increase" } else { "decrease" };
                println!("  - {}: {:.4} -> {:.4} ({} by {:.4})",
                         name, orig, cf, direction, (cf - orig).abs());
            }
        }

        println!("\n{}", "=".repeat(60));
        println!();
    }

    // Summary statistics
    println!("3. Summary statistics over 50 samples...\n");
    let mut valid_count = 0;
    let mut total_changes = 0;
    let mut total_l1 = 0.0;

    for i in 0..50.min(features.len()) {
        let x = &features[i];
        let pred = model.predict(x);
        if pred == 1 { continue; } // Skip HOLD predictions

        let target_class = if pred == 2 { 0 } else { 2 };
        let result = optimizer.generate(x, target_class, 50, 0.05);

        if result.is_valid {
            valid_count += 1;
        }
        total_changes += result.num_features_changed;
        total_l1 += result.l1_distance;
    }

    let n = 50.min(features.len()) as f64;
    println!("   Validity rate: {:.1}%", (valid_count as f64 / n) * 100.0);
    println!("   Avg features changed: {:.2}", total_changes as f64 / n);
    println!("   Avg L1 distance: {:.4}", total_l1 / n);

    println!("\nDone!");
}
