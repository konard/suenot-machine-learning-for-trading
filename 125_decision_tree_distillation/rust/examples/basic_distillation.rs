//! Basic Decision Tree Distillation Example
//!
//! This example demonstrates how to distill a simple decision tree
//! from synthetic data and extract interpretable rules.

use decision_tree_distillation::model::{DecisionTree, DistillationConfig};
use ndarray::{Array1, Array2};
use rand::Rng;

fn main() {
    println!("Decision Tree Distillation Example");
    println!("{}", "=".repeat(50));

    // Generate synthetic training data
    println!("\n1. Generating synthetic trading data...");
    let n_samples = 1000;
    let (features, soft_labels) = generate_synthetic_features(n_samples);
    println!("   Generated {} samples with {} features", n_samples, features.ncols());

    // Configure distillation
    let config = DistillationConfig {
        max_depth: 4,
        min_samples_leaf: 20,
        min_samples_split: 40,
        feature_names: Some(vec![
            "RSI_14".to_string(),
            "MACD".to_string(),
            "Volume_Ratio".to_string(),
            "Volatility".to_string(),
        ]),
        class_names: Some(vec!["DOWN".to_string(), "UP".to_string()]),
    };

    // Train decision tree
    println!("\n2. Training distilled decision tree...");
    let mut tree = DecisionTree::new(config);
    tree.fit(&features, &soft_labels).expect("Failed to fit tree");
    println!("   Tree depth: {}", tree.depth());
    println!("   Number of leaves: {}", tree.n_leaves());

    // Extract rules
    println!("\n3. Extracted Trading Rules:");
    println!("{}", "-".repeat(50));
    let rules = tree.extract_rules().expect("Failed to extract rules");
    for rule in &rules {
        println!("   {}", rule);
    }

    // Get feature importance
    println!("\n4. Feature Importance:");
    println!("{}", "-".repeat(50));
    let importance = tree.feature_importance().expect("Failed to get importance");
    let mut importance_vec: Vec<_> = importance.into_iter().collect();
    importance_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let feature_names = ["RSI_14", "MACD", "Volume_Ratio", "Volatility"];
    for (idx, imp) in importance_vec {
        println!("   {}: {:.3}", feature_names[idx], imp);
    }

    // Make predictions on test data
    println!("\n5. Making predictions on test data...");
    let (test_features, test_labels) = generate_synthetic_features(200);
    let predictions = tree.predict(&test_features).expect("Failed to predict");

    // Calculate accuracy
    let correct: usize = predictions
        .iter()
        .zip(test_labels.iter())
        .map(|(p, l)| {
            let p_class = if *p > 0.5 { 1 } else { 0 };
            let l_class = if *l > 0.5 { 1 } else { 0 };
            if p_class == l_class { 1 } else { 0 }
        })
        .sum();
    let accuracy = correct as f64 / predictions.len() as f64;
    println!("   Test accuracy: {:.2}%", accuracy * 100.0);

    // Calculate fidelity to "teacher" (soft labels)
    let fidelity = tree
        .fidelity_score(&test_features, &test_labels)
        .expect("Failed to calculate fidelity");
    println!("   Fidelity to teacher: {:.2}%", fidelity * 100.0);

    println!("\n{}", "=".repeat(50));
    println!("Distillation complete!");
}

/// Generate synthetic features that simulate a complex teacher model's behavior
fn generate_synthetic_features(n_samples: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();

    let mut features = Array2::<f64>::zeros((n_samples, 4));
    let mut labels = Array1::<f64>::zeros(n_samples);

    for i in 0..n_samples {
        // RSI: 0-100
        let rsi = rng.gen_range(0.0..100.0);
        // MACD: -2 to 2
        let macd = rng.gen_range(-2.0..2.0);
        // Volume ratio: 0.5 to 2.0
        let volume_ratio = rng.gen_range(0.5..2.0);
        // Volatility: 0.01 to 0.05
        let volatility = rng.gen_range(0.01..0.05);

        features[[i, 0]] = rsi;
        features[[i, 1]] = macd;
        features[[i, 2]] = volume_ratio;
        features[[i, 3]] = volatility;

        // Simulate a complex teacher model's predictions
        // Rule: Buy when RSI < 35 AND MACD > 0, or when RSI < 45 AND MACD > 0.5 AND volume_ratio > 1.5
        let buy_signal_1 = rsi < 35.0 && macd > 0.0;
        let buy_signal_2 = rsi < 45.0 && macd > 0.5 && volume_ratio > 1.5;

        // Add some noise to simulate soft labels
        let base_prob: f64 = if buy_signal_1 {
            0.8
        } else if buy_signal_2 {
            0.7
        } else if rsi < 30.0 {
            0.6
        } else if rsi > 70.0 {
            0.3
        } else {
            0.5
        };

        // Add noise
        let noise = rng.gen_range(-0.1..0.1);
        labels[i] = (base_prob + noise).max(0.0).min(1.0);
    }

    (features, labels)
}
