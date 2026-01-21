//! Basic Few-Shot Market Prediction Example
//!
//! This example demonstrates the core few-shot learning functionality
//! for market prediction using synthetic data.
//!
//! Run with: cargo run --example basic_few_shot

use few_shot_market_prediction::methods::{
    FewShotConfig, FewShotLearner, FewShotMethod, FewShotPredictor,
    MetricBasedLearner, MAMLLearner, SiameseLearner, HybridLearner,
};
use few_shot_market_prediction::network::DistanceMetric;
use ndarray::Array2;

fn main() {
    println!("=== Few-Shot Market Prediction: Basic Example ===\n");

    // Generate synthetic market features for demonstration
    let (support_features, support_labels, query_features, query_labels) = generate_synthetic_data();

    println!("Support set: {} examples across 3 classes", support_features.nrows());
    println!("Query set: {} examples to classify\n", query_features.nrows());

    // Define class names
    let class_names = ["Uptrend", "Sideways", "Downtrend"];

    // Method 1: Metric-based learning (Prototypical Networks)
    println!("--- Method 1: Prototypical Networks ---");
    let config_metric = FewShotConfig::default()
        .with_method(FewShotMethod::Metric)
        .with_input_dim(10)
        .with_embedding_dim(8)
        .with_hidden_dims(vec![32, 16])
        .with_distance_metric(DistanceMetric::Euclidean);

    let mut metric_predictor = FewShotPredictor::new(config_metric);
    metric_predictor.fit(&support_features, &support_labels);
    let metric_results = metric_predictor.predict(&query_features);

    println!("Query predictions (metric-based):");
    for (i, result) in metric_results.iter().enumerate() {
        println!(
            "  Query {}: {} (confidence: {:.1}%, reliable: {})",
            i,
            class_names[result.predicted_class],
            result.confidence * 100.0,
            result.is_reliable
        );
    }
    println!();

    // Method 2: MAML-based learning
    println!("--- Method 2: MAML (Model-Agnostic Meta-Learning) ---");
    let config_maml = FewShotConfig::default()
        .with_method(FewShotMethod::MAML)
        .with_input_dim(10)
        .with_embedding_dim(8)
        .with_hidden_dims(vec![32, 16])
        .with_adaptation_steps(3)
        .with_adaptation_lr(0.01);

    let mut maml_learner = MAMLLearner::new(config_maml);
    maml_learner.fit(&support_features, &support_labels);
    let maml_results = maml_learner.predict(&query_features);

    println!("Query predictions (MAML):");
    for (i, result) in maml_results.iter().enumerate() {
        println!(
            "  Query {}: {} (confidence: {:.1}%)",
            i,
            class_names[result.predicted_class],
            result.confidence * 100.0
        );
    }

    // Print adaptation info
    if let Some(adaptation) = maml_learner.last_adaptation() {
        println!("  Adaptation: {} steps, loss {:.4} -> {:.4} ({:.1}% improvement)",
            adaptation.steps,
            adaptation.initial_loss,
            adaptation.final_loss,
            adaptation.improvement * 100.0
        );
    }
    println!();

    // Method 3: Siamese Network
    println!("--- Method 3: Siamese Network ---");
    let config_siamese = FewShotConfig::default()
        .with_method(FewShotMethod::Siamese)
        .with_input_dim(10)
        .with_embedding_dim(8)
        .with_hidden_dims(vec![32, 16]);

    let mut siamese_learner = SiameseLearner::new(config_siamese);
    siamese_learner.fit(&support_features, &support_labels);
    let siamese_results = siamese_learner.predict(&query_features);

    println!("Query predictions (Siamese):");
    for (i, result) in siamese_results.iter().enumerate() {
        println!(
            "  Query {}: {} (confidence: {:.1}%)",
            i,
            class_names[result.predicted_class],
            result.confidence * 100.0
        );
    }
    println!();

    // Method 4: Hybrid approach
    println!("--- Method 4: Hybrid (Metric + Optimization) ---");
    let config_hybrid = FewShotConfig::default()
        .with_method(FewShotMethod::Hybrid)
        .with_input_dim(10)
        .with_embedding_dim(8)
        .with_hidden_dims(vec![32, 16]);

    let mut hybrid_learner = HybridLearner::new(config_hybrid);
    hybrid_learner.fit(&support_features, &support_labels);
    let hybrid_results = hybrid_learner.predict(&query_features);

    println!("Query predictions (Hybrid):");
    for (i, result) in hybrid_results.iter().enumerate() {
        println!(
            "  Query {}: {} (confidence: {:.1}%)",
            i,
            class_names[result.predicted_class],
            result.confidence * 100.0
        );
    }
    println!();

    // Compare actual labels
    println!("--- Ground Truth ---");
    for (i, &label) in query_labels.iter().enumerate() {
        println!("  Query {}: {}", i, class_names[label]);
    }

    // Calculate accuracy for each method
    println!("\n--- Accuracy Summary ---");
    let methods = [
        ("Prototypical", &metric_results),
        ("MAML", &maml_results),
        ("Siamese", &siamese_results),
        ("Hybrid", &hybrid_results),
    ];

    for (name, results) in methods {
        let correct = results
            .iter()
            .zip(&query_labels)
            .filter(|(pred, &actual)| pred.predicted_class == actual)
            .count();
        let accuracy = correct as f64 / query_labels.len() as f64 * 100.0;
        println!("  {}: {:.1}% ({}/{})", name, accuracy, correct, query_labels.len());
    }

    println!("\n=== Example Complete ===");
}

/// Generate synthetic market data for the example.
fn generate_synthetic_data() -> (Array2<f64>, Vec<usize>, Array2<f64>, Vec<usize>) {
    let dim = 10;
    let n_support_per_class = 5;
    let n_query_per_class = 3;
    let n_classes = 3;

    // Base feature patterns for each class
    // Uptrend: positive returns, momentum indicators high
    let uptrend_base: Vec<f64> = vec![0.02, 0.015, 0.7, 65.0, 1.2, 0.8, 0.7, 0.6, 0.3, 0.7];
    // Sideways: near-zero returns, neutral indicators
    let sideways_base: Vec<f64> = vec![0.001, 0.002, 0.5, 50.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5];
    // Downtrend: negative returns, bearish indicators
    let downtrend_base: Vec<f64> = vec![-0.02, -0.015, 0.3, 35.0, 0.7, 0.3, 0.3, 0.4, 0.7, 0.3];

    let bases = [&uptrend_base, &sideways_base, &downtrend_base];

    let mut support_data = Vec::new();
    let mut support_labels = Vec::new();
    let mut query_data = Vec::new();
    let mut query_labels = Vec::new();

    for (class_idx, base) in bases.iter().enumerate() {
        // Support examples
        for i in 0..n_support_per_class {
            let noisy = add_noise(base, 0.1, (class_idx * 100 + i) as u64);
            support_data.extend(noisy);
            support_labels.push(class_idx);
        }

        // Query examples
        for i in 0..n_query_per_class {
            let noisy = add_noise(base, 0.15, (class_idx * 1000 + i) as u64);
            query_data.extend(noisy);
            query_labels.push(class_idx);
        }
    }

    let support = Array2::from_shape_vec(
        (n_classes * n_support_per_class, dim),
        support_data,
    ).unwrap();

    let query = Array2::from_shape_vec(
        (n_classes * n_query_per_class, dim),
        query_data,
    ).unwrap();

    (support, support_labels, query, query_labels)
}

/// Add deterministic pseudo-random noise to a feature vector.
fn add_noise(base: &[f64], noise_level: f64, seed: u64) -> Vec<f64> {
    let mut rng_state = seed;

    base.iter().map(|&v| {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random = ((rng_state >> 16) & 0x7FFF) as f64 / 32768.0 - 0.5;
        v + v.abs().max(0.01) * noise_level * random * 2.0
    }).collect()
}
