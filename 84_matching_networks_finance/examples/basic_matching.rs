//! Basic Matching Network Example
//!
//! This example demonstrates how to use the Matching Network for
//! simple pattern classification.

use matching_networks_finance::{
    MatchingNetwork,
    MarketPattern,
    utils::ClassificationReport,
};
use ndarray::{Array1, Array2};
use rand::Rng;

fn main() {
    println!("=== Matching Networks for Finance - Basic Example ===\n");

    // Generate synthetic data for demonstration
    let (support_features, support_labels, query_features, query_labels) = generate_sample_data();

    println!("Data Generated:");
    println!("  Support set: {} samples", support_features.nrows());
    println!("  Query set: {} samples", query_features.nrows());
    println!("  Number of classes: {}", MarketPattern::count());
    println!();

    // Create a Matching Network
    // Input dim = 20 features, hidden = 64, embedding = 32
    let network = MatchingNetwork::new(
        20,   // input dimension
        64,   // hidden dimension
        32,   // embedding dimension
        true, // use Full Context Embeddings
    );

    println!("Network Configuration:");
    println!("  Input dimension: {}", network.input_dim());
    println!("  Embedding dimension: {}", network.embedding_dim());
    println!("  Uses FCE: {}", network.uses_fce());
    println!();

    // Make predictions
    println!("Making predictions...\n");
    let (predictions, confidences) = network.predict_with_confidence(
        &support_features,
        &support_labels,
        &query_features,
    );

    // Show some predictions
    println!("Sample Predictions:");
    for i in 0..5.min(predictions.len()) {
        let pred_pattern = MarketPattern::from_index(predictions[i])
            .map(|p| format!("{}", p))
            .unwrap_or("Unknown".to_string());
        let true_pattern = MarketPattern::from_index(query_labels[i])
            .map(|p| format!("{}", p))
            .unwrap_or("Unknown".to_string());

        let correct = if predictions[i] == query_labels[i] { "✓" } else { "✗" };

        println!(
            "  Sample {}: Predicted={:<20} True={:<20} Confidence={:.2}% {}",
            i + 1,
            pred_pattern,
            true_pattern,
            confidences[i] * 100.0,
            correct
        );
    }
    println!();

    // Calculate metrics
    let report = ClassificationReport::generate(
        &predictions,
        &query_labels,
        MarketPattern::count(),
    );

    println!("Classification Report:");
    println!("{}", report);

    // Interpret a prediction (show most similar support examples)
    println!("Interpretation for first query:");
    let interpretations = network.interpret(
        &support_features,
        &support_labels,
        &query_features.slice(ndarray::s![0..1, ..]).to_owned(),
        3,
    );

    if let Some(top_matches) = interpretations.first() {
        for (rank, (support_idx, weight, class_idx)) in top_matches.iter().enumerate() {
            let pattern = MarketPattern::from_index(*class_idx)
                .map(|p| format!("{}", p))
                .unwrap_or("Unknown".to_string());
            println!(
                "  #{}: Support sample {} ({}), attention weight: {:.4}",
                rank + 1,
                support_idx,
                pattern,
                weight
            );
        }
    }

    println!("\n=== Example Complete ===");
}

/// Generate synthetic sample data for the example
fn generate_sample_data() -> (Array2<f64>, Array1<usize>, Array2<f64>, Array1<usize>) {
    let mut rng = rand::thread_rng();
    let feature_dim = 20;
    let samples_per_class = 10;
    let query_per_class = 5;
    let num_classes = MarketPattern::count();

    let mut support_features = Vec::new();
    let mut support_labels = Vec::new();
    let mut query_features = Vec::new();
    let mut query_labels = Vec::new();

    // Generate data for each class with class-specific characteristics
    for class_id in 0..num_classes {
        // Class center (different for each class)
        let center: Vec<f64> = (0..feature_dim)
            .map(|j| {
                // Each class has a different center based on class_id
                (class_id as f64) * 0.5 + ((j % 5) as f64) * 0.1
            })
            .collect();

        // Generate support samples
        for _ in 0..samples_per_class {
            let sample: Vec<f64> = center
                .iter()
                .map(|&c| c + rng.gen_range(-0.3..0.3))
                .collect();
            support_features.extend(sample);
            support_labels.push(class_id);
        }

        // Generate query samples
        for _ in 0..query_per_class {
            let sample: Vec<f64> = center
                .iter()
                .map(|&c| c + rng.gen_range(-0.3..0.3))
                .collect();
            query_features.extend(sample);
            query_labels.push(class_id);
        }
    }

    let total_support = num_classes * samples_per_class;
    let total_query = num_classes * query_per_class;

    let support_features = Array2::from_shape_vec(
        (total_support, feature_dim),
        support_features,
    ).unwrap();

    let query_features = Array2::from_shape_vec(
        (total_query, feature_dim),
        query_features,
    ).unwrap();

    let support_labels = Array1::from_vec(support_labels);
    let query_labels = Array1::from_vec(query_labels);

    (support_features, support_labels, query_features, query_labels)
}
