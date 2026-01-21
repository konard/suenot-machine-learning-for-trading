//! Basic Prototypical Network Example
//!
//! This example demonstrates the fundamental concepts of prototypical networks:
//! 1. Creating an embedding network
//! 2. Computing prototypes from support set
//! 3. Classifying query examples
//!
//! Run with: cargo run --example basic_prototypical

use ndarray::{Array1, Array2};
use prototypical_networks_finance::prelude::*;
use rand::prelude::*;

fn main() {
    println!("=== Basic Prototypical Network Example ===\n");

    // Configuration
    let n_classes = 5; // Number of market regimes
    let n_support = 10; // Support examples per class
    let n_query = 5; // Query examples per class
    let input_dim = 12; // Feature dimension
    let embedding_dim = 8; // Embedding dimension

    // Generate synthetic data for demonstration
    println!("1. Generating synthetic market regime data...");
    let (support_features, support_labels, query_features, query_labels) =
        generate_synthetic_data(n_classes, n_support, n_query, input_dim);

    println!("   - Support set: {} samples x {} features", support_features.nrows(), support_features.ncols());
    println!("   - Query set: {} samples x {} features\n", query_features.nrows(), query_features.ncols());

    // Create embedding network
    println!("2. Creating embedding network...");
    let embedding_config = EmbeddingConfig {
        input_dim,
        hidden_dims: vec![32, 16],
        output_dim: embedding_dim,
        normalize_embeddings: true,
        dropout_rate: 0.0, // No dropout for inference
        activation: ActivationType::ReLU,
    };
    let embedding_network = EmbeddingNetwork::new(embedding_config);
    println!("   - Architecture: {} -> [32, 16] -> {}", input_dim, embedding_dim);

    // Compute embeddings for support set
    println!("\n3. Computing embeddings...");
    let support_embeddings = embedding_network.forward_batch(&support_features);
    println!("   - Support embeddings shape: {:?}", support_embeddings.dim());

    // Create prototype computer
    println!("\n4. Computing prototypes for each class...");
    let mut prototype_computer = PrototypeComputer::new(DistanceFunction::Euclidean);

    // Add examples for each class
    for class_idx in 0..n_classes {
        let start = class_idx * n_support;
        let end = start + n_support;
        let class_embeddings = support_embeddings.slice(ndarray::s![start..end, ..]).to_owned();
        prototype_computer.add_class_examples(class_idx, class_embeddings);
    }
    prototype_computer.compute_prototypes();

    // Display prototypes
    for class_idx in 0..n_classes {
        if let Some(prototype) = prototype_computer.get_prototype(class_idx) {
            let regime = MarketRegime::from_index(class_idx).unwrap();
            println!("   - {}: prototype computed (L2 norm: {:.3})",
                regime.name(),
                prototype.iter().map(|x| x * x).sum::<f64>().sqrt()
            );
        }
    }

    // Classify query examples
    println!("\n5. Classifying query examples...");
    let query_embeddings = embedding_network.forward_batch(&query_features);

    let mut correct = 0;
    let mut total = 0;

    for (i, &true_label) in query_labels.iter().enumerate() {
        let query = query_embeddings.row(i).to_owned();
        let (predicted, probabilities) = prototype_computer.classify(&query);

        let confidence = probabilities.iter().cloned().fold(0.0_f64, f64::max);

        if predicted == true_label {
            correct += 1;
        }
        total += 1;

        // Show first few predictions
        if i < 10 {
            let true_regime = MarketRegime::from_index(true_label).unwrap();
            let pred_regime = MarketRegime::from_index(predicted).unwrap();
            let status = if predicted == true_label { "✓" } else { "✗" };
            println!("   Query {}: True={}, Pred={}, Conf={:.1}% {}",
                i + 1,
                true_regime.name(),
                pred_regime.name(),
                confidence * 100.0,
                status
            );
        }
    }

    println!("\n6. Results:");
    println!("   - Accuracy: {}/{} ({:.1}%)", correct, total, (correct as f64 / total as f64) * 100.0);

    // Demonstrate distance calculations
    println!("\n7. Distance function comparison:");
    let sample_query = query_embeddings.row(0).to_owned();

    for distance_fn in [DistanceFunction::Euclidean, DistanceFunction::Cosine, DistanceFunction::Manhattan] {
        let mut computer = PrototypeComputer::new(distance_fn);
        for class_idx in 0..n_classes {
            let start = class_idx * n_support;
            let end = start + n_support;
            let class_embeddings = support_embeddings.slice(ndarray::s![start..end, ..]).to_owned();
            computer.add_class_examples(class_idx, class_embeddings);
        }
        computer.compute_prototypes();

        let (predicted, _) = computer.classify(&sample_query);
        let regime = MarketRegime::from_index(predicted).unwrap();
        println!("   - {:?}: predicts {}", distance_fn, regime.name());
    }

    println!("\n=== Example Complete ===");
}

/// Generate synthetic market regime data
fn generate_synthetic_data(
    n_classes: usize,
    n_support: usize,
    n_query: usize,
    n_features: usize,
) -> (Array2<f64>, Vec<usize>, Array2<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(42);

    let total_support = n_classes * n_support;
    let total_query = n_classes * n_query;

    let mut support_features = Array2::zeros((total_support, n_features));
    let mut support_labels = Vec::with_capacity(total_support);
    let mut query_features = Array2::zeros((total_query, n_features));
    let mut query_labels = Vec::with_capacity(total_query);

    // Generate class-specific patterns
    for class_idx in 0..n_classes {
        // Each class has a characteristic pattern
        let class_bias: Vec<f64> = (0..n_features)
            .map(|f| {
                // Different classes have different feature biases
                match class_idx {
                    0 => 0.3 + 0.1 * f as f64 / n_features as f64, // Strong uptrend
                    1 => 0.1 + 0.05 * f as f64 / n_features as f64, // Weak uptrend
                    2 => 0.0, // Sideways
                    3 => -0.1 - 0.05 * f as f64 / n_features as f64, // Weak downtrend
                    4 => -0.3 - 0.1 * f as f64 / n_features as f64, // Strong downtrend
                    _ => 0.0,
                }
            })
            .collect();

        // Support samples
        for i in 0..n_support {
            let row_idx = class_idx * n_support + i;
            for j in 0..n_features {
                support_features[[row_idx, j]] = class_bias[j] + rng.gen::<f64>() * 0.2 - 0.1;
            }
            support_labels.push(class_idx);
        }

        // Query samples
        for i in 0..n_query {
            let row_idx = class_idx * n_query + i;
            for j in 0..n_features {
                query_features[[row_idx, j]] = class_bias[j] + rng.gen::<f64>() * 0.2 - 0.1;
            }
            query_labels.push(class_idx);
        }
    }

    (support_features, support_labels, query_features, query_labels)
}
