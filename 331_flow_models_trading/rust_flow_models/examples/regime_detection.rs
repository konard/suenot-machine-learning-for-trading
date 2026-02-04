//! Example: Market regime detection using flow latent space
//!
//! This example demonstrates how to use normalizing flows for
//! detecting market regimes by clustering in latent space.

use flow_models_trading::prelude::*;
use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};

fn main() {
    // Initialize logging
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("          Flow Models Trading - Regime Detection");
    println!("═══════════════════════════════════════════════════════════════");

    // Generate market data with distinct regimes
    println!("\n1. Generating market data with distinct regimes...");
    let (data, returns, true_regimes) = generate_regime_data(500, 8);
    println!("   Total samples: {}", data.nrows());

    // Count samples per regime
    let regime_counts: Vec<usize> = (0..4)
        .map(|r| true_regimes.iter().filter(|&&x| x == r).count())
        .collect();
    println!("   Regime distribution: {:?}", regime_counts);

    // Create flow model
    println!("\n2. Creating flow model...");
    let config = FlowConfig::default()
        .with_input_dim(8)
        .with_num_layers(4)
        .with_hidden_dim(64);

    let mut model = NormalizingFlow::new(config);

    // Initialize model
    println!("\n3. Initializing model...");
    let _ = model.forward(&data);

    // Create and fit regime detector
    println!("\n4. Fitting regime detector...");
    let mut detector = RegimeDetector::new(4);
    detector.fit_with_returns(&mut model, &data, &returns);

    println!("\n   Detected regime labels:");
    for (i, label) in detector.regime_labels.iter().enumerate() {
        if let Some(stats) = detector.get_stats(i) {
            println!(
                "     Regime {}: {} (mean_ret={:.4}, vol={:.4}, count={})",
                i, label, stats.mean_return, stats.volatility, stats.count
            );
        } else {
            println!("     Regime {}: {}", i, label);
        }
    }

    // Detect regimes on test data
    println!("\n5. Testing regime detection...");
    let (detected_regimes, distances) = detector.detect(&mut model, &data);

    // Compute accuracy (note: cluster indices may not match true regime indices)
    // We'll compute the best matching
    let accuracy = compute_clustering_accuracy(&true_regimes, &detected_regimes, 4);
    println!("   Clustering accuracy: {:.2}%", accuracy * 100.0);

    // Show regime transitions
    println!("\n6. Regime transitions (first 20 samples):");
    println!("   {:>5} {:>15} {:>15} {:>10}", "Index", "True Regime", "Detected", "Confidence");
    println!("   {:-<50}", "");

    for i in 0..data.nrows().min(20) {
        let true_regime = match true_regimes[i] {
            0 => "High Vol Bull",
            1 => "High Vol Bear",
            2 => "Low Vol Bull",
            3 => "Low Vol Bear",
            _ => "Unknown",
        };
        let detected_label = detector.get_label(detected_regimes[i]);
        let confidence = detector.confidence(&distances.row(i).to_owned());

        println!(
            "   {:>5} {:>15} {:>15} {:>10.2}",
            i, true_regime, detected_label, confidence
        );
    }

    // Regime distribution
    println!("\n7. Detected regime distribution:");
    for regime in 0..4 {
        let count = detected_regimes.iter().filter(|&&r| r == regime).count();
        let label = detector.get_label(regime);
        println!("   {} ({}): {} samples", regime, label, count);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                  Regime detection complete!");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Generate market data with distinct regimes
fn generate_regime_data(n_samples: usize, dim: usize) -> (Array2<f64>, Array1<f64>, Vec<usize>) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    let n_per_regime = n_samples / 4;

    let mut data = Array2::zeros((n_samples, dim));
    let mut returns = Array1::zeros(n_samples);
    let mut regimes = vec![0usize; n_samples];

    // Regime 0: High Vol Bull
    for i in 0..n_per_regime {
        for j in 0..dim {
            data[[i, j]] = normal.sample(&mut rng) * 0.04 + 0.002; // High vol, positive drift
        }
        returns[i] = normal.sample(&mut rng) * 0.04 + 0.002;
        regimes[i] = 0;
    }

    // Regime 1: High Vol Bear
    for i in n_per_regime..2 * n_per_regime {
        for j in 0..dim {
            data[[i, j]] = normal.sample(&mut rng) * 0.04 - 0.002; // High vol, negative drift
        }
        returns[i] = normal.sample(&mut rng) * 0.04 - 0.002;
        regimes[i] = 1;
    }

    // Regime 2: Low Vol Bull
    for i in 2 * n_per_regime..3 * n_per_regime {
        for j in 0..dim {
            data[[i, j]] = normal.sample(&mut rng) * 0.015 + 0.001; // Low vol, positive drift
        }
        returns[i] = normal.sample(&mut rng) * 0.015 + 0.001;
        regimes[i] = 2;
    }

    // Regime 3: Low Vol Bear
    for i in 3 * n_per_regime..n_samples {
        for j in 0..dim {
            data[[i, j]] = normal.sample(&mut rng) * 0.015 - 0.001; // Low vol, negative drift
        }
        returns[i] = normal.sample(&mut rng) * 0.015 - 0.001;
        regimes[i] = 3;
    }

    // Shuffle all together
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    let mut shuffled_data = Array2::zeros((n_samples, dim));
    let mut shuffled_returns = Array1::zeros(n_samples);
    let mut shuffled_regimes = vec![0usize; n_samples];

    for (new_i, &old_i) in indices.iter().enumerate() {
        shuffled_data.row_mut(new_i).assign(&data.row(old_i));
        shuffled_returns[new_i] = returns[old_i];
        shuffled_regimes[new_i] = regimes[old_i];
    }

    (shuffled_data, shuffled_returns, shuffled_regimes)
}

/// Compute clustering accuracy with best matching
fn compute_clustering_accuracy(true_labels: &[usize], pred_labels: &[usize], n_clusters: usize) -> f64 {
    use std::collections::HashMap;

    // Build confusion matrix
    let mut confusion = vec![vec![0usize; n_clusters]; n_clusters];
    for (&t, &p) in true_labels.iter().zip(pred_labels.iter()) {
        if t < n_clusters && p < n_clusters {
            confusion[t][p] += 1;
        }
    }

    // Simple greedy matching (not optimal but good enough)
    let mut used_preds = vec![false; n_clusters];
    let mut correct = 0;

    for true_cluster in 0..n_clusters {
        let mut best_pred = 0;
        let mut best_count = 0;

        for pred_cluster in 0..n_clusters {
            if !used_preds[pred_cluster] && confusion[true_cluster][pred_cluster] > best_count {
                best_count = confusion[true_cluster][pred_cluster];
                best_pred = pred_cluster;
            }
        }

        if best_count > 0 {
            used_preds[best_pred] = true;
            correct += best_count;
        }
    }

    correct as f64 / true_labels.len() as f64
}
