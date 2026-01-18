//! Example: Training the Informer model
//!
//! Run with: cargo run --example train

use informer_probsparse::{
    InformerConfig, InformerModel, DataLoader,
    data::TimeSeriesDataset,
};
use ndarray::Array3;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Informer ProbSparse: Training Example ===\n");

    // Create synthetic data for demonstration
    println!("Creating synthetic training data...");
    let (features, targets) = create_synthetic_data(1000, 6);

    // Create dataset
    let seq_len = 48;
    let pred_len = 12;
    let dataset = TimeSeriesDataset::new(&features, &targets, seq_len, pred_len)?;

    println!("Dataset created:");
    println!("  Total samples: {}", dataset.n_samples);
    println!("  Sequence length: {}", dataset.seq_len);
    println!("  Prediction horizon: {}", dataset.pred_len);
    println!("  Features: {}", dataset.n_features);

    // Split data
    let (train, val, test) = dataset.split(0.7, 0.15);
    println!("\nData split:");
    println!("  Train: {} samples", train.n_samples);
    println!("  Validation: {} samples", val.n_samples);
    println!("  Test: {} samples", test.n_samples);

    // Create model
    println!("\n--- Model Configuration ---\n");

    let config = InformerConfig {
        seq_len,
        pred_len,
        input_features: 6,
        d_model: 32,
        n_heads: 4,
        d_ff: 64,
        n_encoder_layers: 2,
        sampling_factor: 5.0,
        use_distilling: true,
        ..Default::default()
    };

    let model = InformerModel::new(config.clone());

    println!("Model created with ProbSparse attention:");
    println!("  d_model: {}", config.d_model);
    println!("  n_heads: {}", config.n_heads);
    println!("  encoder layers: {}", config.n_encoder_layers);
    println!("  sampling factor: {}", config.sampling_factor);
    println!("  use distilling: {}", config.use_distilling);
    println!("  parameters: ~{}", model.num_parameters());

    // Training loop simulation
    println!("\n--- Training Simulation ---\n");

    let epochs = 5;
    let batch_size = 16;

    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        for (batch_x, batch_y) in train.batches(batch_size, true) {
            // Forward pass
            let predictions = model.predict(&batch_x);

            // Calculate MSE loss (simplified)
            let batch_loss = calculate_mse(&predictions, &batch_y);
            epoch_loss += batch_loss;
            n_batches += 1;

            // In real training, we would:
            // 1. Compute gradients
            // 2. Update weights with optimizer
        }

        let avg_loss = epoch_loss / n_batches as f64;
        println!("Epoch {}/{}: avg_loss = {:.6}", epoch, epochs, avg_loss);
    }

    // Validation
    println!("\n--- Validation ---\n");

    let mut val_loss = 0.0;
    let mut n_val_batches = 0;

    for (batch_x, batch_y) in val.batches(batch_size, false) {
        let predictions = model.predict(&batch_x);
        val_loss += calculate_mse(&predictions, &batch_y);
        n_val_batches += 1;
    }

    println!("Validation loss: {:.6}", val_loss / n_val_batches as f64);

    // Test prediction
    println!("\n--- Sample Predictions ---\n");

    let (test_x, test_y) = test.get_batch(&[0, 1, 2]);
    let predictions = model.predict(&test_x);

    for i in 0..3 {
        println!("Sample {}:", i);
        print!("  Predicted: [");
        for j in 0..pred_len.min(5) {
            print!("{:.4}", predictions[[i, j]]);
            if j < pred_len.min(5) - 1 { print!(", "); }
        }
        println!("{}]", if pred_len > 5 { ", ..." } else { "" });

        print!("  Actual:    [");
        for j in 0..pred_len.min(5) {
            print!("{:.4}", test_y[[i, j]]);
            if j < pred_len.min(5) - 1 { print!(", "); }
        }
        println!("{}]", if pred_len > 5 { ", ..." } else { "" });
    }

    println!("\n=== Training Complete ===");

    Ok(())
}

/// Create synthetic time series data
fn create_synthetic_data(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    use std::f64::consts::PI;

    let features: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let t = i as f64;
            (0..n_features)
                .map(|f| {
                    // Create different patterns for each feature
                    let freq = (f + 1) as f64 * 0.1;
                    let base = (t * freq * 2.0 * PI / 100.0).sin();
                    let noise = rand_normal() * 0.1;
                    base + noise
                })
                .collect()
        })
        .collect();

    // Target is sum of features with lag
    let targets: Vec<f64> = (0..n)
        .map(|i| {
            features[i].iter().sum::<f64>() / n_features as f64
                + rand_normal() * 0.05
        })
        .collect();

    (features, targets)
}

/// Calculate MSE between predictions and targets
fn calculate_mse(predictions: &ndarray::Array2<f64>, targets: &ndarray::Array2<f64>) -> f64 {
    let diff = predictions - targets;
    diff.mapv(|x| x.powi(2)).mean().unwrap_or(0.0)
}

/// Generate random normal number
fn rand_normal() -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
