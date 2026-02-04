//! Example: Train a normalizing flow model
//!
//! This example demonstrates how to create and train a normalizing flow
//! model on synthetic market data.

use flow_models_trading::prelude::*;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

fn main() {
    // Initialize logging
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("            Flow Models Trading - Train Flow Model");
    println!("═══════════════════════════════════════════════════════════════");

    // Create synthetic market data
    println!("\n1. Generating synthetic market data...");
    let (train_data, val_data) = generate_synthetic_data(2000, 8);
    println!("   Train samples: {}", train_data.nrows());
    println!("   Validation samples: {}", val_data.nrows());

    // Create flow model configuration
    println!("\n2. Creating flow model...");
    let config = FlowConfig::default()
        .with_input_dim(8)
        .with_num_layers(4)
        .with_hidden_dim(64);

    println!("   Input dimension: {}", config.input_dim);
    println!("   Number of layers: {}", config.num_layers);
    println!("   Hidden dimension: {}", config.hidden_dim);

    let mut model = NormalizingFlow::new(config);

    // Train model
    println!("\n3. Training flow model...");
    let losses = model.train(&train_data, 30).expect("Training failed");

    println!("\n   Training losses (every 10 epochs):");
    for (i, loss) in losses.iter().enumerate() {
        if i % 10 == 0 {
            println!("   Epoch {:3}: loss = {:.4}", i, loss);
        }
    }

    // Test likelihood computation
    println!("\n4. Testing likelihood computation...");
    let log_probs = model.log_prob(&val_data);
    println!("   Mean log-prob: {:.4}", log_probs.mean().unwrap_or(0.0));
    println!("   Std log-prob: {:.4}", log_probs.std(0.0));
    println!("   Min log-prob: {:.4}", log_probs.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("   Max log-prob: {:.4}", log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // Test sampling
    println!("\n5. Testing sampling...");
    let samples = model.sample(100);
    println!("   Generated {} samples", samples.nrows());
    println!("   Sample mean: {:?}", samples.mean_axis(ndarray::Axis(0)).unwrap().to_vec().iter().take(4).collect::<Vec<_>>());

    // Test reconstruction (invertibility)
    println!("\n6. Testing reconstruction (invertibility)...");
    let test_data = val_data.slice(ndarray::s![0..5, ..]).to_owned();
    let (z, _) = model.forward(&test_data);
    let reconstructed = model.inverse(&z);

    let recon_error: f64 = (&test_data - &reconstructed).mapv(|v| v.abs()).mean().unwrap_or(0.0);
    println!("   Mean absolute reconstruction error: {:.8}", recon_error);

    if recon_error < 1e-5 {
        println!("   [OK] Excellent reconstruction (error < 1e-5)");
    } else if recon_error < 1e-3 {
        println!("   [OK] Good reconstruction (error < 1e-3)");
    } else {
        println!("   [WARN] High reconstruction error");
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    Training complete!");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Generate synthetic market-like data
fn generate_synthetic_data(n_samples: usize, dim: usize) -> (Array2<f64>, Array2<f64>) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    // Generate two clusters (bull and bear regimes)
    let n_half = n_samples / 2;

    let mut data = Array2::zeros((n_samples, dim));

    // Bull regime: slight positive mean, lower volatility
    for i in 0..n_half {
        for j in 0..dim {
            data[[i, j]] = normal.sample(&mut rng) * 0.02 + 0.001;
        }
    }

    // Bear regime: slight negative mean, higher volatility
    for i in n_half..n_samples {
        for j in 0..dim {
            data[[i, j]] = normal.sample(&mut rng) * 0.03 - 0.001;
        }
    }

    // Shuffle
    let mut indices: Vec<usize> = (0..n_samples).collect();
    for i in (1..n_samples).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }

    let mut shuffled = Array2::zeros((n_samples, dim));
    for (new_i, &old_i) in indices.iter().enumerate() {
        shuffled.row_mut(new_i).assign(&data.row(old_i));
    }

    // Split into train/validation
    let train_size = (n_samples as f64 * 0.8) as usize;
    let train_data = shuffled.slice(ndarray::s![0..train_size, ..]).to_owned();
    let val_data = shuffled.slice(ndarray::s![train_size.., ..]).to_owned();

    (train_data, val_data)
}
