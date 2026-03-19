//! Example: Train Disentangled VAE model

use disentangled_vae::{DataLoader, DisentangledVAE, DisentangledVAEConfig, DisentanglementMethod};
use disentangled_vae::api::Kline;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Disentangled VAE Trading - Training Example");
    println!("============================================\n");

    // Generate synthetic data for demonstration
    println!("Generating synthetic price data...");
    let klines = generate_synthetic_klines(1000);
    println!("Generated {} candles\n", klines.len());

    // Prepare dataset
    let loader = DataLoader::new()
        .seq_len(60)
        .target_horizon(12)
        .normalize(true);

    let dataset = loader.prepare_dataset(&klines)?;
    let (train_dataset, test_dataset) = dataset.train_test_split(0.8);

    println!("Dataset prepared:");
    println!("  Training samples: {}", train_dataset.len());
    println!("  Test samples: {}", test_dataset.len());
    println!("  Sequence length: {}", train_dataset.seq_len);
    println!("  Features: {}", train_dataset.num_features);

    // Create model with BetaVAE method
    let config = DisentangledVAEConfig {
        input_dim: train_dataset.num_features,
        latent_dim: 8,
        hidden_dim: 64,
        num_hidden_layers: 2,
        seq_len: train_dataset.seq_len,
        output_dim: 1,
        beta: 4.0,
        method: DisentanglementMethod::BetaVAE,
        ..Default::default()
    };

    println!("\nModel configuration:");
    println!("  Input dim: {}", config.input_dim);
    println!("  Latent dim: {}", config.latent_dim);
    println!("  Hidden dim: {}", config.hidden_dim);
    println!("  Beta: {}", config.beta);
    println!("  Method: {}", config.method);

    let mut model = DisentangledVAE::new(config);

    // Training loop (forward pass demonstration)
    println!("\nTraining (forward pass demonstration)...");
    let batch_size = 16;
    let num_epochs = 3;

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (batch_inputs, batch_targets) in train_dataset.iter_batches(batch_size) {
            let loss = model.compute_loss(&batch_inputs, &batch_targets);
            total_loss += loss;
            num_batches += 1;
        }

        let avg_loss = total_loss / num_batches as f64;
        println!("  Epoch {}: avg loss = {:.6}", epoch + 1, avg_loss);
    }

    // Detailed loss breakdown
    println!("\nDetailed loss on last batch...");
    let indices = train_dataset.random_batch_indices(batch_size);
    let (batch_inputs, batch_targets) = train_dataset.get_batch(&indices);
    let detailed_loss = model.compute_detailed_loss(&batch_inputs, &batch_targets);
    println!("  Reconstruction loss: {:.6}", detailed_loss.reconstruction);
    println!("  KL divergence: {:.6}", detailed_loss.kl_divergence);
    println!("  Disentanglement penalty: {:.6}", detailed_loss.disentanglement_penalty);
    println!("  Total loss: {:.6}", detailed_loss.total);

    // Latent factor analysis
    println!("\nLatent factor statistics:");
    let factor_stats = model.latent_factor_stats(&test_dataset);
    for (i, (mean, std)) in factor_stats.iter().enumerate() {
        println!("  Factor {}: mean={:.4}, std={:.4}", i, mean, std);
    }

    // Evaluation
    println!("\nEvaluating on test set...");
    let predictions = model.predict(&test_dataset);

    let mse: f64 = predictions
        .iter()
        .zip(test_dataset.targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / predictions.len() as f64;

    let mae: f64 = predictions
        .iter()
        .zip(test_dataset.targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>() / predictions.len() as f64;

    let correct_direction = predictions
        .iter()
        .zip(test_dataset.targets.iter())
        .filter(|(p, t)| (**p >= 0.0) == (**t >= 0.0))
        .count();
    let direction_accuracy = correct_direction as f64 / predictions.len() as f64;

    println!("\nTest Results:");
    println!("  MSE: {:.6}", mse);
    println!("  MAE: {:.6}", mae);
    println!("  Direction Accuracy: {:.2}%", direction_accuracy * 100.0);

    println!("\nSample predictions (first 10):");
    println!("  {:>10} {:>10}", "Predicted", "Actual");
    for i in 0..10.min(predictions.len()) {
        println!("  {:>10.6} {:>10.6}", predictions[i], test_dataset.targets[i]);
    }

    println!("\nTraining complete!");

    Ok(())
}

/// Generate synthetic price data for demonstration
fn generate_synthetic_klines(n: usize) -> Vec<Kline> {
    let mut klines = Vec::with_capacity(n);
    let mut price = 100.0;

    for i in 0..n {
        let trend = 0.0001 * i as f64;
        let seasonality = 5.0 * (2.0 * PI * i as f64 / 100.0).sin();
        let noise = rand_normal() * 2.0;

        let change = trend + seasonality * 0.01 + noise * 0.01;
        price *= 1.0 + change;

        let open = price;
        let close = price * (1.0 + rand_normal() * 0.005);
        let high = open.max(close) * (1.0 + rand_normal().abs() * 0.01);
        let low = open.min(close) * (1.0 - rand_normal().abs() * 0.01);
        let volume = 1000.0 + rand_normal().abs() * 500.0;

        klines.push(Kline {
            timestamp: i as u64 * 3600000,
            open,
            high,
            low,
            close,
            volume,
            turnover: close * volume,
        });

        price = close;
    }

    klines
}

fn rand_normal() -> f64 {
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
