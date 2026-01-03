//! Example: Train GLOW model on cryptocurrency data
//!
//! This example demonstrates how to train a GLOW model
//! on historical market data for trading applications.
//!
//! Run with: cargo run --example train_model

use anyhow::Result;
use glow_trading::{
    Candle, FeatureExtractor, Normalizer, GLOWModel, GLOWConfig, Checkpoint,
};
use indicatif::{ProgressBar, ProgressStyle};

fn main() -> Result<()> {
    println!("=== GLOW Trading: Model Training ===\n");

    // Load data from CSV
    let data_file = "btc_data.csv";
    println!("Loading data from {}...", data_file);

    let mut rdr = match csv::Reader::from_path(data_file) {
        Ok(r) => r,
        Err(_) => {
            println!("Data file not found. Please run fetch_data example first:");
            println!("  cargo run --example fetch_data");
            return Ok(());
        }
    };

    let mut candles = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let candle = Candle {
            timestamp: record[0].parse()?,
            open: record[1].parse()?,
            high: record[2].parse()?,
            low: record[3].parse()?,
            close: record[4].parse()?,
            volume: record[5].parse()?,
            turnover: record[6].parse()?,
        };
        candles.push(candle);
    }

    println!("Loaded {} candles", candles.len());

    // Extract features
    println!("\nExtracting features...");
    let lookback = 20;
    let features = FeatureExtractor::extract_features_batch(&candles, lookback);
    println!("Extracted {} feature vectors with {} features each",
             features.nrows(), features.ncols());

    if features.nrows() < 100 {
        println!("Warning: Not enough data for robust training. Need at least 100 samples.");
        println!("Consider fetching more data (e.g., 30+ days).");
    }

    // Normalize features
    println!("\nNormalizing features...");
    let normalizer = Normalizer::fit(&features);
    let normalized = normalizer.transform(&features);

    // Print feature statistics
    println!("\nFeature Statistics (after normalization):");
    println!("{:-<60}", "");
    for i in 0..normalized.ncols().min(8) {
        let col = normalized.column(i);
        let mean = col.mean().unwrap_or(0.0);
        let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("Feature {}: mean={:.4}, min={:.4}, max={:.4}", i, mean, min, max);
    }
    if normalized.ncols() > 8 {
        println!("... and {} more features", normalized.ncols() - 8);
    }

    // Split data
    let train_size = (normalized.nrows() as f64 * 0.8) as usize;
    let train_data = normalized.slice(ndarray::s![..train_size, ..]).to_owned();
    let val_data = normalized.slice(ndarray::s![train_size.., ..]).to_owned();

    println!("\nData split:");
    println!("  Training samples: {}", train_data.nrows());
    println!("  Validation samples: {}", val_data.nrows());

    // Create GLOW model
    println!("\nCreating GLOW model...");
    let config = GLOWConfig {
        num_features: features.ncols(),
        num_levels: 3,
        num_steps: 4,
        hidden_dim: 64,
        learning_rate: 1e-4,
    };
    let mut model = GLOWModel::new(config.clone());

    println!("Model configuration:");
    println!("  Features: {}", config.num_features);
    println!("  Levels: {}", config.num_levels);
    println!("  Steps per level: {}", config.num_steps);
    println!("  Hidden dimension: {}", config.hidden_dim);
    println!("  Total parameters: {}", model.num_parameters());

    // Training loop
    println!("\nTraining...");
    let epochs = 50;
    let pb = ProgressBar::new(epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut best_val_loss = f64::INFINITY;
    let mut train_losses = Vec::new();
    let mut val_losses = Vec::new();

    for epoch in 0..epochs {
        // Compute training loss (negative log-likelihood)
        let train_log_prob = model.log_prob(&train_data);
        let train_nll = -train_log_prob.mean().unwrap_or(0.0);

        // Compute validation loss
        let val_log_prob = model.log_prob(&val_data);
        let val_nll = -val_log_prob.mean().unwrap_or(0.0);

        train_losses.push(train_nll);
        val_losses.push(val_nll);

        if val_nll < best_val_loss {
            best_val_loss = val_nll;
        }

        pb.set_message(format!(
            "Train NLL: {:.4}, Val NLL: {:.4}, Best: {:.4}",
            train_nll, val_nll, best_val_loss
        ));
        pb.inc(1);
    }
    pb.finish_with_message("Training complete!");

    // Print training summary
    println!("\n=== Training Summary ===");
    println!("Final Train NLL: {:.4}", train_losses.last().unwrap_or(&0.0));
    println!("Final Val NLL: {:.4}", val_losses.last().unwrap_or(&0.0));
    println!("Best Val NLL: {:.4}", best_val_loss);

    // Bits per dimension
    let bpd = best_val_loss / (config.num_features as f64 * 2.0_f64.ln());
    println!("Bits per dimension: {:.4}", bpd);

    // Test generation
    println!("\n=== Model Evaluation ===");

    // Sample from the model
    let samples = model.sample(100, 1.0);
    println!("Generated {} samples", samples.nrows());

    // Check sample statistics
    let sample_mean = samples.mean().unwrap_or(0.0);
    let sample_var = samples.mapv(|v| (v - sample_mean).powi(2)).mean().unwrap_or(0.0);
    println!("Sample mean: {:.4} (should be ~0 for normalized)", sample_mean);
    println!("Sample variance: {:.4} (should be ~1 for normalized)", sample_var);

    // Test invertibility
    println!("\nTesting invertibility...");
    let test_sample = train_data.slice(ndarray::s![0..5, ..]).to_owned();
    let (z, _) = model.forward(&test_sample);
    let reconstructed = model.inverse(&z);

    let reconstruction_error: f64 = (&test_sample - &reconstructed)
        .mapv(|v| v.abs())
        .mean()
        .unwrap_or(0.0);
    println!("Mean absolute reconstruction error: {:.6}", reconstruction_error);

    // Save model
    let output_file = "glow_model.bin";
    println!("\nSaving model to {}...", output_file);

    let mut checkpoint = Checkpoint::new(model);
    checkpoint.set_normalizer(normalizer);
    checkpoint.save(output_file)?;

    println!("Model saved successfully!");
    println!("\nTo run backtest, use:");
    println!("  cargo run --example backtest");

    Ok(())
}
