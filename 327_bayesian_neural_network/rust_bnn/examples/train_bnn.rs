//! Example: Train a Bayesian Neural Network on cryptocurrency data.
//!
//! Usage:
//!     cargo run --example train_bnn -- --symbol BTCUSDT

use anyhow::Result;
use bayesian_nn_trading::prelude::*;
use bayesian_nn_trading::bnn::config::KLSchedule;
use bayesian_nn_trading::features::engine::FeatureEngine;
use clap::Parser;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train BNN on Bybit data")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Timeframe (60 = 1h)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles
    #[arg(short, long, default_value = "500")]
    limit: u32,

    /// Training epochs
    #[arg(short, long, default_value = "50")]
    epochs: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "BNN Training - Cryptocurrency Trading".blue().bold());
    println!("{}", "=".repeat(60).blue());

    // Fetch data
    println!("\n{} Fetching data from Bybit...", "[1/4]".cyan());
    let client = BybitClient::new();
    let klines = client
        .get_klines_extended(&args.symbol, &args.interval, args.limit)
        .await?;

    println!("  {} {} candles for {}", "Fetched".green(), klines.len(), args.symbol);

    // Create features
    println!("\n{} Creating features...", "[2/4]".cyan());
    let feature_engine = FeatureEngine::new()
        .lookback(20)
        .prediction_horizon(5)
        .return_threshold(0.002);

    let features = feature_engine.create_features(&klines)?;

    println!("  {} {} samples with {} features",
        "Created".green(),
        features.x.nrows(),
        features.x.ncols()
    );

    // Show class distribution
    let n_up = features.y_class.iter().filter(|&&y| y == 0).count();
    let n_neutral = features.y_class.iter().filter(|&&y| y == 1).count();
    let n_down = features.y_class.iter().filter(|&&y| y == 2).count();
    let total = features.y_class.len();

    println!("  Class distribution:");
    println!("    UP:      {} ({:.1}%)", n_up, n_up as f64 / total as f64 * 100.0);
    println!("    NEUTRAL: {} ({:.1}%)", n_neutral, n_neutral as f64 / total as f64 * 100.0);
    println!("    DOWN:    {} ({:.1}%)", n_down, n_down as f64 / total as f64 * 100.0);

    // Initialize model
    println!("\n{} Initializing BNN...", "[3/4]".cyan());
    let config = BNNConfig::with_input_dim(features.x.ncols())
        .hidden_dims(vec![64, 32, 16])
        .num_classes(3)
        .n_samples_inference(50);

    let mut model = BayesianNN::new(config.clone());

    println!("  {} BNN with {} parameters",
        "Created".green(),
        model.num_parameters()
    );
    println!("  Architecture: {} -> {:?} -> {}",
        config.input_dim, config.hidden_dims, config.num_classes);

    // Training loop (simplified - real training would use proper gradient descent)
    println!("\n{} Training (demonstration)...", "[4/4]".cyan());

    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    let kl_schedule = KLSchedule::Linear { warmup_epochs: 10 };

    for epoch in 0..args.epochs {
        let kl_weight = kl_schedule.get_weight(epoch);

        // Forward pass with sampling
        let _output = model.forward(&features.x, true);
        let kl = model.kl_divergence();

        // Note: Real training would compute gradients and update parameters
        // This is a demonstration of the forward pass and KL computation

        if epoch % 10 == 0 {
            pb.println(format!(
                "  Epoch {}: KL={:.4}, KL_weight={:.2}",
                epoch, kl / features.x.nrows() as f64, kl_weight
            ));
        }

        pb.inc(1);
    }
    pb.finish_with_message("Training complete");

    // Evaluate on last samples
    println!("\n{}", "Model Predictions:".blue().bold());

    let test_samples = 5;
    let test_start = features.x.nrows() - test_samples;

    for i in 0..test_samples {
        let idx = test_start + i;
        let sample = features.x.slice(ndarray::s![idx..idx+1, ..]).to_owned();

        let estimate = model.predict_with_uncertainty(&sample);

        let actual_class = features.y_class[idx];
        let predicted_class = estimate.predicted_class();
        let confidence = estimate.mean_confidence();

        let class_names = ["UP", "NEUTRAL", "DOWN"];
        let correct = if actual_class as usize == predicted_class { "OK".green() } else { "X".red() };

        println!(
            "  Sample {}: Predicted={} ({:.1}%), Actual={}, Confidence={:.2} [{}]",
            i + 1,
            class_names[predicted_class],
            estimate.predicted_probability() * 100.0,
            class_names[actual_class as usize],
            confidence,
            correct
        );
    }

    // Show layer statistics
    println!("\n{}", "Layer Statistics:".blue().bold());
    for (i, stats) in model.get_layer_statistics().iter().enumerate() {
        println!(
            "  Layer {}: mu={:.4} (+/-{:.4}), sigma={:.4}",
            i, stats.weight_mu_mean, stats.weight_mu_std, stats.weight_sigma_mean
        );
    }

    println!("\n{}", "Training Demo Complete!".green().bold());
    println!("\nNote: This is a demonstration. Production training requires:");
    println!("  - Proper gradient computation");
    println!("  - ELBO loss optimization");
    println!("  - Train/validation split");
    println!("  - Learning rate scheduling");

    Ok(())
}
