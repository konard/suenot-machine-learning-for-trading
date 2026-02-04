//! Example: Train a normalizing flow on financial returns
//!
//! This example demonstrates how to:
//! 1. Prepare return data
//! 2. Create a RealNVP flow
//! 3. Train the flow
//! 4. Generate samples from the learned distribution

use normalizing_flows_finance::prelude::*;
use normalizing_flows_finance::utils::returns::{generate_synthetic_returns, standardize};
use normalizing_flows_finance::utils::statistics::describe;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Normalizing Flows - Training Example");
    println!("═══════════════════════════════════════════════════════════");

    // Step 1: Prepare data
    println!("\n[1] Preparing training data...");

    // Try to fetch real data, fall back to synthetic
    let returns = match fetch_real_data() {
        Some(r) => {
            println!("    Using real BTC/USDT data");
            r
        }
        None => {
            println!("    Using synthetic fat-tailed data");
            generate_synthetic_returns(500, 4.0) // Student-t with df=4
        }
    };

    let stats = describe(&returns);
    println!("    Samples: {}", stats.count);
    println!("    Mean:    {:+.4}", stats.mean);
    println!("    Std:     {:.4}", stats.std);
    println!("    Kurt:    {:.2} (excess)", stats.kurtosis);

    // Standardize for training
    let standardized = standardize(&returns);

    // Step 2: Create model
    println!("\n[2] Creating RealNVP model...");
    let config = FlowConfig::new(1)
        .with_n_layers(6)
        .with_hidden_dim(64)
        .with_learning_rate(0.001);

    println!("    Dimension:     {}", config.dim);
    println!("    Layers:        {}", config.n_layers);
    println!("    Hidden dim:    {}", config.hidden_dim);
    println!("    Learning rate: {}", config.learning_rate);

    let mut flow = RealNVP::new(config);

    // Step 3: Train
    println!("\n[3] Training flow...");
    let history = flow.train(&standardized, 100)?;

    println!("\n    Training complete!");
    println!("    Epochs:           {}", history.n_epochs());
    println!("    Final train loss: {:.4}", history.final_train_loss().unwrap_or(0.0));
    println!("    Final val loss:   {:.4}", history.final_val_loss().unwrap_or(0.0));
    println!("    Best val loss:    {:.4}", history.best_val_loss().unwrap_or(0.0));

    // Step 4: Generate samples
    println!("\n[4] Generating samples from learned distribution...");
    let samples = flow.sample_flat(10000);

    let sample_stats = describe(&samples);
    println!("    Generated {} samples", sample_stats.count);
    println!("    Mean:    {:+.4} (should be ~0)", sample_stats.mean);
    println!("    Std:     {:.4} (should be ~1)", sample_stats.std);
    println!("    Kurt:    {:.2} (learned)", sample_stats.kurtosis);

    // Step 5: Compare distributions
    println!("\n[5] Comparing distributions...");
    println!("    {:^15} {:^15} {:^15}", "Metric", "Original", "Generated");
    println!("    {}", "-".repeat(50));

    let orig_stats = describe(&standardized);
    println!("    {:^15} {:>15.4} {:>15.4}", "Mean", orig_stats.mean, sample_stats.mean);
    println!("    {:^15} {:>15.4} {:>15.4}", "Std", orig_stats.std, sample_stats.std);
    println!("    {:^15} {:>15.4} {:>15.4}", "Skewness", orig_stats.skewness, sample_stats.skewness);
    println!("    {:^15} {:>15.4} {:>15.4}", "Kurtosis", orig_stats.kurtosis, sample_stats.kurtosis);
    println!("    {:^15} {:>15.4} {:>15.4}", "1st %ile", orig_stats.q1, sample_stats.q1);
    println!("    {:^15} {:>15.4} {:>15.4}", "Median", orig_stats.median, sample_stats.median);
    println!("    {:^15} {:>15.4} {:>15.4}", "3rd %ile", orig_stats.q3, sample_stats.q3);

    // Step 6: Log probability check
    println!("\n[6] Log probability verification...");
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];
    println!("    {:^10} {:^15}", "x", "log p(x)");
    println!("    {}", "-".repeat(30));

    for &x in &test_points {
        let input = ndarray::Array1::from_vec(vec![x]);
        let log_prob = flow.log_prob(&input);
        println!("    {:^10.1} {:>15.4}", x, log_prob);
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Training example complete!");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}

fn fetch_real_data() -> Option<Vec<f64>> {
    let client = BybitClient::new();

    match client.get_klines_sync("BTCUSDT", "D", 500) {
        Ok(klines) => {
            let returns = calculate_log_returns(&klines);
            if returns.len() > 100 {
                Some(returns)
            } else {
                None
            }
        }
        Err(_) => None,
    }
}
