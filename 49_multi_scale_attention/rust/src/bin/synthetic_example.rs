//! Synthetic data example for Multi-Scale Attention
//!
//! This example demonstrates the complete workflow with synthetic data:
//! 1. Generate synthetic multi-scale data
//! 2. Build the model
//! 3. Make predictions
//! 4. Backtest a trading strategy

use multi_scale_attention::{
    generate_synthetic_data,
    MultiScaleAttention,
    MultiScaleAttentionConfig,
    ScaleConfig,
    backtest_multi_scale_strategy,
    BacktestConfig,
};
use std::collections::HashMap;

fn main() {
    println!("================================================================================");
    println!("   MULTI-SCALE ATTENTION - SYNTHETIC DATA EXAMPLE");
    println!("================================================================================");

    // Configuration
    let scales = ["1min", "5min", "1H"];
    let lookbacks: HashMap<String, usize> = [
        ("1min".to_string(), 60),
        ("5min".to_string(), 24),
        ("1H".to_string(), 12),
    ]
    .into_iter()
    .collect();
    let n_features = 15;

    // Step 1: Generate synthetic data
    println!("\n1. Generating synthetic data...");
    let n_samples = 500;
    let (sequences, targets) = generate_synthetic_data(n_samples, &scales, n_features, &lookbacks);

    println!("   Generated {} samples", n_samples);
    for scale in &scales {
        let shape = sequences.get(*scale).map(|s| s.dim()).unwrap_or((0, 0, 0));
        println!("   Scale {}: shape {:?}", scale, shape);
    }

    // Step 2: Create the model
    println!("\n2. Creating multi-scale attention model...");
    let config = MultiScaleAttentionConfig {
        scale_configs: vec![
            ScaleConfig {
                name: "1min".to_string(),
                input_dim: n_features,
                seq_len: 60,
            },
            ScaleConfig {
                name: "5min".to_string(),
                input_dim: n_features,
                seq_len: 24,
            },
            ScaleConfig {
                name: "1H".to_string(),
                input_dim: n_features,
                seq_len: 12,
            },
        ],
        d_model: 64,
        n_heads: 4,
        n_encoder_layers: 2,
        dropout: 0.1,
        output_dim: 1,
    };

    let model = MultiScaleAttention::new(config);
    println!("   Model created with {} scales", model.scale_names().len());

    // Step 3: Make predictions (inference only - no training in this example)
    println!("\n3. Making predictions...");
    let predictions = model.forward(&sequences);
    println!("   Predictions shape: ({}, {})", predictions.direction.nrows(), predictions.direction.ncols());

    // Show some sample predictions
    println!("\n   Sample predictions (first 5):");
    println!("   {:>10} {:>12} {:>12} {:>12}", "Direction", "Short-term", "Med-term", "Uncertainty");
    for i in 0..5.min(predictions.direction.nrows()) {
        println!(
            "   {:>10.4} {:>12.6} {:>12.6} {:>12.6}",
            predictions.direction[[i, 0]],
            predictions.short_term[[i, 0]],
            predictions.medium_term[[i, 0]],
            predictions.uncertainty[[i, 0]]
        );
    }

    // Step 4: Get scale importance
    println!("\n4. Scale importance weights:");
    let importance = model.get_scale_importance();
    for (scale, weight) in &importance {
        println!("   {}: {:.4}", scale, weight);
    }

    // Step 5: Backtest with synthetic prices
    println!("\n5. Backtesting strategy...");

    // Generate synthetic prices (random walk)
    let mut prices = Vec::with_capacity(n_samples);
    let mut price = 100.0;
    for i in 0..n_samples {
        // Use target as the return for synthetic price generation
        let return_val = if i < targets.len() { targets[i] } else { 0.0 };
        price *= 1.0 + return_val;
        prices.push(price);
    }

    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        transaction_cost: 0.001,
        confidence_threshold: 0.55,
        max_position_size: 1.0,
        stop_loss: None,
        take_profit: None,
    };

    let result = backtest_multi_scale_strategy(&predictions, &prices, &backtest_config);
    println!("{}", result.report());

    // Step 6: Summary statistics
    println!("6. Direction prediction distribution:");
    let long_signals = predictions.direction.iter().filter(|&&d| d > 0.5).count();
    let short_signals = predictions.direction.iter().filter(|&&d| d <= 0.5).count();
    println!("   Long signals (>0.5):  {}", long_signals);
    println!("   Short signals (<=0.5): {}", short_signals);

    println!("\n================================================================================");
    println!("   EXAMPLE COMPLETED SUCCESSFULLY");
    println!("================================================================================");
}
