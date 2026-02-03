//! Example: Saliency map computation for trading
//!
//! This example demonstrates how to compute saliency maps
//! and use them for trading signal filtering.

use ndarray::Array2;
use saliency_maps_trading::{
    models::network::TradingNetwork,
    saliency::gradient::{SaliencyComputer, SaliencyTrader},
    api::bybit::{BybitClient, Interval},
    data::processor::DataProcessor,
};

fn main() {
    env_logger::init();

    println!("Saliency Maps for Trading");
    println!("=========================\n");

    // Configuration
    let sequence_length = 30;
    let num_features = 5;  // OHLCV
    let hidden_sizes = vec![64, 32];

    // Create network
    println!("Creating trading network...");
    let network = TradingNetwork::new(sequence_length, num_features, &hidden_sizes);
    println!("  Input size: {} ({}x{})",
             network.input_size(), sequence_length, num_features);

    // Create saliency computer
    let saliency_computer = SaliencyComputer::new(network.clone());

    // Create sample input (simulated normalized data)
    println!("\nCreating sample input data...");
    let sample_input = Array2::from_shape_fn((sequence_length, num_features), |(i, j)| {
        // Simulate normalized price data with some pattern
        let trend = i as f64 / sequence_length as f64;
        let feature_offset = j as f64 * 0.1;
        trend + feature_offset + (i as f64 * 0.5).sin() * 0.1
    });

    println!("  Shape: {:?}", sample_input.dim());

    // Compute saliency maps using different methods
    println!("\nComputing saliency maps...\n");

    // 1. Vanilla gradient
    println!("1. Vanilla Gradient:");
    let vanilla = saliency_computer.vanilla_gradient(&sample_input);
    print_saliency_summary(&vanilla, "vanilla");

    // 2. Gradient × Input
    println!("\n2. Gradient × Input:");
    let grad_input = saliency_computer.gradient_x_input(&sample_input);
    print_saliency_summary(&grad_input, "grad_x_input");

    // 3. Integrated Gradients
    println!("\n3. Integrated Gradients:");
    let integrated = saliency_computer.integrated_gradients(&sample_input, 20);
    print_saliency_summary(&integrated, "integrated");

    // 4. SmoothGrad
    println!("\n4. SmoothGrad:");
    let smooth = saliency_computer.smoothgrad(&sample_input, 0.1, 10);
    print_saliency_summary(&smooth, "smoothgrad");

    // Analyze feature importance
    println!("\n{:-<60}", "");
    println!("Feature Importance Analysis (Gradient × Input method):");
    println!("{:-<60}", "");

    let feature_names = ["Open", "High", "Low", "Close", "Volume"];
    let feature_imp = saliency_computer.feature_importance(&grad_input);

    let mut indexed: Vec<(usize, f64)> = feature_imp.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, importance) in &indexed {
        let bar_len = (importance * 50.0) as usize;
        let bar: String = "█".repeat(bar_len.min(50));
        println!("  {:8}: {:6.4} {}", feature_names[*idx], importance, bar);
    }

    // Analyze temporal importance
    println!("\nTemporal Importance (last 10 time steps):");
    let temporal_imp = saliency_computer.temporal_importance(&grad_input);
    let last_10: Vec<f64> = temporal_imp.iter().rev().take(10).cloned().collect();

    for (i, imp) in last_10.iter().rev().enumerate() {
        let bar_len = (imp * 100.0) as usize;
        let bar: String = "▓".repeat(bar_len.min(40));
        println!("  t-{:2}: {:6.4} {}", 10 - i, imp, bar);
    }

    // Concentration analysis
    println!("\n{:-<60}", "");
    println!("Saliency Concentration Analysis:");
    println!("{:-<60}", "");

    let concentration = saliency_computer.concentration(&grad_input);
    println!("  Concentration score: {:.4}", concentration);
    println!("  Interpretation: {}",
             if concentration > 0.5 { "Model focuses on specific features" }
             else if concentration > 0.3 { "Moderate feature focus" }
             else { "Diffuse attention across features" });

    // Trading signal generation
    println!("\n{:-<60}", "");
    println!("Trading Signal Generation:");
    println!("{:-<60}", "");

    // Create saliency-based trader
    let trader = SaliencyTrader::new(
        network,
        0.55,  // min confidence
        0.3,   // min concentration
        vec![3, 4],  // interpretable features: Close, Volume
    );

    let (signal, position_size) = trader.generate_signal(&sample_input);

    println!("  Prediction signal: {}", match signal {
        1 => "LONG",
        -1 => "SHORT",
        _ => "NO TRADE"
    });
    println!("  Position size: {:.2}%", position_size * 100.0);

    println!("\n✓ Saliency analysis complete!");
}

fn print_saliency_summary(saliency: &Array2<f64>, method: &str) {
    let sum: f64 = saliency.iter().sum();
    let mean = sum / saliency.len() as f64;
    let max = saliency.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = saliency.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  Sum: {:.6}", sum);
    println!("  Mean: {:.6}", mean);
    println!("  Range: [{:.6}, {:.6}]", min, max);
}
