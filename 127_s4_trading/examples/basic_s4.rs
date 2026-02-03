//! Basic S4 example demonstrating model usage.
//!
//! This example shows how to:
//! - Create an S4 model
//! - Generate synthetic data
//! - Make predictions

use s4_trading::prelude::*;
use s4_trading::data::bybit::generate_synthetic_data;
use s4_trading::model::s4::{S4Config, S4Layer, S4DLayer};

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== S4 Trading: Basic Example ===\n");

    // 1. Demonstrate S4 layer basics
    println!("1. S4 Layer Demonstration");
    println!("   Creating S4 layer with d_state=32...");

    let layer = S4Layer::new(1, 32);
    println!("   Step size (dt): {:.6}", layer.dt());

    // Generate a simple sine wave input
    let input: Vec<f64> = (0..100)
        .map(|i| (i as f64 * 0.1).sin())
        .collect();

    let output = layer.forward(&input);
    println!("   Input length: {}", input.len());
    println!("   Output length: {}", output.len());
    println!("   First 5 outputs: {:?}", &output[..5]);
    println!();

    // 2. Demonstrate S4D (diagonal) layer
    println!("2. S4D (Diagonal) Layer Demonstration");
    let s4d_layer = S4DLayer::new(32);
    let s4d_output = s4d_layer.forward(&input);
    println!("   S4D output (first 5): {:?}", &s4d_output[..5]);
    println!();

    // 3. Create and use full S4 model
    println!("3. Full S4 Model for Trading");

    let config = S4Config {
        d_model: 64,
        d_state: 64,
        n_layers: 4,
        dropout: 0.1,
        bidirectional: false,
        dt_min: 0.001,
        dt_max: 0.1,
    };

    println!("   Model config:");
    println!("     d_model: {}", config.d_model);
    println!("     d_state: {}", config.d_state);
    println!("     n_layers: {}", config.n_layers);

    let model = s4_trading::model::s4::S4Model::new(config);

    // 4. Generate synthetic market data
    println!("\n4. Generating Synthetic Market Data");
    let data = generate_synthetic_data("BTCUSDT", "60", 500);

    println!("   Symbol: {}", data.symbol);
    println!("   Interval: {}", data.interval);
    println!("   Data points: {}", data.candles.len());
    println!("   Source: {}", data.source);

    if let Some(first) = data.candles.first() {
        println!("   First candle: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
            first.open, first.high, first.low, first.close, first.volume);
    }
    if let Some(last) = data.candles.last() {
        println!("   Last candle:  O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
            last.open, last.high, last.low, last.close, last.volume);
    }

    // 5. Make predictions
    println!("\n5. Making Predictions");

    let features = data.to_features();
    let signal = model.predict(&features);

    println!("   Signal: {}", signal);
    println!("   Confidence: {:.2}%", signal.confidence() * 100.0);
    println!("   Direction: {}", match signal.direction() {
        1 => "LONG",
        -1 => "SHORT",
        _ => "NEUTRAL",
    });

    // 6. Analyze model state
    println!("\n6. Model State Analysis");
    let state = model.get_state(&features);

    let state_mean: f64 = state.iter().sum::<f64>() / state.len() as f64;
    let state_std: f64 = (state.iter()
        .map(|s| (s - state_mean).powi(2))
        .sum::<f64>() / state.len() as f64)
        .sqrt();

    println!("   Hidden state dimension: {}", state.len());
    println!("   State mean: {:.6}", state_mean);
    println!("   State std: {:.6}", state_std);
    println!("   State range: [{:.6}, {:.6}]",
        state.iter().cloned().fold(f64::INFINITY, f64::min),
        state.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // 7. Compute SSM kernel
    println!("\n7. SSM Kernel Analysis");
    let kernel = layer.compute_kernel(50);

    println!("   Kernel length: {}", kernel.len());
    println!("   First 10 kernel values: {:?}",
        &kernel[..10].iter().map(|k| format!("{:.4}", k)).collect::<Vec<_>>());

    // Check kernel decay (should decay for stable system)
    let kernel_decay = kernel[kernel.len()-1].abs() / kernel[0].abs().max(1e-10);
    println!("   Kernel decay ratio: {:.6}", kernel_decay);

    println!("\n=== Example Complete ===");
}
