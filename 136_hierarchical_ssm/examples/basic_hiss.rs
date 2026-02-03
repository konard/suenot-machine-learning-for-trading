//! Basic example: create and run a Hierarchical SSM on synthetic data.

use hierarchical_ssm::data::bybit::generate_synthetic_candles;
use hierarchical_ssm::data::features::{compute_features, normalize_features};
use hierarchical_ssm::model::{HiSSConfig, HierarchicalSSM};

fn main() {
    println!("=== Hierarchical SSM (HiSS) - Basic Example ===\n");

    // Configure the model
    let config = HiSSConfig {
        input_dim: 8,
        hidden_dim: 32,
        output_dim: 3,
        num_levels: 3,
        downsample_factors: vec![4, 4],
        ssm_state_dim: 8,
        dropout: 0.1,
    };

    println!("Model Configuration:");
    println!("  Input dim:    {}", config.input_dim);
    println!("  Hidden dim:   {}", config.hidden_dim);
    println!("  Num levels:   {}", config.num_levels);
    println!("  Downsample:   {:?}", config.downsample_factors);
    println!("  SSM state:    {}", config.ssm_state_dim);
    println!();

    // Create model
    let model = HierarchicalSSM::new(config);

    // Generate synthetic BTC-like price data
    println!("Generating synthetic candle data (200 candles, base price $50,000)...");
    let candles = generate_synthetic_candles(200, 50000.0);
    println!("  Generated {} candles", candles.len());
    println!(
        "  Price range: ${:.2} - ${:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
    );

    // Compute features
    let features = compute_features(&candles);
    println!("  Computed {} feature rows", features.len());

    // Normalize
    let normalized = normalize_features(&features);
    println!("  Normalized to zero mean, unit variance");
    println!();

    // Run predictions on sliding windows
    let window_size = 64;
    let num_predictions = normalized.len().saturating_sub(window_size);
    println!(
        "Running {} predictions with window size {}...\n",
        num_predictions, window_size
    );

    let mut long_count = 0;
    let mut short_count = 0;
    let mut flat_count = 0;

    for i in 0..num_predictions.min(10) {
        let window = &normalized[i..i + window_size];
        let pred = model.predict(window);

        let signal = if pred.direction_probability > 0.6 && pred.predicted_return > 0.001 {
            long_count += 1;
            "LONG"
        } else if pred.direction_probability < 0.4 && pred.predicted_return < -0.001 {
            short_count += 1;
            "SHORT"
        } else {
            flat_count += 1;
            "FLAT"
        };

        println!(
            "  Step {:3}: return={:+.6}, dir_prob={:.4}, vol={:.6} -> {}",
            i, pred.predicted_return, pred.direction_probability, pred.predicted_volatility, signal
        );
    }

    // Run remaining predictions silently
    for i in 10..num_predictions {
        let window = &normalized[i..i + window_size];
        let pred = model.predict(window);

        if pred.direction_probability > 0.6 && pred.predicted_return > 0.001 {
            long_count += 1;
        } else if pred.direction_probability < 0.4 && pred.predicted_return < -0.001 {
            short_count += 1;
        } else {
            flat_count += 1;
        }
    }

    println!("\n--- Signal Distribution ---");
    println!("  LONG:  {} ({:.1}%)", long_count, 100.0 * long_count as f64 / num_predictions.max(1) as f64);
    println!("  SHORT: {} ({:.1}%)", short_count, 100.0 * short_count as f64 / num_predictions.max(1) as f64);
    println!("  FLAT:  {} ({:.1}%)", flat_count, 100.0 * flat_count as f64 / num_predictions.max(1) as f64);
    println!("\nDone!");
}
