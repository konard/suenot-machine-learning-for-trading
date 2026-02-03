//! Basic example: Create and run the SSM-Transformer Hybrid model
//! on synthetic data.

use ssm_transformer_hybrid::prelude::*;
use ssm_transformer_hybrid::data::bybit::generate_synthetic_klines;
use ssm_transformer_hybrid::data::features::{FeatureGenerator, FeatureRow};

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== SSM-Transformer Hybrid Basic Example ===\n");

    // Create model: 15 features, 32-dim hidden, 4 SSM + 1 attention layers
    let model = SSMTransformerModel::new(
        FeatureRow::feature_count(), // 15 features
        32,                          // d_model
        4,                           // SSM layers
        1,                           // Attention layers
        8,                           // SSM state size
        4,                           // Attention heads
    );

    println!("Model: {}", model.summary());
    println!("Parameters: {}\n", model.num_parameters());

    // Generate synthetic data
    let klines = generate_synthetic_klines(200);
    println!("Generated {} synthetic klines", klines.len());

    // Compute features
    let feature_gen = FeatureGenerator::new();
    let features = feature_gen.generate(&klines);

    let valid_features: Vec<Vec<f64>> = features
        .iter()
        .filter_map(|f| f.as_ref().map(|r| r.to_vec()))
        .collect();

    println!("Valid feature rows: {}\n", valid_features.len());

    // Run model on last 50 steps
    let seq_len = 50;
    if valid_features.len() >= seq_len {
        let sequence = &valid_features[valid_features.len() - seq_len..];
        let preds = model.forward(sequence);

        println!("Predictions for last {} steps:", seq_len);
        println!("  Direction probability: {:.4}", preds.direction);
        println!("  Predicted volatility:  {:.6}", preds.volatility);
        println!("  Predicted return:      {:.6}", preds.return_magnitude);

        // Interpret
        if preds.direction > 0.55 {
            println!("\n  Signal: LONG (confidence: {:.1}%)", (preds.direction - 0.5) * 200.0);
        } else if preds.direction < 0.45 {
            println!("\n  Signal: SHORT (confidence: {:.1}%)", (0.5 - preds.direction) * 200.0);
        } else {
            println!("\n  Signal: FLAT (no position)");
        }
    }

    println!("\nDone!");
}
