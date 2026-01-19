//! Example: Making predictions with the Informer model
//!
//! Run with: cargo run --example predict

use informer_probsparse::{InformerConfig, InformerModel};
use ndarray::Array3;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Informer ProbSparse: Prediction Example ===\n");

    // Create model
    let config = InformerConfig {
        seq_len: 48,
        pred_len: 12,
        input_features: 6,
        d_model: 32,
        n_heads: 4,
        d_ff: 64,
        n_encoder_layers: 2,
        sampling_factor: 5.0,
        use_distilling: true,
        ..Default::default()
    };

    let model = InformerModel::new(config.clone());

    println!("Model Configuration:");
    println!("  Input: {} timesteps x {} features", config.seq_len, config.input_features);
    println!("  Output: {} timesteps prediction", config.pred_len);
    println!("  Architecture: ProbSparse Attention with distilling");

    // Create sample input (batch of 2)
    println!("\n--- Generating Sample Input ---\n");

    let x = create_sample_input(2, config.seq_len, config.input_features);
    println!("Input shape: [{}, {}, {}]", x.dim().0, x.dim().1, x.dim().2);

    // Make prediction
    println!("\n--- Making Predictions ---\n");

    let predictions = model.predict(&x);

    println!("Predictions shape: [{}, {}]", predictions.dim().0, predictions.dim().1);

    // Show predictions
    for batch in 0..predictions.dim().0 {
        println!("\nBatch {}:", batch);
        print!("  Predictions (next {} steps): [", config.pred_len);
        for t in 0..config.pred_len {
            print!("{:.4}", predictions[[batch, t]]);
            if t < config.pred_len - 1 { print!(", "); }
        }
        println!("]");
    }

    // Prediction with attention weights
    println!("\n--- Predictions with Attention Weights ---\n");

    let (predictions, attention_weights) = model.forward(&x, true);

    println!("Number of attention weight layers: {}", attention_weights.len());

    for (i, weights) in attention_weights.iter().enumerate() {
        if let Some(ref tw) = weights.temporal_weights {
            let dims = tw.dim();
            println!("Layer {} attention shape: [{}, {}, {}, {}]",
                i, dims.0, dims.1, dims.2, dims.3);
        }

        // Show top attended positions
        let top_positions = weights.top_k_positions(3);
        if !top_positions.is_empty() && !top_positions[0].is_empty() {
            println!("  Position 0 attends most to: {:?}", top_positions[0]);
        }
    }

    // Interpret predictions
    println!("\n--- Prediction Interpretation ---\n");

    for batch in 0..predictions.dim().0 {
        let preds: Vec<f64> = (0..config.pred_len)
            .map(|t| predictions[[batch, t]])
            .collect();

        let avg_prediction = preds.iter().sum::<f64>() / preds.len() as f64;

        let signal = if avg_prediction > 0.001 {
            "BULLISH (expected positive returns)"
        } else if avg_prediction < -0.001 {
            "BEARISH (expected negative returns)"
        } else {
            "NEUTRAL (no clear direction)"
        };

        println!("Batch {}: {}", batch, signal);
        println!("  Average predicted return: {:.4}%", avg_prediction * 100.0);
    }

    println!("\n=== Prediction Complete ===");

    Ok(())
}

/// Create sample input data
fn create_sample_input(batch_size: usize, seq_len: usize, n_features: usize) -> Array3<f64> {
    use std::f64::consts::PI;

    Array3::from_shape_fn((batch_size, seq_len, n_features), |(b, t, f)| {
        let base = (t as f64 * 0.1 + b as f64 * 0.5).sin();
        let noise = rand_normal() * 0.1;
        let feature_offset = (f as f64 * PI / 6.0).cos() * 0.2;
        base + noise + feature_offset
    })
}

/// Generate random normal number
fn rand_normal() -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
