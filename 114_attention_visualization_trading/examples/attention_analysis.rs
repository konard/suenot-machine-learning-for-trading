//! Comprehensive attention analysis example.
//!
//! This example demonstrates advanced attention analysis techniques:
//! 1. Layer-wise attention comparison
//! 2. Head specialization analysis
//! 3. Attention pattern classification

use attention_visualization_trading::{
    AttentionTransformer,
    BybitClient,
    FeatureEngineering,
};
use attention_visualization_trading::model::TransformerConfig;

fn main() {
    println!("=== Comprehensive Attention Analysis ===\n");

    // Create model
    let config = TransformerConfig {
        input_dim: 8,
        d_model: 64,
        n_heads: 4,
        n_layers: 3,
        ..Default::default()
    };
    let model = AttentionTransformer::new(config.clone());

    // Generate data
    let client = BybitClient::new("BTCUSDT", "60");
    let candles = client.generate_synthetic_data(200);
    let features = FeatureEngineering::prepare_features(&candles);

    // Create sequences
    let seq_len = 60;
    let (sequences, _) = FeatureEngineering::create_sequences(&features, seq_len);
    println!("Created {} sequences of length {}\n", sequences.len(), seq_len);

    // Analyze multiple samples
    let n_samples = 10;
    let mut all_confidences = Vec::new();
    let mut all_entropies = Vec::new();

    println!("=== Analyzing {} Samples ===\n", n_samples);

    for i in 0..n_samples.min(sequences.len()) {
        let (pred_ret, pred_dir, _pred_vol, attention) = model.forward(&sequences[i]);

        let confidence = attention.compute_confidence(seq_len);
        let entropy = attention.compute_entropy();

        all_confidences.push(confidence);
        all_entropies.push(entropy);

        let signal = if pred_dir > 0.6 && pred_ret > 0.001 {
            "LONG"
        } else if pred_dir < 0.4 && pred_ret < -0.001 {
            "SHORT"
        } else {
            "HOLD"
        };

        println!(
            "Sample {}: Return={:+.4}, Direction={:.2}, Confidence={:.2}, Signal={}",
            i + 1, pred_ret, pred_dir, confidence, signal
        );
    }

    // Summary statistics
    println!("\n=== Summary Statistics ===");

    let avg_confidence: f64 = all_confidences.iter().sum::<f64>() / all_confidences.len() as f64;
    let avg_entropy: f64 = all_entropies.iter().sum::<f64>() / all_entropies.len() as f64;

    let max_confidence = all_confidences.iter().cloned().fold(0.0_f64, f64::max);
    let min_confidence = all_confidences.iter().cloned().fold(1.0_f64, f64::min);

    println!("  Average Confidence: {:.4}", avg_confidence);
    println!("  Min Confidence: {:.4}", min_confidence);
    println!("  Max Confidence: {:.4}", max_confidence);
    println!("  Average Entropy: {:.4}", avg_entropy);

    // Layer-wise analysis (using last sample)
    println!("\n=== Layer-wise Attention Analysis ===");

    let (_, _, _, attention) = model.forward(&sequences[0]);
    for (layer_idx, layer_attn) in attention.layers.iter().enumerate() {
        // Calculate average attention focus for this layer
        let mut layer_entropy = 0.0;
        let mut count = 0;

        for batch in layer_attn {
            for head in batch {
                for row in head {
                    let entropy: f64 = row
                        .iter()
                        .filter(|&&w| w > 1e-9)
                        .map(|&w| -w * w.ln())
                        .sum();
                    layer_entropy += entropy;
                    count += 1;
                }
            }
        }

        let avg_layer_entropy = if count > 0 {
            layer_entropy / count as f64
        } else {
            0.0
        };

        let max_entropy = (seq_len as f64).ln();
        let layer_focus = 1.0 - (avg_layer_entropy / max_entropy);

        println!("  Layer {}: Entropy={:.4}, Focus={:.4}", layer_idx, avg_layer_entropy, layer_focus);
    }

    // Head pattern analysis
    println!("\n=== Head Pattern Analysis (Last Layer) ===");

    if let Some(last_layer) = attention.layers.last() {
        if let Some(batch) = last_layer.first() {
            for (head_idx, head_attn) in batch.iter().enumerate() {
                // Calculate diagonal dominance (local attention)
                let mut diag_sum = 0.0;
                for i in 0..head_attn.len() {
                    diag_sum += head_attn[i][i];
                }
                let diag_ratio = diag_sum / head_attn.len() as f64;

                // Calculate attention to recent positions
                let recent_window = 5;
                let mut recent_sum = 0.0;
                let last_row = &head_attn[head_attn.len() - 1];
                for i in (last_row.len().saturating_sub(recent_window))..last_row.len() {
                    recent_sum += last_row[i];
                }

                let pattern = if diag_ratio > 0.3 {
                    "Local (diagonal)"
                } else if recent_sum > 0.5 {
                    "Recent focus"
                } else {
                    "Distributed"
                };

                println!(
                    "  Head {}: Diagonal={:.2}, Recent={:.2}, Pattern: {}",
                    head_idx, diag_ratio, recent_sum, pattern
                );
            }
        }
    }

    println!("\n=== Analysis Complete ===");
}
