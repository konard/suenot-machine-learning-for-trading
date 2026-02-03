//! Basic attention visualization example.
//!
//! This example demonstrates how to:
//! 1. Create a Transformer model
//! 2. Run a forward pass
//! 3. Extract and visualize attention weights

use attention_visualization_trading::{
    AttentionTransformer,
    BybitClient,
    FeatureEngineering,
};
use attention_visualization_trading::model::TransformerConfig;

fn main() {
    println!("=== Attention Visualization Example ===\n");

    // Create model configuration
    let config = TransformerConfig {
        input_dim: 8,
        d_model: 64,
        n_heads: 4,
        n_layers: 3,
        d_ff: 256,
        dropout: 0.1,
        max_seq_len: 512,
    };

    println!("Model Configuration:");
    println!("  Input dim: {}", config.input_dim);
    println!("  Model dim: {}", config.d_model);
    println!("  Heads: {}", config.n_heads);
    println!("  Layers: {}", config.n_layers);
    println!();

    // Create model
    let model = AttentionTransformer::new(config);

    // Generate synthetic data
    let client = BybitClient::new("BTCUSDT", "60");
    let candles = client.generate_synthetic_data(100);
    println!("Generated {} synthetic candles", candles.len());

    // Prepare features
    let features = FeatureEngineering::prepare_features(&candles);
    println!("Prepared {} feature vectors with {} features each",
        features.len(),
        features.first().map(|f| f.len()).unwrap_or(0)
    );

    // Create a sequence (last 60 candles)
    let seq_len = 60;
    let sequence: Vec<Vec<f64>> = features[features.len() - seq_len..].to_vec();
    println!("\nRunning forward pass with sequence length {}...", seq_len);

    // Forward pass
    let (pred_return, pred_direction, pred_volatility, attention) = model.forward(&sequence);

    println!("\n=== Predictions ===");
    println!("  Predicted Return: {:.4}", pred_return);
    println!("  Direction Probability: {:.4}", pred_direction);
    println!("  Predicted Volatility: {:.4}", pred_volatility);

    // Analyze attention
    println!("\n=== Attention Analysis ===");
    println!("  Number of layers: {}", attention.layers.len());

    let entropy = attention.compute_entropy();
    let confidence = attention.compute_confidence(seq_len);
    println!("  Average entropy: {:.4}", entropy);
    println!("  Confidence score: {:.4}", confidence);

    // Get last layer attention
    if let Some(last_attn) = attention.get_last_layer_attention() {
        println!("\n=== Last Layer Attention Pattern ===");
        println!("  Shape: {}x{}", last_attn.len(), last_attn.first().map(|r| r.len()).unwrap_or(0));

        // Print attention from last position to first 10 positions
        println!("\n  Attention from last position to first 10 positions:");
        let last_row = &last_attn[last_attn.len() - 1];
        for (i, &w) in last_row.iter().take(10).enumerate() {
            let bar_len = (w * 50.0) as usize;
            let bar: String = "█".repeat(bar_len);
            println!("    t-{}: {:.4} {}", seq_len - i, w, bar);
        }

        // Find top attended positions
        let mut indexed: Vec<(usize, f64)> = last_row.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\n  Top 5 attended positions:");
        for (i, (pos, weight)) in indexed.iter().take(5).enumerate() {
            println!("    {}. Position t-{}: {:.4}", i + 1, seq_len - pos, weight);
        }
    }

    println!("\n=== Example Complete ===");
}
