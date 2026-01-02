//! Train Graph Attention Network
//!
//! This example demonstrates creating and using a GAT model
//! for cryptocurrency trading signals.
//!
//! Run with: cargo run --example train_gat

use anyhow::Result;
use gat_trading::gat::GraphAttentionNetwork;
use gat_trading::graph::{GraphBuilder, SparseGraph};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() -> Result<()> {
    println!("=== Graph Attention Network Training Demo ===\n");

    // Configuration
    let n_assets = 10;
    let n_features = 16;
    let hidden_dim = 32;
    let num_heads = 4;

    let symbols = vec![
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "NEARUSDT",
        "UNIUSDT", "AAVEUSDT", "DOGEUSDT", "SHIBUSDT", "ARBUSDT",
    ];

    println!("Configuration:");
    println!("  Assets: {}", n_assets);
    println!("  Input features: {}", n_features);
    println!("  Hidden dimension: {}", hidden_dim);
    println!("  Attention heads: {}", num_heads);

    // Create sample graph
    println!("\n1. Building asset graph...");
    let adj = GraphBuilder::sample_adjacency(n_assets);
    let graph = SparseGraph::from_dense(&adj);
    println!("   Graph: {} nodes, {} edges", graph.num_nodes(), graph.num_edges());

    // Create GAT
    println!("\n2. Creating Graph Attention Network...");
    let gat = GraphAttentionNetwork::new(n_features, hidden_dim, num_heads)?;
    println!("   Layers: {}", gat.num_layers());
    println!("   Parameters: {}", gat.num_parameters());

    // Create sample features
    println!("\n3. Generating sample features...");
    let features = Array2::random((n_assets, n_features), Uniform::new(-1.0, 1.0));
    println!("   Feature matrix shape: {:?}", features.dim());

    // Forward pass
    println!("\n4. Running forward pass...");
    let embeddings = gat.forward(&features, &graph);
    println!("   Output embeddings shape: {:?}", embeddings.dim());

    // Get attention weights
    println!("\n5. Computing attention weights...");
    let attention = gat.get_attention_weights(&features, &graph);
    println!("   Attention matrix shape: {:?}", attention.dim());

    // Show significant attention weights
    println!("\n   Significant attention weights (> 0.15):");
    for (i, from_symbol) in symbols.iter().enumerate() {
        for (j, to_symbol) in symbols.iter().enumerate() {
            if i != j && attention[[i, j]] > 0.15 {
                println!("   {} -> {}: {:.3}", from_symbol, to_symbol, attention[[i, j]]);
            }
        }
    }

    // Generate trading signals
    println!("\n6. Generating trading signals...");
    let signals = gat.predict_signals(&features, &graph);
    println!("\n   Trading signals:");
    for (i, symbol) in symbols.iter().enumerate() {
        let signal = signals[i];
        let action = if signal > 0.3 {
            "STRONG BUY"
        } else if signal > 0.1 {
            "BUY"
        } else if signal > -0.1 {
            "HOLD"
        } else if signal > -0.3 {
            "SELL"
        } else {
            "STRONG SELL"
        };
        println!("   {}: {:+.3} ({})", symbol, signal, action);
    }

    // Signal propagation example
    println!("\n7. Signal propagation example...");
    println!("   Simulating BTC bullish signal...");

    let mut modified_features = features.clone();
    modified_features[[0, 0]] = 1.5; // Strong bullish signal for BTC

    let new_signals = gat.predict_signals(&modified_features, &graph);

    println!("\n   Signal changes after BTC bullish event:");
    for (i, symbol) in symbols.iter().enumerate() {
        let old_signal = signals[i];
        let new_signal = new_signals[i];
        let change = new_signal - old_signal;
        if change.abs() > 0.01 {
            println!(
                "   {}: {:+.3} -> {:+.3} (change: {:+.3})",
                symbol, old_signal, new_signal, change
            );
        }
    }

    // Model serialization
    println!("\n8. Model serialization...");
    let json = gat.to_json()?;
    println!("   Serialized to JSON: {} bytes", json.len());

    let gat_restored = GraphAttentionNetwork::from_json(&json)?;
    println!("   Restored from JSON successfully");
    println!("   Parameters match: {}", gat.num_parameters() == gat_restored.num_parameters());

    // Custom architecture
    println!("\n9. Custom architecture example...");
    let custom_gat = GraphAttentionNetwork::with_layers(
        n_features,
        &[64, 32, 16],  // 3 layers
        2,               // 2 heads
    )?;
    println!("   Custom GAT: {} layers, {} parameters", custom_gat.num_layers(), custom_gat.num_parameters());

    println!("\n=== Demo Complete ===");
    println!("In production, use real market data and train with gradient descent.\n");

    Ok(())
}
