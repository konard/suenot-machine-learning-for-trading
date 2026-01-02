//! Training script for E-GNN model (simplified demo)
//!
//! Example: cargo run --bin train_egnn

use equivariant_gnn_trading::{EquivariantGNN, MarketGraph, Candle};
use std::collections::HashMap;

fn main() {
    println!("=== E-GNN Training Demo ===\n");

    // Generate synthetic training data
    println!("Generating synthetic market data...");
    let symbols = vec!["BTC", "ETH", "SOL", "BNB", "XRP"];
    let mut candles_map = HashMap::new();

    for (i, symbol) in symbols.iter().enumerate() {
        let mut candles = Vec::new();
        let base = 100.0 * (i + 1) as f64;

        for t in 0..500 {
            let trend = (t as f64 * 0.02).sin() * 10.0;
            let noise = rand::random::<f64>() * 2.0 - 1.0;
            let corr = if i > 0 { (t as f64 * 0.02).sin() * 5.0 } else { 0.0 };

            let close = base + trend + noise + corr;
            candles.push(Candle::new(
                t as u64 * 3600000,
                close - noise.abs(),
                close + 1.0 + noise.abs(),
                close - 1.0 - noise.abs(),
                close,
                1000.0 + noise * 100.0,
                close * 1000.0,
            ));
        }
        candles_map.insert(symbol.to_string(), candles);
    }

    // Build graph
    let graph_builder = MarketGraph::new(0.3);
    let graph = graph_builder.from_candles(&candles_map);
    println!("Graph: {} nodes, {} edges\n", graph.num_nodes(), graph.num_edges());

    // Create model
    let model = EquivariantGNN::new(graph.node_feature_dim(), 64, 3, 4);

    // Training loop (simplified - just forward passes)
    println!("Training (simplified demo)...\n");

    for epoch in 0..10 {
        let output = model.forward(&graph);

        // Compute mock loss (direction entropy)
        let mut loss = 0.0;
        for i in 0..graph.num_nodes() {
            for j in 0..3 {
                let p = output.direction_probs[[i, j]];
                if p > 1e-10 { loss -= p * p.ln(); }
            }
        }
        loss /= graph.num_nodes() as f64;

        if epoch % 2 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch + 1, loss);
        }
    }

    // Final predictions
    println!("\n=== Final Predictions ===\n");

    let output = model.forward(&graph);
    for (i, node) in graph.nodes.iter().enumerate() {
        let signals = model.signals_from_output(&output, 0.35);
        let signal_str = match signals[i] {
            1 => "LONG",
            -1 => "SHORT",
            _ => "HOLD",
        };
        println!("{}: {} (size: {:.1}%)", node.symbol, signal_str, output.position_sizes[i] * 100.0);
    }

    println!("\nTraining demo complete!");
    println!("Note: This is a simplified demo. Full training requires gradient computation.");
}
