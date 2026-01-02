//! Basic MPNN example with synthetic data.
//!
//! This example demonstrates:
//! - Creating a market graph
//! - Building an MPNN model
//! - Running forward passes
//! - Generating trading signals

use mpnn_trading::{
    graph::MarketGraph,
    mpnn::{AggregationType, MPNN, MPNNConfig},
    strategy::{MPNNStrategy, Signal},
};
use ndarray::Array1;
use std::collections::HashMap;

fn main() {
    println!("=== Basic MPNN Example ===\n");

    // Step 1: Create a market graph with synthetic data
    println!("1. Creating market graph...");
    let mut graph = create_sample_graph();
    println!("   Created graph with {} nodes and {} edges\n",
        graph.node_count(), graph.edge_count());

    // Step 2: Create MPNN model
    println!("2. Creating MPNN model...");
    let config = MPNNConfig {
        input_dim: 8, // 8 features per node
        hidden_dim: 32,
        output_dim: 16,
        num_layers: 3,
        aggregation: AggregationType::Mean,
        dropout: 0.1,
        use_edge_features: false,
        num_heads: 4,
    };
    let mpnn = MPNN::from_config(config);
    println!("   Model: {} layers, {} hidden dim\n", 3, 32);

    // Step 3: Run forward pass
    println!("3. Running forward pass...");
    let output = mpnn.forward(&mut graph).expect("Forward pass failed");
    println!("   Output shape: {:?}\n", output.dim());

    // Step 4: Get node embeddings
    println!("4. Extracting node embeddings...");
    let embeddings = mpnn.get_embeddings(&mut graph).expect("Embedding extraction failed");
    println!("   Embedding shape: {:?}\n", embeddings.dim());

    // Step 5: Generate trading signals
    println!("5. Generating trading signals...");
    let strategy = MPNNStrategy::new(mpnn)
        .with_thresholds(0.2, -0.2)
        .with_min_confidence(0.3);

    let signals = strategy.generate_signals(&mut graph, 1704067200)
        .expect("Signal generation failed");

    println!("\n   Trading Signals:");
    println!("   {:-<50}", "");
    println!("   {:12} {:>10} {:>12} {:>10}", "Symbol", "Signal", "Score", "Confidence");
    println!("   {:-<50}", "");

    for signal in &signals {
        println!("   {:12} {:>10?} {:>12.4} {:>10.4}",
            signal.symbol,
            signal.signal_type,
            signal.score,
            signal.confidence
        );
    }

    // Step 6: Generate portfolio weights
    println!("\n6. Generating portfolio weights...");
    let weights = strategy.generate_weights(&signals);

    if weights.is_empty() {
        println!("   No actionable signals - staying in cash\n");
    } else {
        println!("   Suggested allocation:");
        for (symbol, weight) in &weights {
            println!("   {} -> {:.2}%", symbol, weight * 100.0);
        }
    }

    // Step 7: Get top signals
    println!("\n7. Top 3 signals by absolute score:");
    let top_signals = strategy.top_signals(&signals, 3);
    for signal in top_signals {
        let direction = if signal.score > 0.0 { "LONG" } else { "SHORT" };
        println!("   {} {} (score: {:.4})", signal.symbol, direction, signal.score);
    }

    println!("\n=== Example Complete ===");
}

/// Create a sample market graph with realistic crypto features.
fn create_sample_graph() -> MarketGraph {
    let mut graph = MarketGraph::new();

    // Node features: [return, volatility, skewness, kurtosis, max_ret, min_ret, momentum, sharpe]
    let symbols_features = vec![
        ("BTCUSDT", vec![0.02, 0.15, 0.3, 2.5, 0.08, -0.06, 0.03, 0.45]),
        ("ETHUSDT", vec![0.03, 0.20, 0.4, 3.0, 0.10, -0.08, 0.04, 0.42]),
        ("SOLUSDT", vec![0.04, 0.30, 0.5, 4.0, 0.15, -0.12, 0.05, 0.35]),
        ("BNBUSDT", vec![0.015, 0.12, 0.2, 2.0, 0.06, -0.04, 0.02, 0.40]),
        ("XRPUSDT", vec![0.01, 0.18, 0.35, 2.8, 0.09, -0.07, 0.01, 0.28]),
        ("ADAUSDT", vec![0.02, 0.22, 0.4, 3.2, 0.11, -0.09, 0.02, 0.32]),
        ("DOGEUSDT", vec![0.05, 0.40, 0.8, 5.0, 0.20, -0.15, 0.06, 0.25]),
        ("DOTUSDT", vec![0.025, 0.25, 0.45, 3.5, 0.12, -0.10, 0.03, 0.30]),
    ];

    // Add nodes
    for (symbol, features) in &symbols_features {
        graph.add_node(*symbol, Array1::from_vec(features.clone()));
    }

    // Add edges based on typical crypto correlations
    // BTC correlations (hub)
    graph.add_edge(0, 1, 0.85); // BTC-ETH
    graph.add_edge(0, 2, 0.70); // BTC-SOL
    graph.add_edge(0, 3, 0.75); // BTC-BNB
    graph.add_edge(0, 4, 0.65); // BTC-XRP
    graph.add_edge(0, 5, 0.60); // BTC-ADA
    graph.add_edge(0, 6, 0.55); // BTC-DOGE
    graph.add_edge(0, 7, 0.60); // BTC-DOT

    // ETH correlations
    graph.add_edge(1, 2, 0.75); // ETH-SOL
    graph.add_edge(1, 7, 0.70); // ETH-DOT

    // Layer 1 correlations
    graph.add_edge(2, 5, 0.65); // SOL-ADA
    graph.add_edge(2, 7, 0.68); // SOL-DOT
    graph.add_edge(5, 7, 0.60); // ADA-DOT

    // Meme coin correlation
    graph.add_edge(4, 6, 0.50); // XRP-DOGE

    graph
}
