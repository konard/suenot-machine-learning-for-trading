//! Basic Dynamic GNN Example
//!
//! This example demonstrates how to:
//! 1. Create a dynamic graph from cryptocurrency data
//! 2. Initialize and run a Dynamic GNN model
//! 3. Generate trading signals from predictions
//!
//! Run with: cargo run --example basic_gnn

use dynamic_gnn_trading::prelude::*;
use dynamic_gnn_trading::gnn::GNNConfig;
use dynamic_gnn_trading::graph::{DynamicGraph, GraphConfig, NodeFeatures, EdgeFeatures};
use ndarray::Array1;

fn main() {
    println!("=== Dynamic GNN Trading - Basic Example ===\n");

    // Step 1: Create a dynamic graph
    println!("Step 1: Creating dynamic graph for crypto assets...");

    let config = GraphConfig {
        max_nodes: 10,
        correlation_threshold: 0.5,
        correlation_window: 3600,
        temporal_edges: true,
    };

    let mut graph = DynamicGraph::with_config(config);

    // Add cryptocurrency nodes with features
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOGEUSDT"];
    let prices = [50000.0, 3000.0, 100.0, 35.0, 0.15];
    let volumes = [1_000_000.0, 500_000.0, 200_000.0, 100_000.0, 800_000.0];

    for (i, symbol) in symbols.iter().enumerate() {
        let features = NodeFeatures::new(prices[i], volumes[i], 1000);
        graph.add_node(*symbol, features);
        println!("  Added node: {} (price: ${:.2})", symbol, prices[i]);
    }

    // Add edges based on correlations
    println!("\nStep 2: Adding correlation edges...");

    let correlations = [
        ("BTCUSDT", "ETHUSDT", 0.85),
        ("BTCUSDT", "SOLUSDT", 0.72),
        ("ETHUSDT", "SOLUSDT", 0.78),
        ("SOLUSDT", "AVAXUSDT", 0.82),
        ("BTCUSDT", "DOGEUSDT", 0.55),
    ];

    for (src, tgt, corr) in correlations {
        let edge_features = EdgeFeatures::with_correlation(corr, 1000);
        graph.add_edge(src, tgt, edge_features);
        println!("  Edge: {} <-> {} (correlation: {:.2})", src, tgt, corr);
    }

    // Print graph statistics
    let stats = graph.stats();
    println!("\nGraph Statistics:");
    println!("  Nodes: {}", stats.node_count);
    println!("  Edges: {}", stats.edge_count);
    println!("  Density: {:.4}", stats.density);
    println!("  Avg Degree: {:.2}", stats.avg_degree);

    // Step 3: Initialize GNN model
    println!("\nStep 3: Initializing Dynamic GNN model...");

    let gnn_config = GNNConfig {
        input_dim: NodeFeatures::feature_dim(),
        hidden_dims: vec![32, 16],
        output_dim: 8,
        num_heads: 2,
        dropout: 0.1,
        use_temporal: true,
        memory_size: 50,
        learning_rate: 0.001,
    };

    let mut model = DynamicGNN::new(gnn_config);
    println!("  Model parameters: {}", model.param_count());

    // Step 4: Run forward pass
    println!("\nStep 4: Running GNN forward pass...");

    let (features, node_ids) = graph.feature_matrix();
    let (adjacency, _) = graph.adjacency_matrix();

    println!("  Feature matrix shape: {:?}", features.shape());
    println!("  Adjacency matrix shape: {:?}", adjacency.shape());

    let embeddings = model.get_embeddings(&features, &adjacency);
    println!("  Output embeddings shape: {:?}", embeddings.shape());

    // Step 5: Generate predictions
    println!("\nStep 5: Generating predictions...");

    for (i, symbol) in node_ids.iter().enumerate() {
        let embedding = embeddings.row(i).to_owned();
        let (p_down, p_neutral, p_up) = model.predict_direction(&embedding);

        let direction = if p_up > p_down && p_up > 0.4 {
            "BULLISH"
        } else if p_down > p_up && p_down > 0.4 {
            "BEARISH"
        } else {
            "NEUTRAL"
        };

        println!(
            "  {}: {} (up: {:.1}%, down: {:.1}%, neutral: {:.1}%)",
            symbol,
            direction,
            p_up * 100.0,
            p_down * 100.0,
            p_neutral * 100.0
        );
    }

    // Step 6: Predict edge relationships
    println!("\nStep 6: Predicting edge relationships...");

    for i in 0..node_ids.len() {
        for j in (i + 1)..node_ids.len() {
            let emb_i = embeddings.row(i).to_owned();
            let emb_j = embeddings.row(j).to_owned();
            let edge_prob = model.predict_edge(&emb_i, &emb_j);

            if edge_prob > 0.5 {
                println!(
                    "  {} <-> {}: {:.1}% likely connected",
                    node_ids[i],
                    node_ids[j],
                    edge_prob * 100.0
                );
            }
        }
    }

    // Step 7: Simulate graph evolution
    println!("\nStep 7: Simulating graph evolution...");

    // Update node features (simulate price changes)
    let price_changes = [0.02, -0.01, 0.05, -0.02, 0.03];

    for (i, symbol) in symbols.iter().enumerate() {
        let new_price = prices[i] * (1.0 + price_changes[i]);
        let new_features = NodeFeatures::new(new_price, volumes[i] * 1.1, 2000);
        graph.update_node(symbol, new_features);
    }

    graph.tick(2000);

    // Take snapshot
    let _snapshot = graph.snapshot();
    println!("  Graph snapshot taken at t=2000");

    // Get updated embeddings
    let (new_features, _) = graph.feature_matrix();
    let (new_adjacency, _) = graph.adjacency_matrix();
    let new_embeddings = model.get_embeddings(&new_features, &new_adjacency);

    println!("  New embeddings computed after graph evolution");
    println!("  Temporal memory updated");

    println!("\n=== Example Complete ===");
    println!("\nNext steps:");
    println!("  - Run 'cargo run --example live_trading' for real-time data");
    println!("  - Run 'cargo run --example backtest' for backtesting");
}
