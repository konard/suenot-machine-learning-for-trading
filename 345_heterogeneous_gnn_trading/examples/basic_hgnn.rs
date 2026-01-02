//! Basic Heterogeneous GNN Example
//!
//! This example demonstrates how to:
//! 1. Create a heterogeneous graph with multiple node and edge types
//! 2. Initialize and run a Heterogeneous GNN model
//! 3. Generate trading signals from predictions
//!
//! Run with: cargo run --example basic_hgnn

use heterogeneous_gnn_trading::prelude::*;
use heterogeneous_gnn_trading::gnn::HGNNConfig;
use heterogeneous_gnn_trading::graph::{HeterogeneousGraph, GraphSchema, NodeType, EdgeType};
use std::collections::HashMap;

fn main() {
    println!("=== Heterogeneous GNN Trading - Basic Example ===\n");

    // Step 1: Create a heterogeneous graph with schema
    println!("Step 1: Creating heterogeneous graph with trading schema...");

    let schema = GraphSchema::trading_schema();
    let mut graph = HeterogeneousGraph::new(schema);

    // Add Asset nodes
    println!("\n  Adding Asset nodes:");
    let assets = [
        ("BTCUSDT", 50000.0, 1_000_000.0, 0.02),
        ("ETHUSDT", 3000.0, 500_000.0, 0.03),
        ("SOLUSDT", 100.0, 200_000.0, 0.05),
        ("AVAXUSDT", 35.0, 100_000.0, 0.04),
        ("ARBUSDT", 1.2, 80_000.0, 0.06),
    ];

    for (symbol, price, volume, vol) in &assets {
        let features = AssetFeatures::new(*price, *volume, *vol);
        graph.add_node(*symbol, NodeType::Asset, features.into());
        println!("    {} - Price: ${:.2}, Volume: ${:.0}", symbol, price, volume);
    }

    // Add Exchange node
    println!("\n  Adding Exchange node:");
    let exchange_features = ExchangeFeatures::new(5_000_000.0, 100, 0.95);
    graph.add_node("Bybit", NodeType::Exchange, exchange_features.into());
    println!("    Bybit - Total Volume: $5M, Pairs: 100");

    // Add Wallet node (whale)
    println!("\n  Adding Wallet node (whale):");
    let wallet_features = WalletFeatures::new(10_000_000.0, 500, true);
    graph.add_node("Whale_001", NodeType::Wallet, wallet_features.into());
    println!("    Whale_001 - Balance: $10M, Is Whale: true");

    // Add Market Condition node
    println!("\n  Adding MarketCondition node:");
    let market_features = MarketConditionFeatures::new(55.0, 65.0, 0.5);
    graph.add_node("BullMarket", NodeType::MarketCondition, market_features.into());
    println!("    BullMarket - BTC Dom: 55%, Fear/Greed: 65");

    // Step 2: Add edges of different types
    println!("\nStep 2: Adding typed edges...");

    // Correlation edges (Asset-Asset)
    println!("\n  Correlation edges:");
    let correlations = [
        ("BTCUSDT", "ETHUSDT", 0.85),
        ("BTCUSDT", "SOLUSDT", 0.72),
        ("ETHUSDT", "SOLUSDT", 0.78),
        ("SOLUSDT", "AVAXUSDT", 0.82),
        ("ETHUSDT", "ARBUSDT", 0.68),
    ];

    for (src, tgt, corr) in &correlations {
        let features = EdgeFeatures::with_correlation(*corr, 1000);
        graph.add_edge(*src, *tgt, EdgeType::Correlation, features);
        println!("    {} <--> {} (corr: {:.2})", src, tgt, corr);
    }

    // TradesOn edges (Asset-Exchange)
    println!("\n  TradesOn edges:");
    for (symbol, _, volume, _) in &assets {
        let features = EdgeFeatures::with_volume(*volume, 1000);
        graph.add_edge(*symbol, "Bybit", EdgeType::TradesOn, features);
        println!("    {} --> Bybit", symbol);
    }

    // Holds edges (Wallet-Asset)
    println!("\n  Holds edges:");
    let holdings = [
        ("Whale_001", "BTCUSDT", 100.0),
        ("Whale_001", "ETHUSDT", 500.0),
        ("Whale_001", "SOLUSDT", 5000.0),
    ];

    for (wallet, asset, amount) in &holdings {
        let features = EdgeFeatures::with_holding(*amount, 1000);
        graph.add_edge(*wallet, *asset, EdgeType::Holds, features);
        println!("    {} --> {} ({} units)", wallet, asset, amount);
    }

    // Sensitivity edges (Asset-MarketCondition)
    println!("\n  Sensitivity edges:");
    for (symbol, _, _, _) in &assets {
        let features = EdgeFeatures::with_correlation(0.7, 1000);
        graph.add_edge(*symbol, "BullMarket", EdgeType::Sensitivity, features);
        println!("    {} --> BullMarket", symbol);
    }

    // Print graph statistics
    let stats = graph.stats();
    println!("\n  Graph Statistics:");
    println!("    Total Nodes: {}", stats.node_count);
    println!("    Total Edges: {}", stats.edge_count);
    println!("    Density: {:.4}", stats.density);
    println!("    Avg Degree: {:.2}", stats.avg_degree);

    println!("\n  Nodes by type:");
    for (node_type, count) in &stats.nodes_per_type {
        if *count > 0 {
            println!("    {:?}: {}", node_type, count);
        }
    }

    println!("\n  Edges by type:");
    for (edge_type, count) in &stats.edges_per_type {
        if *count > 0 {
            println!("    {:?}: {}", edge_type, count);
        }
    }

    // Step 3: Initialize HGNN model
    println!("\nStep 3: Initializing Heterogeneous GNN model...");

    let config = HGNNConfig::default();
    let model = HeterogeneousGNN::new(config);
    println!("  Model parameters: {}", model.param_count());
    println!("  Using metapaths: Yes");
    println!("  Attention heads: 4");

    // Step 4: Get embeddings
    println!("\nStep 4: Computing node embeddings...");

    let embeddings = model.get_embeddings(&graph);
    println!("  Generated embeddings for {} nodes", embeddings.len());

    // Step 5: Generate predictions
    println!("\nStep 5: Generating predictions for assets...");

    for (symbol, _, _, _) in &assets {
        if let Some(embedding) = embeddings.get(*symbol) {
            let (p_down, p_neutral, p_up) = model.predict_direction(embedding);
            let magnitude = model.predict_magnitude(embedding);

            let direction = if p_up > p_down && p_up > 0.4 {
                "BULLISH"
            } else if p_down > p_up && p_down > 0.4 {
                "BEARISH"
            } else {
                "NEUTRAL"
            };

            println!(
                "  {}: {} (up: {:.1}%, down: {:.1}%, neutral: {:.1}%) | Expected: {:.2}%",
                symbol,
                direction,
                p_up * 100.0,
                p_down * 100.0,
                p_neutral * 100.0,
                magnitude * 100.0
            );
        }
    }

    // Step 6: Check metapath-based relationships
    println!("\nStep 6: Analyzing metapath relationships...");

    let schema = graph.schema();
    for metapath in schema.metapaths() {
        println!("  Metapath: {} ({})", metapath.name, metapath.short_repr());
        println!("    Description: {}", metapath.description);
    }

    // Step 7: Predict edge relationships
    println!("\nStep 7: Predicting edge relationships...");

    let asset_symbols: Vec<&str> = assets.iter().map(|(s, _, _, _)| *s).collect();

    for i in 0..asset_symbols.len() {
        for j in (i + 1)..asset_symbols.len() {
            if let (Some(emb_i), Some(emb_j)) = (
                embeddings.get(asset_symbols[i]),
                embeddings.get(asset_symbols[j]),
            ) {
                let edge_prob = model.predict_edge(emb_i, emb_j);

                if edge_prob > 0.5 {
                    println!(
                        "  {} <--> {}: {:.1}% likely correlated",
                        asset_symbols[i],
                        asset_symbols[j],
                        edge_prob * 100.0
                    );
                }
            }
        }
    }

    // Step 8: Analyze whale influence
    println!("\nStep 8: Analyzing whale influence...");

    let whale_neighbors = graph.get_neighbors_by_edge_type("Whale_001", EdgeType::Holds);
    println!("  Whale_001 holds {} assets:", whale_neighbors.len());
    for (node, edge) in &whale_neighbors {
        println!(
            "    {} (weight: {:.2})",
            node.id,
            edge.features.weight
        );
    }

    println!("\n=== Example Complete ===");
    println!("\nKey concepts demonstrated:");
    println!("  1. Multiple node types: Asset, Exchange, Wallet, MarketCondition");
    println!("  2. Multiple edge types: Correlation, TradesOn, Holds, Sensitivity");
    println!("  3. Type-aware message passing and aggregation");
    println!("  4. Metapath-based semantic relationships");
    println!("\nNext steps:");
    println!("  - Run 'cargo run --example live_trading' for real-time data");
    println!("  - Run 'cargo run --example backtest' for backtesting");
}
