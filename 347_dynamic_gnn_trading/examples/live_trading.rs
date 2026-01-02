//! Live Trading Demo with Bybit Data
//!
//! This example demonstrates:
//! 1. Fetching real-time data from Bybit
//! 2. Building dynamic graph from market data
//! 3. Running GNN predictions
//! 4. Generating trading signals
//!
//! Run with: cargo run --example live_trading

use dynamic_gnn_trading::data::{BybitClient, BybitConfig, FeatureEngine};
use dynamic_gnn_trading::graph::{DynamicGraph, GraphConfig, NodeFeatures, EdgeFeatures};
use dynamic_gnn_trading::gnn::{DynamicGNN, GNNConfig};
use dynamic_gnn_trading::strategy::{SignalGenerator, TradingStrategy, StrategyConfig};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dynamic GNN Live Trading Demo ===\n");

    // Configuration
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"];

    // Step 1: Initialize Bybit client
    println!("Step 1: Connecting to Bybit API...");
    let bybit_config = BybitConfig::default();
    let client = BybitClient::new(bybit_config)?;
    println!("  Connected to {}", "api.bybit.com");

    // Step 2: Fetch market data
    println!("\nStep 2: Fetching market data...");

    let tickers = client.get_tickers(&symbols.iter().map(|s| *s).collect::<Vec<_>>()).await?;

    println!("  Received {} tickers", tickers.len());
    for (symbol, ticker) in &tickers {
        println!(
            "    {}: ${:.2} ({:+.2}%)",
            symbol, ticker.last_price, ticker.price_change_24h
        );
    }

    // Fetch klines for each symbol
    println!("\n  Fetching historical klines...");
    let mut feature_engines: HashMap<String, FeatureEngine> = HashMap::new();
    let mut returns_map: HashMap<String, Vec<f64>> = HashMap::new();

    for symbol in &symbols {
        let klines = client.get_klines(symbol, "15", 100).await?;
        println!("    {}: {} candles", symbol, klines.len());

        let mut engine = FeatureEngine::new();
        let mut returns = Vec::new();

        for kline in &klines {
            engine.update_from_kline(kline);
        }

        returns.extend(engine.returns_history());
        feature_engines.insert(symbol.to_string(), engine);
        returns_map.insert(symbol.to_string(), returns);
    }

    // Step 3: Build dynamic graph
    println!("\nStep 3: Building dynamic graph...");

    let graph_config = GraphConfig {
        max_nodes: symbols.len(),
        correlation_threshold: 0.5,
        correlation_window: 3600,
        temporal_edges: true,
    };

    let mut graph = DynamicGraph::with_config(graph_config);

    // Add nodes with features
    for symbol in &symbols {
        if let Some(ticker) = tickers.get(*symbol) {
            let features = NodeFeatures {
                price: ticker.last_price,
                price_change_24h: ticker.price_change_24h,
                volume_24h: ticker.volume_24h,
                spread: ticker.spread_pct(),
                ..Default::default()
            };
            graph.add_node(*symbol, features);
        }
    }

    // Update correlations
    graph.update_correlations(&returns_map, 0.5);

    let stats = graph.stats();
    println!("  Nodes: {}, Edges: {}", stats.node_count, stats.edge_count);

    // Print edges
    for edge in graph.edges() {
        println!(
            "    {} <-> {}: corr={:.3}",
            edge.source, edge.target, edge.features.correlation
        );
    }

    // Step 4: Initialize GNN model
    println!("\nStep 4: Initializing GNN model...");

    let gnn_config = GNNConfig {
        input_dim: NodeFeatures::feature_dim(),
        hidden_dims: vec![64, 32],
        output_dim: 16,
        num_heads: 4,
        use_temporal: true,
        ..Default::default()
    };

    let mut model = DynamicGNN::new(gnn_config);
    println!("  Model initialized with {} parameters", model.param_count());

    // Step 5: Run inference
    println!("\nStep 5: Running GNN inference...");

    let (features, node_ids) = graph.feature_matrix();
    let (adjacency, _) = graph.adjacency_matrix();

    let output = model.forward(&features, &adjacency, None);
    println!("  Output shape: {:?}", output.shape());

    // Step 6: Generate signals
    println!("\nStep 6: Generating trading signals...");

    let mut signal_gen = SignalGenerator::new();

    for (i, symbol) in node_ids.iter().enumerate() {
        let embedding = output.row(i).to_owned();
        let (p_down, p_neutral, p_up) = model.predict_direction(&embedding);

        if let Some(ticker) = tickers.get(symbol) {
            let signal = signal_gen.generate(
                symbol,
                ticker.last_price,
                (p_down, p_neutral, p_up),
                0.7, // confidence
                dynamic_gnn_trading::utils::now_ms(),
            );

            let emoji = match signal.signal_type {
                dynamic_gnn_trading::strategy::SignalType::Buy => "🟢",
                dynamic_gnn_trading::strategy::SignalType::Sell => "🔴",
                dynamic_gnn_trading::strategy::SignalType::Hold => "⚪",
            };

            println!(
                "  {} {}: {} (strength: {:.1}%)",
                emoji,
                symbol,
                signal.signal_type,
                signal.strength * 100.0
            );
            println!("      {}", signal.reason);
        }
    }

    // Step 7: Strategy evaluation
    println!("\nStep 7: Strategy summary...");

    let strategy_config = StrategyConfig {
        min_confidence: 0.6,
        max_position_pct: 0.05,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.03,
        ..Default::default()
    };

    let strategy = TradingStrategy::new(strategy_config);

    let signal_stats = signal_gen.stats();
    println!("  Total signals: {}", signal_stats.total_signals);
    println!("  Buy signals: {}", signal_stats.buy_signals);
    println!("  Sell signals: {}", signal_stats.sell_signals);
    println!("  Hold signals: {}", signal_stats.hold_signals);
    println!("  Avg confidence: {:.1}%", signal_stats.avg_confidence * 100.0);

    println!("\n=== Demo Complete ===");
    println!("\nNote: This is a demonstration only.");
    println!("Real trading requires proper risk management and testing.");

    Ok(())
}
