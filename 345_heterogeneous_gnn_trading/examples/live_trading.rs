//! Live Trading Example with Heterogeneous GNN
//!
//! This example demonstrates how to:
//! 1. Fetch real-time data from Bybit
//! 2. Build and update a heterogeneous graph
//! 3. Generate trading signals using HGNN
//!
//! Run with: cargo run --example live_trading

use heterogeneous_gnn_trading::prelude::*;
use heterogeneous_gnn_trading::data::{BybitClient, BybitConfig, GraphBuilder};
use heterogeneous_gnn_trading::gnn::HGNNConfig;
use heterogeneous_gnn_trading::graph::{HeterogeneousGraph, GraphSchema, NodeType, EdgeType};
use heterogeneous_gnn_trading::strategy::{TradingStrategy, StrategyConfig, SignalAggregator};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Heterogeneous GNN Live Trading Demo ===\n");

    // Configuration
    let symbols = vec![
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "ARBUSDT",
        "MATICUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT", "AAVEUSDT",
    ];

    println!("Configuration:");
    println!("  Symbols: {:?}", symbols);
    println!("  Exchange: Bybit");
    println!();

    // Step 1: Initialize Bybit client
    println!("Step 1: Initializing Bybit client...");
    let client = match BybitClient::default_client() {
        Ok(c) => {
            println!("  Client initialized successfully");
            c
        }
        Err(e) => {
            println!("  Warning: Could not create client: {}", e);
            println!("  Using simulated data instead...");
            return run_simulated_demo().await;
        }
    };

    // Step 2: Fetch market data
    println!("\nStep 2: Fetching market data from Bybit...");

    let symbol_refs: Vec<&str> = symbols.iter().map(|s| s.as_ref()).collect();
    let tickers = match client.get_tickers(&symbol_refs).await {
        Ok(t) => {
            println!("  Fetched {} tickers", t.len());
            t
        }
        Err(e) => {
            println!("  Warning: Could not fetch tickers: {}", e);
            println!("  Using simulated data instead...");
            return run_simulated_demo().await;
        }
    };

    // Print ticker summary
    println!("\n  Current prices:");
    for symbol in &symbols {
        if let Some(ticker) = tickers.get(*symbol) {
            println!(
                "    {} - ${:.2} ({:+.2}%)",
                symbol, ticker.last_price, ticker.price_change_24h
            );
        }
    }

    // Step 3: Build heterogeneous graph
    println!("\nStep 3: Building heterogeneous graph...");

    let graph_builder = GraphBuilder::new(symbols.iter().map(|s| s.to_string()).collect())
        .with_correlation_threshold(0.5);

    // Note: In production, you would fetch historical data for correlation
    let price_history: HashMap<String, Vec<f64>> = HashMap::new();
    let graph = graph_builder.build_from_tickers(&tickers, &price_history);

    let stats = graph.stats();
    println!("  Graph built: {} nodes, {} edges", stats.node_count, stats.edge_count);

    // Step 4: Initialize HGNN model
    println!("\nStep 4: Initializing HGNN model...");
    let config = HGNNConfig::default();
    let model = HeterogeneousGNN::new(config);
    println!("  Model ready with {} parameters", model.param_count());

    // Step 5: Generate embeddings and signals
    println!("\nStep 5: Generating trading signals...");

    let embeddings = model.get_embeddings(&graph);
    let mut signals = Vec::new();

    for symbol in &symbols {
        if let Some(embedding) = embeddings.get(*symbol) {
            let (p_down, p_neutral, p_up) = model.predict_direction(embedding);
            let magnitude = model.predict_magnitude(embedding);

            let signal_type = SignalType::from_probs(p_down, p_neutral, p_up);
            let strength = (p_up - p_down).abs();
            let confidence = 1.0 - p_neutral;

            let signal = Signal::new(*symbol, signal_type, strength)
                .with_confidence(confidence)
                .with_expected_return(magnitude)
                .with_source("hgnn");

            signals.push(signal);
        }
    }

    // Print signals
    println!("\n  Trading Signals:");
    println!("  {:<12} {:^12} {:>8} {:>10} {:>12}", "Symbol", "Signal", "Strength", "Confidence", "Exp. Return");
    println!("  {}", "-".repeat(60));

    for signal in &signals {
        let signal_str = match signal.signal_type {
            SignalType::StrongBuy => "STRONG BUY",
            SignalType::Buy => "BUY",
            SignalType::Neutral => "NEUTRAL",
            SignalType::Sell => "SELL",
            SignalType::StrongSell => "STRONG SELL",
        };

        println!(
            "  {:<12} {:^12} {:>7.1}% {:>9.1}% {:>+11.2}%",
            signal.symbol,
            signal_str,
            signal.strength * 100.0,
            signal.confidence * 100.0,
            signal.expected_return * 100.0
        );
    }

    // Step 6: Initialize trading strategy
    println!("\nStep 6: Generating orders...");

    let strategy_config = StrategyConfig {
        max_position_size: 0.05,
        max_total_exposure: 0.5,
        max_leverage: 2.0,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.04,
        min_signal_strength: 0.3,
        min_confidence: 0.5,
        symbols: symbols.iter().map(|s| s.to_string()).collect(),
    };

    let mut strategy = TradingStrategy::new(strategy_config, 100_000.0);

    // Update prices
    for (symbol, ticker) in &tickers {
        strategy.update_price(symbol, ticker.last_price);
    }

    // Generate orders
    let orders = strategy.generate_orders(&signals);

    if orders.is_empty() {
        println!("  No orders generated (signals below thresholds)");
    } else {
        println!("\n  Generated Orders:");
        println!("  {:<12} {:>6} {:>10} {:>12} {:>12}", "Symbol", "Side", "Size", "Stop Loss", "Take Profit");
        println!("  {}", "-".repeat(56));

        for order in &orders {
            let side = match order.side {
                heterogeneous_gnn_trading::strategy::OrderSide::Buy => "BUY",
                heterogeneous_gnn_trading::strategy::OrderSide::Sell => "SELL",
            };

            println!(
                "  {:<12} {:>6} {:>10.4} {:>12} {:>12}",
                order.symbol,
                side,
                order.size,
                order.stop_loss.map(|p| format!("${:.2}", p)).unwrap_or("-".to_string()),
                order.take_profit.map(|p| format!("${:.2}", p)).unwrap_or("-".to_string()),
            );
        }
    }

    println!("\n=== Live Trading Demo Complete ===");
    println!("\nNote: This is a demonstration only. No real orders were placed.");
    println!("In production, you would:");
    println!("  1. Continuously fetch new data");
    println!("  2. Update the graph in real-time");
    println!("  3. Re-compute embeddings and signals");
    println!("  4. Execute orders via Bybit API");

    Ok(())
}

/// Run simulated demo when API is unavailable
async fn run_simulated_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Running Simulated Demo ---\n");

    // Simulated data
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "ARBUSDT"];
    let prices = [50000.0, 3000.0, 100.0, 35.0, 1.2];
    let volumes = [1_000_000.0, 500_000.0, 200_000.0, 100_000.0, 80_000.0];
    let changes = [2.5, 1.8, -0.5, 3.2, -1.1];

    println!("Simulated market data:");
    for i in 0..symbols.len() {
        println!(
            "  {} - ${:.2} ({:+.2}%)",
            symbols[i], prices[i], changes[i]
        );
    }

    // Create graph
    let schema = GraphSchema::trading_schema();
    let mut graph = HeterogeneousGraph::new(schema);

    for i in 0..symbols.len() {
        let features = AssetFeatures::new(prices[i], volumes[i], 0.02);
        graph.add_node(symbols[i], NodeType::Asset, features.into());
    }

    // Add correlations
    let correlations = [
        (0, 1, 0.85), (0, 2, 0.72), (1, 2, 0.78),
        (2, 3, 0.82), (1, 4, 0.68),
    ];

    for (i, j, corr) in &correlations {
        let features = EdgeFeatures::with_correlation(*corr, 1000);
        graph.add_edge(symbols[*i], symbols[*j], EdgeType::Correlation, features);
    }

    let stats = graph.stats();
    println!("\nGraph: {} nodes, {} edges", stats.node_count, stats.edge_count);

    // Generate signals
    let config = HGNNConfig::default();
    let model = HeterogeneousGNN::new(config);
    let embeddings = model.get_embeddings(&graph);

    println!("\nSimulated Signals:");
    for symbol in &symbols {
        if let Some(embedding) = embeddings.get(*symbol) {
            let (p_down, p_neutral, p_up) = model.predict_direction(embedding);
            let signal = if p_up > p_down { "BUY" } else if p_down > p_up { "SELL" } else { "HOLD" };
            println!("  {} - {} (confidence: {:.1}%)", symbol, signal, (1.0 - p_neutral) * 100.0);
        }
    }

    println!("\n--- Simulated Demo Complete ---");
    Ok(())
}
