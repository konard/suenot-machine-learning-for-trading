//! Bybit live signals example.
//!
//! This example demonstrates:
//! - Fetching real data from Bybit
//! - Building a market graph from correlations
//! - Generating trading signals using MPNN

use mpnn_trading::{
    data::{BybitClient, default_symbols},
    graph::{GraphBuilder, TechnicalFeatures, get_crypto_sectors},
    mpnn::{AggregationType, MPNN, MPNNConfig},
    strategy::{MPNNStrategy, RegimeDetector, MarketRegime},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Bybit MPNN Trading Signals ===\n");

    // Step 1: Initialize Bybit client
    println!("1. Connecting to Bybit API...");
    let client = BybitClient::new();

    // Step 2: Define trading universe
    let symbols = default_symbols();
    println!("   Trading universe: {} symbols", symbols.len());
    for symbol in &symbols[..5] {
        println!("     - {}", symbol);
    }
    println!("     ... and {} more\n", symbols.len() - 5);

    // Step 3: Fetch historical candles
    println!("2. Fetching historical data (1h candles, last 100 bars)...");
    let candles = client.fetch_candles(&symbols, "60", 100).await?;
    println!("   Fetched data for {} symbols\n", candles.len());

    // Step 4: Build market graph
    println!("3. Building market graph from correlations...");
    let graph_builder = GraphBuilder::new()
        .correlation_threshold(0.5)
        .min_data_points(30)
        .with_sectors(get_crypto_sectors());

    let mut graph = graph_builder.build_from_candles(&candles)?;
    println!("   Graph: {} nodes, {} edges", graph.node_count(), graph.edge_count());

    // Show top correlations
    let mut edge_weights: Vec<(&str, &str, f64)> = graph.edges
        .iter()
        .map(|e| {
            let src = &graph.nodes[e.source].symbol;
            let tgt = &graph.nodes[e.target].symbol;
            (src.as_str(), tgt.as_str(), e.weight)
        })
        .collect();
    edge_weights.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n   Top 5 correlations:");
    for (src, tgt, weight) in edge_weights.iter().take(5) {
        println!("     {} <-> {}: {:.3}", src, tgt, weight);
    }
    println!();

    // Step 5: Update node features with technical indicators
    println!("4. Computing technical features...");
    for node in &mut graph.nodes {
        if let Some(symbol_candles) = candles.get(&node.symbol) {
            node.features = TechnicalFeatures::compute(symbol_candles, 20);
        }
    }

    // Need to pad features to match input_dim
    let feature_dim = graph.nodes.first().map(|n| n.feature_dim()).unwrap_or(8);
    println!("   Feature dimension: {}\n", feature_dim);

    // Step 6: Detect market regime
    println!("5. Detecting market regime...");
    let regime_detector = RegimeDetector::new();
    let regime = regime_detector.detect(&graph);
    let regime_str = match regime {
        MarketRegime::HighCorrelation => "High Correlation (Trending)",
        MarketRegime::LowCorrelation => "Low Correlation (Mean-Reverting)",
        MarketRegime::Normal => "Normal",
        MarketRegime::Crisis => "Crisis (Risk-Off)",
    };
    println!("   Current regime: {}\n", regime_str);

    // Step 7: Create MPNN model
    println!("6. Creating MPNN model...");
    let config = MPNNConfig {
        input_dim: feature_dim,
        hidden_dim: 64,
        output_dim: 32,
        num_layers: 3,
        aggregation: match regime {
            MarketRegime::HighCorrelation => AggregationType::Mean,
            MarketRegime::LowCorrelation => AggregationType::Max,
            _ => AggregationType::Attention,
        },
        dropout: 0.1,
        use_edge_features: false,
        num_heads: 4,
    };
    let mpnn = MPNN::from_config(config);
    println!("   Model configured with {:?} aggregation\n",
        match regime {
            MarketRegime::HighCorrelation => "Mean",
            MarketRegime::LowCorrelation => "Max",
            _ => "Attention",
        });

    // Step 8: Generate signals
    println!("7. Generating trading signals...");
    let strategy = MPNNStrategy::new(mpnn)
        .with_thresholds(0.15, -0.15)
        .with_min_confidence(0.4)
        .with_max_position_size(0.15);

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let signals = strategy.generate_signals(&mut graph, timestamp)?;

    // Step 9: Display results
    println!("\n   All Signals:");
    println!("   {:-<60}", "");
    println!("   {:12} {:>12} {:>10} {:>10} {:>12}",
        "Symbol", "Signal", "Score", "Conf", "Action");
    println!("   {:-<60}", "");

    for signal in &signals {
        let action = if signal.is_buy() {
            "CONSIDER BUY"
        } else if signal.is_sell() {
            "CONSIDER SELL"
        } else {
            "HOLD"
        };

        println!("   {:12} {:>12?} {:>10.4} {:>10.2} {:>12}",
            signal.symbol,
            signal.signal_type,
            signal.score,
            signal.confidence,
            action
        );
    }

    // Step 10: Portfolio recommendations
    println!("\n8. Portfolio Recommendations:");
    println!("   {:-<40}", "");

    let buy_signals = strategy.buy_signals(&signals);
    let sell_signals = strategy.sell_signals(&signals);

    if !buy_signals.is_empty() {
        println!("\n   BUY Candidates:");
        for signal in buy_signals.iter().take(3) {
            println!("     {} (score: {:.4}, confidence: {:.2})",
                signal.symbol, signal.score, signal.confidence);
        }
    }

    if !sell_signals.is_empty() {
        println!("\n   SELL Candidates:");
        for signal in sell_signals.iter().take(3) {
            println!("     {} (score: {:.4}, confidence: {:.2})",
                signal.symbol, signal.score, signal.confidence);
        }
    }

    // Portfolio weights
    let weights = strategy.generate_weights(&signals);
    if !weights.is_empty() {
        println!("\n   Suggested Allocation:");
        let mut sorted_weights: Vec<_> = weights.iter().collect();
        sorted_weights.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (symbol, weight) in sorted_weights.iter().take(5) {
            println!("     {}: {:.1}%", symbol, weight * 100.0);
        }
    } else {
        println!("\n   No strong signals - recommend staying in cash/stables");
    }

    println!("\n=== Signal Generation Complete ===\n");
    println!("DISCLAIMER: This is for educational purposes only.");
    println!("Do not use these signals for actual trading without proper risk management.");

    Ok(())
}
