//! Basic subgraph mining example
//!
//! This example demonstrates how to:
//! 1. Create a correlation network from synthetic data
//! 2. Mine subgraph patterns
//! 3. Generate trading signals

use subgraph_mining_trading::{
    graph::{FinancialGraph, GraphBuilder, GraphType, Node},
    mining::{GSpanMiner, Pattern, PatternMatcher},
    trading::{SignalGenerator, Signal},
    data::Candle,
};
use std::collections::HashMap;

fn main() {
    println!("=== Subgraph Mining for Trading - Basic Example ===\n");

    // Step 1: Create synthetic market data
    println!("Step 1: Creating synthetic market data...");
    let data = create_synthetic_data();
    println!("  Created data for {} symbols", data.len());

    // Step 2: Build correlation graph
    println!("\nStep 2: Building correlation graph...");
    let builder = GraphBuilder::new()
        .correlation_threshold(0.6)
        .window_size(24);

    let graph = builder.build_correlation_network(&data).unwrap();
    println!("  Graph has {} nodes and {} edges", graph.node_count(), graph.edge_count());
    println!("  Density: {:.2}%", graph.density() * 100.0);

    // Step 3: Mine patterns
    println!("\nStep 3: Mining subgraph patterns...");
    let miner = GSpanMiner::new()
        .min_support(1)
        .max_size(5);

    let patterns = miner.mine(&graph).unwrap();
    println!("  Found {} patterns:", patterns.len());

    for pattern in &patterns {
        println!("    - {:?}: {} nodes, {} edges, support={}",
            pattern.pattern_type,
            pattern.node_count(),
            pattern.edge_count(),
            pattern.support
        );
    }

    // Step 4: Find triangles
    println!("\nStep 4: Analyzing graph structure...");
    let triangles = subgraph_mining_trading::graph::find_triangles(&graph);
    println!("  Found {} triangles:", triangles.len());
    for (a, b, c) in &triangles {
        let w1 = graph.edge_weight(a, b).unwrap_or(0.0);
        let w2 = graph.edge_weight(b, c).unwrap_or(0.0);
        let w3 = graph.edge_weight(a, c).unwrap_or(0.0);
        let avg = (w1 + w2 + w3) / 3.0;
        println!("    {} - {} - {} (avg corr: {:.2})", a, b, c, avg);
    }

    // Step 5: Find stars
    let stars = subgraph_mining_trading::graph::find_stars(&graph, 3);
    println!("\n  Found {} star patterns (hub with 3+ connections):", stars.len());
    for (hub, spokes) in &stars {
        println!("    {} -> [{}]", hub, spokes.join(", "));
    }

    // Step 6: Generate signals
    println!("\nStep 5: Generating trading signals...");
    let mut signal_gen = SignalGenerator::new(patterns);
    let signals = signal_gen.generate(&graph);

    println!("  Generated {} signals:", signals.len());
    for signal in &signals {
        println!("    {:?} on {} - {} (strength: {:.2}, confidence: {:.2})",
            signal.signal_type,
            signal.symbols.join(", "),
            signal.reason,
            signal.strength,
            signal.confidence
        );
    }

    // Step 7: Compute centrality
    println!("\nStep 6: Computing node centrality...");
    let centrality = subgraph_mining_trading::graph::compute_centrality(&graph);
    let mut sorted: Vec<_> = centrality.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("  Top 5 most central nodes:");
    for (symbol, score) in sorted.iter().take(5) {
        println!("    {}: {:.2}", symbol, score);
    }

    println!("\n=== Example Complete ===");
}

/// Create synthetic OHLCV data for testing
fn create_synthetic_data() -> HashMap<String, Vec<Candle>> {
    let symbols = vec![
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT",
    ];

    let base_time = 1704067200; // 2024-01-01 00:00:00 UTC
    let num_candles = 100;

    // Create correlated returns
    let btc_returns = generate_random_returns(num_candles, 0.001, 0.02);

    let mut data = HashMap::new();

    for (i, symbol) in symbols.iter().enumerate() {
        let correlation = match i {
            0 => 1.0,       // BTC with itself
            1 => 0.85,      // ETH highly correlated with BTC
            2 => 0.80,      // SOL highly correlated
            3 => 0.75,      // BNB correlated
            4 => 0.60,      // XRP less correlated
            5 => 0.55,      // ADA
            6 => 0.70,      // AVAX
            7 => 0.65,      // DOT
            8 => 0.60,      // MATIC
            _ => 0.50,      // LINK
        };

        let returns = correlate_returns(&btc_returns, correlation);
        let candles = returns_to_candles(&returns, base_time, 100.0 + i as f64 * 10.0);

        data.insert(symbol.to_string(), candles);
    }

    data
}

/// Generate random returns
fn generate_random_returns(n: usize, mean: f64, std: f64) -> Vec<f64> {
    use std::f64::consts::PI;

    (0..n)
        .map(|i| {
            // Simple pseudo-random using sine
            let x = (i as f64 * 0.1).sin() * 2.0 + (i as f64 * 0.3).cos();
            mean + std * x * 0.5
        })
        .collect()
}

/// Create correlated returns from base returns
fn correlate_returns(base: &[f64], correlation: f64) -> Vec<f64> {
    base.iter()
        .enumerate()
        .map(|(i, &r)| {
            let noise = ((i as f64 * 0.7).sin()) * 0.01;
            r * correlation + noise * (1.0 - correlation)
        })
        .collect()
}

/// Convert returns to candles
fn returns_to_candles(returns: &[f64], base_time: i64, initial_price: f64) -> Vec<Candle> {
    let mut price = initial_price;
    let mut candles = Vec::new();

    for (i, &ret) in returns.iter().enumerate() {
        let open = price;
        price *= 1.0 + ret;
        let close = price;

        let high = open.max(close) * (1.0 + ret.abs() * 0.1);
        let low = open.min(close) * (1.0 - ret.abs() * 0.1);
        let volume = 1000.0 + (i as f64 * 10.0).sin().abs() * 500.0;

        candles.push(Candle::new(
            base_time + (i as i64 * 3600), // Hourly
            open,
            high,
            low,
            close,
            volume,
        ));
    }

    candles
}
