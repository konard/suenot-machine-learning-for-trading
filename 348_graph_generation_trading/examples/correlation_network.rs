//! Correlation network example with simulated Bybit data.
//!
//! This example shows how to build a correlation-based market graph
//! from price data.

use graph_generation_trading::{
    data::{MarketData, OHLCV},
    graph::{CorrelationMatrix, CorrelationMethod, GraphBuilder, GraphType},
    trading::GraphSignals,
};
use chrono::{Utc, Duration};
use rand::Rng;

fn main() {
    println!("=== Correlation Network Example ===\n");

    // Generate simulated market data
    let symbols = vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "SOLUSDT".to_string(),
        "AVAXUSDT".to_string(),
        "DOGEUSDT".to_string(),
        "SHIBUSDT".to_string(),
        "MATICUSDT".to_string(),
        "DOTUSDT".to_string(),
    ];

    println!("Generating simulated price data for {} symbols...", symbols.len());
    let market_data = generate_correlated_data(&symbols, 500);

    println!("Generated {} candles per symbol\n", market_data.num_candles());

    // Build different types of graphs
    println!("--- Building Correlation Graphs ---\n");

    // 1. Threshold-based graph
    println!("1. Threshold Graph (threshold=0.6):");
    let threshold_graph = GraphBuilder::new()
        .with_method(CorrelationMethod::Pearson)
        .with_graph_type(GraphType::Threshold)
        .with_threshold(0.6)
        .build(&market_data)
        .unwrap();

    println!("   Nodes: {}, Edges: {}, Density: {:.4}",
        threshold_graph.node_count(),
        threshold_graph.edge_count(),
        threshold_graph.density()
    );

    // 2. KNN graph
    println!("\n2. KNN Graph (k=3):");
    let knn_graph = GraphBuilder::new()
        .with_method(CorrelationMethod::Pearson)
        .with_graph_type(GraphType::KNN)
        .with_k(3)
        .build(&market_data)
        .unwrap();

    println!("   Nodes: {}, Edges: {}, Density: {:.4}",
        knn_graph.node_count(),
        knn_graph.edge_count(),
        knn_graph.density()
    );

    // 3. Minimum Spanning Tree
    println!("\n3. Minimum Spanning Tree:");
    let mst_graph = GraphBuilder::new()
        .with_method(CorrelationMethod::Pearson)
        .with_graph_type(GraphType::MST)
        .build(&market_data)
        .unwrap();

    println!("   Nodes: {}, Edges: {}", mst_graph.node_count(), mst_graph.edge_count());
    println!("   MST Edges:");
    for (s1, s2, weight) in mst_graph.edges() {
        println!("     {} -- {} : {:.4}", s1, s2, weight);
    }

    // 4. Full correlation graph
    println!("\n4. Full Correlation Graph:");
    let full_graph = GraphBuilder::new()
        .with_method(CorrelationMethod::Pearson)
        .with_graph_type(GraphType::Full)
        .build(&market_data)
        .unwrap();

    println!("   Average correlation: {:.4}", full_graph.average_edge_weight());

    // Generate trading signals
    println!("\n--- Trading Signals from Graph ---\n");
    let signals = GraphSignals::new(&threshold_graph);

    // Community detection
    let communities = signals.detect_communities();
    println!("Detected {} communities:", communities.len());
    for (i, community) in communities.iter().enumerate() {
        println!("  Community {}: {}", i + 1, community.join(", "));
    }

    // Hub detection
    println!("\nHub Assets (by centrality):");
    let hubs = signals.detect_hubs(3);
    for (symbol, score) in hubs {
        println!("  {}: {:.4}", symbol, score);
    }

    // Network stress indicator
    let stress = signals.network_stress();
    println!("\nNetwork Stress Indicator: {:.4}", stress);

    // Market regime
    let regime = signals.regime_indicator();
    println!("Market Regime: {:?}", regime);
    println!("Recommended position multiplier: {:.2}", regime.position_multiplier());
    println!("Recommended max leverage: {:.1}x", regime.max_leverage());

    // Centrality-based trading signals
    println!("\n--- Centrality-Based Trading Signals ---");
    let trading_signals = signals.centrality_signals(0.3, 0.3);
    for (symbol, signal) in &trading_signals {
        let direction = if *signal > 0.0 {
            "LONG"
        } else if *signal < 0.0 {
            "SHORT"
        } else {
            "NEUTRAL"
        };
        println!("  {}: {} ({})", symbol, signal, direction);
    }

    println!("\n=== Example Complete ===");
}

/// Generate correlated market data for testing
fn generate_correlated_data(symbols: &[String], num_candles: usize) -> MarketData {
    let mut rng = rand::thread_rng();
    let mut market_data = MarketData::new(symbols.to_vec(), "1h");

    // Base prices for each symbol
    let base_prices: Vec<f64> = vec![40000.0, 2000.0, 100.0, 30.0, 0.08, 0.00001, 0.8, 5.0];

    // Generate common market factor
    let market_factor: Vec<f64> = (0..num_candles)
        .map(|i| {
            let trend = (i as f64 * 0.01).sin() * 0.02;
            let noise = rng.gen_range(-0.01..0.01);
            trend + noise
        })
        .collect();

    // Generate sector factors
    let l1_factor: Vec<f64> = (0..num_candles)
        .map(|_| rng.gen_range(-0.005..0.005))
        .collect();

    let meme_factor: Vec<f64> = (0..num_candles)
        .map(|_| rng.gen_range(-0.02..0.02))
        .collect();

    for (idx, symbol) in symbols.iter().enumerate() {
        let base_price = base_prices.get(idx).copied().unwrap_or(100.0);
        let mut price = base_price;

        let candles: Vec<OHLCV> = (0..num_candles)
            .map(|i| {
                let now = Utc::now() - Duration::hours((num_candles - i) as i64);

                // Combine factors based on symbol type
                let market_effect = market_factor[i];
                let sector_effect = match symbol.as_str() {
                    "BTCUSDT" | "ETHUSDT" | "SOLUSDT" | "AVAXUSDT" => l1_factor[i],
                    "DOGEUSDT" | "SHIBUSDT" => meme_factor[i],
                    _ => rng.gen_range(-0.005..0.005),
                };
                let idiosyncratic = rng.gen_range(-0.005..0.005);

                let return_val = market_effect + sector_effect + idiosyncratic;
                price *= 1.0 + return_val;

                let volatility = price * 0.02;
                let high = price + rng.gen_range(0.0..volatility);
                let low = price - rng.gen_range(0.0..volatility);
                let open = price - rng.gen_range(-volatility / 2.0..volatility / 2.0);
                let volume = rng.gen_range(1000.0..10000.0);

                OHLCV::new(now, open, high, low, price, volume)
            })
            .collect();

        market_data.add_candles(symbol, candles);
    }

    market_data
}
