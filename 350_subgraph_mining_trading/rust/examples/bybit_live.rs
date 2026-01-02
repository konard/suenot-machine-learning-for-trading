//! Live Bybit data example
//!
//! This example demonstrates how to:
//! 1. Fetch real market data from Bybit
//! 2. Build correlation network
//! 3. Mine patterns and generate signals

use subgraph_mining_trading::{
    data::{BybitClient, DataCache},
    graph::GraphBuilder,
    mining::GSpanMiner,
    trading::SignalGenerator,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    println!("=== Subgraph Mining - Live Bybit Data ===\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Select symbols to analyze
    let symbols = BybitClient::layer1_pairs();
    println!("Analyzing {} Layer-1 tokens:", symbols.len());
    for symbol in &symbols {
        println!("  - {}", symbol);
    }

    // Fetch candle data
    println!("\nFetching 1-hour candles...");
    let data = client.fetch_candles_multi(&symbols, "1h", 100).await?;

    println!("Fetched data for {} symbols", data.len());
    for (symbol, candles) in &data {
        if let Some(last) = candles.last() {
            println!("  {} - {} candles, last close: ${:.2}",
                symbol,
                candles.len(),
                last.close
            );
        }
    }

    // Build correlation graph
    println!("\nBuilding correlation network...");
    let builder = GraphBuilder::new()
        .correlation_threshold(0.7)
        .window_size(24);

    match builder.build_correlation_network(&data) {
        Ok(graph) => {
            println!("Graph constructed:");
            println!("  Nodes: {}", graph.node_count());
            println!("  Edges: {}", graph.edge_count());
            println!("  Density: {:.2}%", graph.density() * 100.0);

            // Show edges
            println!("\nStrong correlations (>0.7):");
            for (from, to, weight) in graph.edges() {
                println!("  {} <-> {} : {:.3}", from, to, weight);
            }

            // Find patterns
            println!("\nMining patterns...");
            let miner = GSpanMiner::new()
                .min_support(1)
                .max_size(5);

            let patterns = miner.mine(&graph)?;
            println!("Found {} patterns:", patterns.len());

            for pattern in &patterns {
                println!("  {:?} - support: {}, avg_weight: {:.3}",
                    pattern.pattern_type,
                    pattern.support,
                    pattern.avg_weight
                );
            }

            // Find triangles
            let triangles = subgraph_mining_trading::graph::find_triangles(&graph);
            if !triangles.is_empty() {
                println!("\nTriangles (correlated triplets):");
                for (a, b, c) in &triangles {
                    println!("  {} - {} - {}", a, b, c);
                }
            }

            // Generate signals
            println!("\nGenerating signals...");
            let mut signal_gen = SignalGenerator::new(patterns);
            let signals = signal_gen.generate(&graph);

            if signals.is_empty() {
                println!("  No signals generated");
            } else {
                for signal in &signals {
                    println!("  {:?}: {} - {}",
                        signal.signal_type,
                        signal.symbols.join(", "),
                        signal.reason
                    );
                }
            }
        }
        Err(e) => {
            println!("Failed to build graph: {}", e);
        }
    }

    // Fetch current tickers
    println!("\nFetching current prices...");
    let tickers = client.fetch_tickers(&symbols[..3]).await?;

    for ticker in &tickers {
        println!("  {} - Price: ${:.2}, 24h Change: {:.2}%",
            ticker.symbol,
            ticker.last_price,
            ticker.price_change_24h
        );
    }

    println!("\n=== Complete ===");
    Ok(())
}
