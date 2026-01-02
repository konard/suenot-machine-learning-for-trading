//! Build and visualize market graph
//!
//! Example: cargo run --bin build_market_graph

use equivariant_gnn_trading::{BybitClient, MarketGraph};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Market Graph Builder ===\n");

    let client = BybitClient::new();
    let symbols = BybitClient::popular_symbols();

    println!("Fetching data for {} symbols...\n", symbols.len());

    let mut candles_map = HashMap::new();
    for symbol in &symbols[..10] { // Use top 10
        if let Ok(candles) = client.get_klines(symbol, "60", 168, None, None).await {
            candles_map.insert(symbol.to_string(), candles);
            print!(".");
        }
    }
    println!("\n");

    // Build graph with different thresholds
    for threshold in [0.2, 0.4, 0.6] {
        let builder = MarketGraph::new(threshold);
        let graph = builder.from_candles(&candles_map);

        println!("Correlation threshold {:.1}:", threshold);
        println!("  Nodes: {}", graph.num_nodes());
        println!("  Edges: {}", graph.num_edges());
        println!("  Avg degree: {:.1}", graph.num_edges() as f64 / graph.num_nodes() as f64);
        println!();
    }

    // Build detailed graph
    let builder = MarketGraph::new(0.3);
    let graph = builder.from_candles(&candles_map);

    println!("=== Graph Structure (threshold=0.3) ===\n");

    for node in &graph.nodes {
        let neighbors = graph.neighbors(node.idx);
        println!("{} connected to: {:?}",
            node.symbol,
            neighbors.iter()
                .filter_map(|&i| graph.nodes.get(i).map(|n| n.symbol.as_str()))
                .collect::<Vec<_>>());
    }

    println!("\n=== Node Coordinates (PCA-like embedding) ===\n");

    for node in &graph.nodes {
        println!("{}: [{:.3}, {:.3}, {:.3}]",
            node.symbol,
            node.coordinates[0],
            node.coordinates[1],
            node.coordinates[2]);
    }

    println!("\nDone!");
    Ok(())
}
