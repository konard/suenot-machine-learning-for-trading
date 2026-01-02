//! Demo of E-GNN Trading Model
//!
//! Example: cargo run --bin demo_egnn

use equivariant_gnn_trading::{
    EquivariantGNN, MarketGraph, BybitClient,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== E-GNN Trading Model Demo ===\n");

    // Fetch real data from Bybit
    let client = BybitClient::new();
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"];

    println!("Fetching market data from Bybit...");
    let mut candles_map = HashMap::new();

    for symbol in &symbols {
        if let Ok(candles) = client.get_klines(symbol, "60", 168, None, None).await {
            println!("  {}: {} candles", symbol, candles.len());
            candles_map.insert(symbol.to_string(), candles);
        }
    }

    if candles_map.is_empty() {
        println!("No data fetched. Using synthetic data for demo.");
        // Create synthetic data
        use equivariant_gnn_trading::Candle;
        for (i, symbol) in symbols.iter().enumerate() {
            let mut candles = Vec::new();
            let base_price = 100.0 * (i + 1) as f64;
            for t in 0..100 {
                let noise = (t as f64 * 0.1).sin() * 5.0;
                candles.push(Candle::new(
                    t as u64 * 3600000,
                    base_price + noise,
                    base_price + noise + 2.0,
                    base_price + noise - 2.0,
                    base_price + noise + 1.0,
                    1000.0,
                    base_price * 1000.0,
                ));
            }
            candles_map.insert(symbol.to_string(), candles);
        }
    }

    // Build market graph
    println!("\nBuilding market graph...");
    let graph_builder = MarketGraph::new(0.3);
    let graph = graph_builder.from_candles(&candles_map);

    println!("Graph statistics:");
    println!("  Nodes: {}", graph.num_nodes());
    println!("  Edges: {}", graph.num_edges());
    println!("  Node feature dim: {}", graph.node_feature_dim());
    println!("  Coord dim: {}", graph.coord_dim());

    // Create E-GNN model
    println!("\nCreating E-GNN model...");
    let model = EquivariantGNN::new(
        graph.node_feature_dim(),
        64,  // hidden_dim
        3,   // coord_dim
        4,   // num_layers
    );

    println!("Model config:");
    println!("  Hidden dim: {}", model.config().hidden_dim);
    println!("  Num layers: {}", model.config().num_layers);
    println!("  Output classes: {}", model.config().output_classes);

    // Forward pass
    println!("\nRunning forward pass...");
    let output = model.forward(&graph);

    println!("\n=== Trading Signals ===\n");

    for (i, node) in graph.nodes.iter().enumerate() {
        let probs = &output.direction_probs;
        let short_p = probs[[i, 0]];
        let hold_p = probs[[i, 1]];
        let long_p = probs[[i, 2]];

        let signal = if long_p > 0.4 { "LONG" }
            else if short_p > 0.4 { "SHORT" }
            else { "HOLD" };

        println!("{}: {} (S:{:.1}% H:{:.1}% L:{:.1}%) | Size: {:.1}% | Vol: {:.2}%",
            node.symbol, signal,
            short_p * 100.0, hold_p * 100.0, long_p * 100.0,
            output.position_sizes[i] * 100.0,
            output.volatility[i] * 100.0);
    }

    println!("\n=== Updated Coordinates (Learned Embeddings) ===\n");

    for (i, node) in graph.nodes.iter().enumerate() {
        println!("{}: [{:.3}, {:.3}, {:.3}]",
            node.symbol,
            output.coordinates[[i, 0]],
            output.coordinates[[i, 1]],
            output.coordinates[[i, 2]]);
    }

    println!("\nDemo complete!");

    Ok(())
}
