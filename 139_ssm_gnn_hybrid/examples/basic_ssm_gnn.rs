//! Basic SSM-GNN hybrid model demonstration.
//!
//! Shows how to create the model, process simulated data,
//! and generate trading signals.

use ssm_gnn_hybrid::data::bybit::generate_simulated_klines;
use ssm_gnn_hybrid::data::features::{build_correlation_graph, compute_technical_features};
use ssm_gnn_hybrid::model::SsmGnnHybrid;

fn main() {
    println!("=== SSM-GNN Hybrid - Basic Example ===\n");

    // Generate simulated data for 3 assets
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let klines: Vec<Vec<_>> = symbols
        .iter()
        .map(|s| generate_simulated_klines(s, 100))
        .collect();

    println!("Generated simulated kline data for {} assets", symbols.len());

    // Compute technical features
    let features: Vec<Vec<Vec<f64>>> = klines
        .iter()
        .map(|k| compute_technical_features(k, 14))
        .collect();

    println!("Computed technical features: {} features per time step", features[0][0].len());

    // Build correlation graph
    let close_prices: Vec<Vec<f64>> = klines
        .iter()
        .map(|k| k.iter().map(|c| c.close).collect())
        .collect();

    let (src, dst, weights) = build_correlation_graph(&close_prices, 0.3);
    let edge_index = vec![src.clone(), dst.clone()];
    println!("Built correlation graph: {} edges", src.len());

    // Create model (5 features, d_model=32, d_state=8, 3 classes)
    let model = SsmGnnHybrid::new(5, 32, 8, 3);

    // Window features for model input (last 50 time steps)
    let window = 50;
    let windowed: Vec<Vec<Vec<f64>>> = features
        .iter()
        .map(|f| f[f.len() - window..].to_vec())
        .collect();

    // Generate predictions
    let predictions = model.predict_signals(&windowed, &edge_index);

    println!("\n--- Trading Signals ---");
    for (sym, pred) in symbols.iter().zip(predictions.iter()) {
        println!(
            "{}: Signal={:?}, Confidence={:.4}",
            sym, pred.signal, pred.confidence
        );
    }

    // Show edge weights
    println!("\n--- Graph Edges (top 5) ---");
    let mut edge_info: Vec<(usize, usize, f64)> = src
        .iter()
        .zip(dst.iter())
        .zip(weights.iter())
        .map(|((&s, &d), &w)| (s, d, w))
        .collect();
    edge_info.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    for (s, d, w) in edge_info.iter().take(5) {
        println!("  {} -> {}: correlation={:.4}", symbols[*s], symbols[*d], w);
    }

    println!("\nDone!");
}
