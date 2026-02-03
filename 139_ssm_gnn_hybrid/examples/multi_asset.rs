//! Multi-asset SSM-GNN example with Bybit crypto data.
//!
//! Demonstrates fetching data for multiple crypto assets from Bybit
//! and running the SSM-GNN model on them.

use ssm_gnn_hybrid::data::bybit::BybitClient;
use ssm_gnn_hybrid::data::features::{build_correlation_graph, compute_technical_features};
use ssm_gnn_hybrid::model::SsmGnnHybrid;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    println!("=== SSM-GNN Hybrid - Multi-Asset Crypto Example ===\n");

    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT"];
    let client = BybitClient::new();

    // Fetch kline data for each symbol
    let mut all_klines = Vec::new();
    for sym in &symbols {
        println!("Fetching data for {}...", sym);
        match client.get_klines(sym, "60", 200).await {
            Ok(klines) => {
                println!("  Got {} candles", klines.len());
                all_klines.push(klines);
            }
            Err(e) => {
                println!("  Error: {}. Using simulated data.", e);
                all_klines.push(ssm_gnn_hybrid::data::bybit::generate_simulated_klines(sym, 200));
            }
        }
    }

    // Compute features
    let features: Vec<Vec<Vec<f64>>> = all_klines
        .iter()
        .map(|k| compute_technical_features(k, 14))
        .collect();

    // Build graph
    let close_prices: Vec<Vec<f64>> = all_klines
        .iter()
        .map(|k| k.iter().map(|c| c.close).collect())
        .collect();
    let (src, dst, _) = build_correlation_graph(&close_prices, 0.3);
    let edge_index = vec![src, dst];

    // Create model
    let model = SsmGnnHybrid::new(5, 64, 16, 3);

    // Window features
    let min_len = features.iter().map(|f| f.len()).min().unwrap_or(0);
    let window = 50.min(min_len);
    let windowed: Vec<Vec<Vec<f64>>> = features
        .iter()
        .map(|f| f[f.len() - window..].to_vec())
        .collect();

    // Generate signals
    let predictions = model.predict_signals(&windowed, &edge_index);

    println!("\n--- Multi-Asset Trading Signals ---");
    println!("{:<12} {:<10} {:<12}", "Symbol", "Signal", "Confidence");
    println!("{}", "-".repeat(34));
    for (sym, pred) in symbols.iter().zip(predictions.iter()) {
        let signal_str = match pred.signal {
            ssm_gnn_hybrid::model::Signal::Long => "LONG",
            ssm_gnn_hybrid::model::Signal::Short => "SHORT",
            ssm_gnn_hybrid::model::Signal::Neutral => "NEUTRAL",
        };
        println!("{:<12} {:<10} {:.4}", sym, signal_str, pred.confidence);
    }

    println!("\nDone!");
}
