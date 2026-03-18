//! Stock market SSM-GNN example.
//!
//! Demonstrates using the SSM-GNN hybrid model with simulated
//! stock market data alongside cryptocurrency data for comparison.

use ssm_gnn_hybrid::data::bybit::generate_simulated_klines;
use ssm_gnn_hybrid::data::features::{build_correlation_graph, compute_technical_features};
use ssm_gnn_hybrid::data::stock::generate_simulated_stock_klines;
use ssm_gnn_hybrid::model::SsmGnnHybrid;

fn run_analysis(name: &str, symbols: &[&str], klines: Vec<Vec<ssm_gnn_hybrid::data::bybit::Kline>>) {
    println!("--- {} ---", name);
    println!("Assets: {:?}", symbols);

    // Compute features
    let features: Vec<Vec<Vec<f64>>> = klines
        .iter()
        .map(|k| compute_technical_features(k, 14))
        .collect();

    // Build correlation graph
    let close_prices: Vec<Vec<f64>> = klines
        .iter()
        .map(|k| k.iter().map(|c| c.close).collect())
        .collect();

    let (src, dst, weights) = build_correlation_graph(&close_prices, 0.3);
    let edge_index = vec![src, dst];
    println!("Graph edges: {}", weights.len());

    // Create model and predict
    let model = SsmGnnHybrid::new(5, 32, 8, 3);
    let window = 50;
    let windowed: Vec<Vec<Vec<f64>>> = features
        .iter()
        .map(|f| f[f.len() - window..].to_vec())
        .collect();

    let predictions = model.predict_signals(&windowed, &edge_index);

    println!("{:<8} {:>8} {:>10}", "Asset", "Signal", "Confidence");
    println!("{}", "-".repeat(28));
    for (sym, pred) in symbols.iter().zip(predictions.iter()) {
        let signal_str = match pred.signal {
            ssm_gnn_hybrid::model::Signal::Long => "LONG",
            ssm_gnn_hybrid::model::Signal::Short => "SHORT",
            ssm_gnn_hybrid::model::Signal::Neutral => "NEUTRAL",
        };
        println!("{:<8} {:>8} {:>10.4}", sym, signal_str, pred.confidence);
    }
    println!();
}

fn main() {
    println!("=== SSM-GNN Hybrid: Stock vs Crypto Comparison ===\n");

    // Stock market analysis
    let stock_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"];
    let stock_klines: Vec<Vec<_>> = stock_tickers
        .iter()
        .map(|t| generate_simulated_stock_klines(t, 252))
        .collect();
    run_analysis("Stock Market (FAANG+)", &stock_tickers, stock_klines);

    // Cryptocurrency analysis
    let crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT"];
    let crypto_klines: Vec<Vec<_>> = crypto_symbols
        .iter()
        .map(|s| generate_simulated_klines(s, 200))
        .collect();
    run_analysis("Cryptocurrency (Bybit)", &crypto_symbols, crypto_klines);

    // Mixed portfolio
    let mixed_labels = ["AAPL", "MSFT", "BTCUSDT", "ETHUSDT"];
    let mixed_klines = vec![
        generate_simulated_stock_klines("AAPL", 200),
        generate_simulated_stock_klines("MSFT", 200),
        generate_simulated_klines("BTCUSDT", 200),
        generate_simulated_klines("ETHUSDT", 200),
    ];
    run_analysis("Mixed Portfolio (Stocks + Crypto)", &mixed_labels, mixed_klines);

    println!("Done!");
}
