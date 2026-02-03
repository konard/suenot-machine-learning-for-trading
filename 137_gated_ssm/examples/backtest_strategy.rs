//! Backtesting example for Gated SSM strategy.
//!
//! Fetches crypto data from Bybit, generates signals, and runs a backtest.

use gated_ssm::backtest::BacktestEngine;
use gated_ssm::data::{compute_features, fetch_bybit_klines};
use gated_ssm::trading::GatedSSMStrategy;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Gated SSM Backtest Example ===\n");

    // Fetch data
    println!("Fetching BTCUSDT daily data from Bybit...");
    let klines = fetch_bybit_klines("BTCUSDT", "D", 500).await?;
    println!("Fetched {} klines", klines.len());

    // Compute features
    let features = compute_features(&klines);
    println!("Computed {} feature vectors", features.len());

    // Generate signals
    let mut strategy = GatedSSMStrategy::new(4, 32, 0.05, 60);
    let mut signals = Vec::new();
    for feat in &features {
        let signal = strategy.on_bar(feat);
        signals.push(signal.to_position());
    }

    // Extract close prices (aligned with features)
    let prices: Vec<f64> = klines[1..].iter().map(|k| k.close).collect();

    // Run backtest
    let engine = BacktestEngine::new(0.001);
    let metrics = engine.run(&prices, &signals);

    println!("\n{}", metrics);

    // Also test on ETH
    println!("\n--- ETH Backtest ---\n");
    println!("Fetching ETHUSDT daily data from Bybit...");
    let eth_klines = fetch_bybit_klines("ETHUSDT", "D", 500).await?;
    println!("Fetched {} klines", eth_klines.len());

    let eth_features = compute_features(&eth_klines);
    strategy.reset();
    let mut eth_signals = Vec::new();
    for feat in &eth_features {
        let signal = strategy.on_bar(feat);
        eth_signals.push(signal.to_position());
    }

    let eth_prices: Vec<f64> = eth_klines[1..].iter().map(|k| k.close).collect();
    let eth_metrics = engine.run(&eth_prices, &eth_signals);
    println!("{}", eth_metrics);

    Ok(())
}
