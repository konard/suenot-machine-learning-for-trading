//! # Feature Engineering Example
//!
//! Demonstrates feature extraction from order book and trade data.
//!
//! Run with: `cargo run --example feature_engineering`

use anyhow::Result;
use order_flow_imbalance::BybitClient;
use order_flow_imbalance::features::engine::FeatureEngine;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              Feature Engineering Demo                      ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    let client = BybitClient::new();
    let mut feature_engine = FeatureEngine::new();

    println!("Extracting features from BTCUSDT market data...");
    println!();

    // Collect some data first
    println!("Warming up feature engine...");
    for i in 1..=20 {
        let orderbook = client.get_orderbook("BTCUSDT", 50).await?;
        let trades = client.get_trades("BTCUSDT", 50).await?;

        feature_engine.update_orderbook(&orderbook);
        for trade in &trades {
            feature_engine.update_trade(trade);
        }

        print!("\r  Progress: {}/20 ", i);
        sleep(Duration::from_millis(200)).await;
    }
    println!();
    println!();

    // Now extract features
    let orderbook = client.get_orderbook("BTCUSDT", 50).await?;
    let features = feature_engine.extract_features(&orderbook);

    println!("═══════════════════════════════════════════════════════════");
    println!("              EXTRACTED FEATURES                           ");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Group features by category
    println!("ORDER BOOK FEATURES");
    println!("───────────────────────────────────────────────────────────");
    for name in &["mid_price", "spread_bps", "spread_zscore",
                  "depth_imbalance_l1", "depth_imbalance_l5",
                  "weighted_imbalance", "bid_depth_l5", "ask_depth_l5",
                  "depth_ratio", "bid_slope", "ask_slope", "slope_asymmetry"] {
        if let Some(value) = features.get(name) {
            println!("  {:25} {:>15.6}", name, value);
        }
    }
    println!();

    println!("ORDER FLOW FEATURES");
    println!("───────────────────────────────────────────────────────────");
    for name in &["ofi_1min", "ofi_5min", "ofi_15min", "ofi_cumulative", "ofi_zscore"] {
        if let Some(value) = features.get(name) {
            println!("  {:25} {:>15.6}", name, value);
        }
    }
    println!();

    println!("VPIN FEATURES");
    println!("───────────────────────────────────────────────────────────");
    for name in &["vpin", "vpin_zscore"] {
        if let Some(value) = features.get(name) {
            println!("  {:25} {:>15.6}", name, value);
        }
    }
    println!();

    println!("TRADE FLOW FEATURES");
    println!("───────────────────────────────────────────────────────────");
    for name in &["trade_volume_1min", "trade_imbalance", "buy_volume_1min",
                  "sell_volume_1min", "trade_count_1min", "avg_trade_size",
                  "has_large_trade"] {
        if let Some(value) = features.get(name) {
            println!("  {:25} {:>15.6}", name, value);
        }
    }
    println!();

    println!("MOMENTUM FEATURES");
    println!("───────────────────────────────────────────────────────────");
    for name in &["momentum_1min", "momentum_5min", "volatility_1min"] {
        if let Some(value) = features.get(name) {
            println!("  {:25} {:>15.6}", name, value);
        }
    }
    println!();

    println!("TIME FEATURES (Cyclical Encoding)");
    println!("───────────────────────────────────────────────────────────");
    for name in &["hour_sin", "hour_cos", "minute_sin", "minute_cos"] {
        if let Some(value) = features.get(name) {
            println!("  {:25} {:>15.6}", name, value);
        }
    }
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Total Features: {}", features.len());
    println!("  Timestamp:      {}", features.timestamp);
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Show feature array
    println!("Feature vector (for ML model):");
    println!("───────────────────────────────────────────────────────────");
    let array = features.to_array();
    print!("  [");
    for (i, val) in array.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        if i % 5 == 0 && i > 0 {
            print!("\n   ");
        }
        print!("{:.4}", val);
    }
    println!("]");
    println!();

    // List all feature names
    println!("All feature names:");
    println!("───────────────────────────────────────────────────────────");
    let names = feature_engine.feature_names();
    for (i, name) in names.iter().enumerate() {
        print!("  {:>2}. {:20}", i + 1, name);
        if (i + 1) % 3 == 0 {
            println!();
        }
    }
    println!();

    Ok(())
}
