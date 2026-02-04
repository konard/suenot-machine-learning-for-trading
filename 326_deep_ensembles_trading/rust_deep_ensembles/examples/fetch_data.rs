//! Example: Fetching market data from Bybit
//!
//! This example demonstrates how to fetch cryptocurrency market data
//! from Bybit exchange using the API client.

use deep_ensembles_trading::api::client::BybitClient;
use deep_ensembles_trading::features::engine::FeatureEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Bybit Data Fetcher Example ===\n");

    // Create API client
    let client = BybitClient::new();

    // Symbols to fetch
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    // Step 1: Fetch tickers
    println!("1. Fetching current tickers...");
    let tickers = client.get_tickers_filtered(&symbols).await?;

    for ticker in &tickers {
        println!("   {}: ${}", ticker.symbol, ticker.last_price);
    }

    // Step 2: Fetch klines for BTC
    println!("\n2. Fetching BTC/USDT klines (1h, last 100 candles)...");
    let klines = client.get_klines("BTCUSDT", "60", 100).await?;

    println!("   Fetched {} klines", klines.len());
    if let (Some(first), Some(last)) = (klines.first(), klines.last()) {
        println!("   First candle: Open=${:.2}, Close=${:.2}", first.open, first.close);
        println!("   Last candle:  Open=${:.2}, Close=${:.2}", last.open, last.close);
    }

    // Step 3: Fetch order book
    println!("\n3. Fetching BTC/USDT order book (depth=10)...");
    let orderbook = client.get_orderbook("BTCUSDT", 10).await?;

    if let Some(mid) = orderbook.mid_price() {
        println!("   Mid price: ${:.2}", mid);
    }
    if let Some(spread) = orderbook.spread_bps() {
        println!("   Spread: {:.2} bps", spread);
    }
    println!("   Bid-Ask Imbalance: {:.2}", orderbook.imbalance());

    // Step 4: Compute features
    println!("\n4. Computing features from klines...");
    let features = FeatureEngine::compute_features(&klines);
    let targets = FeatureEngine::compute_targets(&klines, 5);

    println!("   Features shape: {}x{}", features.nrows(), features.ncols());
    println!("   Targets shape: {}", targets.len());

    // Show feature statistics
    println!("\n5. Feature statistics (last row):");
    let engine = FeatureEngine::new();
    let last_row = features.row(features.nrows() - 1);
    for (i, name) in engine.feature_names.iter().enumerate() {
        if i < last_row.len() {
            println!("   {}: {:.4}", name, last_row[i]);
        }
    }

    // Step 5: Fetch multiple symbols in parallel
    println!("\n6. Fetching klines for multiple symbols in parallel...");
    let all_klines = client.get_klines_batch(&symbols, "60", 50).await?;

    for (symbol, klines) in &all_klines {
        println!("   {}: {} klines", symbol, klines.len());
    }

    println!("\n=== Data fetching complete! ===");
    Ok(())
}
