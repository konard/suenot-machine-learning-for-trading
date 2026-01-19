//! Example: Fetch cryptocurrency data from Bybit.
//!
//! This example demonstrates how to:
//! 1. Create a Bybit client
//! 2. Fetch historical price data
//! 3. Calculate basic metrics
//!
//! Run with: cargo run --example fetch_data

use anyhow::Result;
use llm_risk_assessment::data::{BybitClient, MultiSymbolData};
use tracing::info;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("Starting data fetch example");

    // Create Bybit client
    let client = BybitClient::new();

    // Get top symbols by volume
    info!("Fetching top symbols by volume...");
    let top_symbols = client.get_top_symbols_by_volume(5).await?;
    println!("\nTop 5 symbols by volume:");
    for (i, symbol) in top_symbols.iter().enumerate() {
        println!("  {}. {}", i + 1, symbol);
    }

    // Fetch data for BTC and ETH
    let symbols = vec!["BTCUSDT", "ETHUSDT"];
    info!("Fetching historical data for {} symbols...", symbols.len());

    let data = client.fetch_multi_symbol(&symbols, "60", 7).await?;

    // Display summary
    println!("\n=== Data Summary ===");
    for symbol in data.symbols() {
        if let Some(ohlcv) = data.get(symbol) {
            println!("\n{}: {} candles", symbol, ohlcv.len());

            if let Some(first) = ohlcv.first() {
                if let Some(last) = ohlcv.last() {
                    let price_change = (last.close - first.close) / first.close * 100.0;
                    println!("  First close: ${:.2}", first.close);
                    println!("  Last close:  ${:.2}", last.close);
                    println!("  Change:      {:.2}%", price_change);
                }
            }

            // Calculate volatility
            if let Some(volatility) = data.calculate_volatility(symbol, 24) {
                if let Some(last_vol) = volatility.last() {
                    println!("  24h volatility: {:.4}%", last_vol * 100.0);
                }
            }
        }
    }

    // Calculate and display returns
    println!("\n=== Returns Analysis ===");
    for symbol in data.symbols() {
        if let Some(returns) = data.calculate_returns(symbol) {
            let positive = returns.iter().filter(|&&r| r > 0.0).count();
            let negative = returns.iter().filter(|&&r| r < 0.0).count();

            let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let max_return = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_return = returns.iter().cloned().fold(f64::INFINITY, f64::min);

            println!("\n{} returns:", symbol);
            println!("  Positive periods: {}", positive);
            println!("  Negative periods: {}", negative);
            println!("  Average return:   {:.4}%", avg_return * 100.0);
            println!("  Max return:       {:.4}%", max_return * 100.0);
            println!("  Min return:       {:.4}%", min_return * 100.0);
        }
    }

    println!("\nData fetch complete!");

    Ok(())
}
