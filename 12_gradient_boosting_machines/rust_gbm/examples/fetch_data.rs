//! Example: Fetching cryptocurrency data from Bybit
//!
//! Run with: cargo run --example fetch_data

use anyhow::Result;
use rust_gbm::data::{BybitClient, Interval};

#[tokio::main]
async fn main() -> Result<()> {
    println!("Bybit Data Fetching Example");
    println!("{}", "=".repeat(40));

    let client = BybitClient::new();

    // Fetch multiple symbols
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for symbol in symbols {
        println!("\n📊 {}", symbol);

        // Fetch 1-hour candles
        let candles = client
            .get_klines(symbol, Interval::Hour1, Some(100), None, None)
            .await?;

        if let Some(last) = candles.last() {
            println!("   Last Close: ${:.2}", last.close);
            println!("   24h Volume: {:.2}", candles.iter().take(24).map(|c| c.volume).sum::<f64>());
        }

        // Fetch order book
        let orderbook = client.get_orderbook(symbol, Some(5)).await?;
        if let Some(spread_pct) = orderbook.spread_pct() {
            println!("   Spread: {:.4}%", spread_pct);
        }
    }

    Ok(())
}
