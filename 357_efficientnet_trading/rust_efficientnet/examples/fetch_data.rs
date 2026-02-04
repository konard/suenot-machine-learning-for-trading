//! Example: Fetch market data from Bybit
//!
//! This example demonstrates how to fetch cryptocurrency data
//! from Bybit exchange using the API client.

use efficientnet_trading::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("EfficientNet Trading - Fetch Data Example");
    println!("==========================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch single timeframe data
    println!("Fetching BTCUSDT 1-hour data...");
    let btc_data = client.get_klines("BTCUSDT", "60", 100).await?;

    println!("Symbol: {}", btc_data.symbol);
    println!("Interval: {}m", btc_data.interval);
    println!("Fetched {} klines", btc_data.len());

    if let Some(latest) = btc_data.latest() {
        println!("\nLatest candle:");
        println!("  Time: {}", latest.timestamp);
        println!("  Open: ${:.2}", latest.open);
        println!("  High: ${:.2}", latest.high);
        println!("  Low: ${:.2}", latest.low);
        println!("  Close: ${:.2}", latest.close);
        println!("  Volume: {:.2}", latest.volume);
    }

    // Calculate some basic statistics
    let closes = btc_data.close_prices();
    let returns = btc_data.returns();

    if !closes.is_empty() {
        let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;
        let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\nPrice Statistics:");
        println!("  Average: ${:.2}", avg_price);
        println!("  Min: ${:.2}", min_price);
        println!("  Max: ${:.2}", max_price);
        println!("  Range: ${:.2}", max_price - min_price);
    }

    if !returns.is_empty() {
        let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let volatility = (returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
            / returns.len() as f64)
            .sqrt();

        println!("\nReturn Statistics:");
        println!("  Average Return: {:.4}%", avg_return * 100.0);
        println!("  Volatility: {:.4}%", volatility * 100.0);
    }

    // Fetch multi-timeframe data
    println!("\n------------------------------------------");
    println!("Fetching multi-timeframe data...\n");

    let timeframes = vec!["1", "5", "15", "60"];
    let multi_data = client.get_multi_timeframe("ETHUSDT", &timeframes, 50).await?;

    for data in &multi_data {
        println!(
            "  {}m: {} candles (latest close: ${:.2})",
            data.interval,
            data.len(),
            data.latest().map(|k| k.close).unwrap_or(0.0)
        );
    }

    println!("\nData fetching complete!");

    Ok(())
}
