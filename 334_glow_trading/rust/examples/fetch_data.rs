//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to fetch historical OHLCV data
//! from Bybit exchange for use with the GLOW model.
//!
//! Run with: cargo run --example fetch_data

use anyhow::Result;
use chrono::{Duration, Utc};
use glow_trading::{BybitClient, Interval};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("=== GLOW Trading: Data Fetcher ===\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Define parameters
    let symbol = "BTCUSDT";
    let interval = Interval::OneHour;
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(7);

    println!("Fetching {} data from Bybit...", symbol);
    println!("Interval: 1 hour");
    println!("Period: {} to {}", start_time, end_time);
    println!();

    // Fetch data
    let candles = client
        .get_klines(symbol, interval, start_time, end_time)
        .await?;

    println!("Fetched {} candles\n", candles.len());

    // Print first few candles
    println!("First 5 candles:");
    println!("{:-<80}", "");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume"
    );
    println!("{:-<80}", "");

    for candle in candles.iter().take(5) {
        let dt = chrono::DateTime::from_timestamp_millis(candle.timestamp)
            .unwrap()
            .format("%Y-%m-%d %H:%M");
        println!(
            "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            dt, candle.open, candle.high, candle.low, candle.close, candle.volume
        );
    }
    println!("{:-<80}", "");

    // Calculate some statistics
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let returns: Vec<f64> = closes
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
        / returns.len() as f64;
    let volatility = variance.sqrt();

    let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n=== Statistics ===");
    println!("Current Price: ${:.2}", closes.last().unwrap_or(&0.0));
    println!("Min Price: ${:.2}", min_price);
    println!("Max Price: ${:.2}", max_price);
    println!("Mean Return: {:.4}%", mean_return * 100.0);
    println!("Volatility (hourly): {:.4}%", volatility * 100.0);
    println!("Volatility (annualized): {:.2}%", volatility * (365.25 * 24.0_f64).sqrt() * 100.0);

    // Also fetch current ticker
    println!("\n=== Current Market ===");
    let ticker = client.get_ticker(symbol).await?;
    println!("Symbol: {}", ticker.symbol);
    println!("Last Price: ${:.2}", ticker.last_price);
    println!("Bid: ${:.2}", ticker.bid_price);
    println!("Ask: ${:.2}", ticker.ask_price);
    println!("24h Volume: {:.2}", ticker.volume_24h);
    println!("24h Change: {:.2}%", ticker.price_change_percent_24h * 100.0);

    // Save to CSV
    let output_file = "btc_data.csv";
    let mut wtr = csv::Writer::from_path(output_file)?;
    wtr.write_record(&["timestamp", "open", "high", "low", "close", "volume", "turnover"])?;

    for candle in &candles {
        wtr.write_record(&[
            candle.timestamp.to_string(),
            candle.open.to_string(),
            candle.high.to_string(),
            candle.low.to_string(),
            candle.close.to_string(),
            candle.volume.to_string(),
            candle.turnover.to_string(),
        ])?;
    }
    wtr.flush()?;

    println!("\nData saved to {}", output_file);

    Ok(())
}
