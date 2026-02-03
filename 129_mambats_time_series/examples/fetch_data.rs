//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the Bybit API client
//! to fetch historical candlestick data.
//!
//! Run with: cargo run --example fetch_data

use chrono::{Duration, Utc};
use mambats_time_series::{BybitClient, Interval};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    println!("MambaTS - Bybit Data Fetcher Example");
    println!("=====================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Define time range (last 7 days)
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(7);

    println!("Fetching BTCUSDT 1-hour candles...");
    println!("From: {}", start_time);
    println!("To:   {}", end_time);
    println!();

    // Fetch data
    let candles = client
        .get_klines("BTCUSDT", Interval::OneHour, start_time, end_time)
        .await?;

    println!("Fetched {} candles\n", candles.len());

    // Display first few candles
    println!("First 5 candles:");
    println!("{:-<80}", "");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume"
    );
    println!("{:-<80}", "");

    for candle in candles.iter().take(5) {
        let datetime = candle.datetime().map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        println!(
            "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            datetime, candle.open, candle.high, candle.low, candle.close, candle.volume
        );
    }

    println!("{:-<80}", "");

    // Calculate some statistics
    if !candles.is_empty() {
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_price: f64 = closes.iter().sum::<f64>() / closes.len() as f64;

        let total_volume: f64 = candles.iter().map(|c| c.volume).sum();

        println!("\nStatistics:");
        println!("  Min Price:    ${:.2}", min_price);
        println!("  Max Price:    ${:.2}", max_price);
        println!("  Avg Price:    ${:.2}", avg_price);
        println!("  Total Volume: {:.2} BTC", total_volume);

        // Calculate returns
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0] - 1.0) * 100.0).collect();
        let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let volatility: f64 = {
            let variance: f64 = returns.iter().map(|&r| (r - avg_return).powi(2)).sum::<f64>()
                / returns.len() as f64;
            variance.sqrt()
        };

        println!("  Avg Hourly Return: {:.4}%", avg_return);
        println!("  Hourly Volatility: {:.4}%", volatility);
    }

    // Fetch current ticker
    println!("\n\nCurrent Market Ticker:");
    println!("{:-<50}", "");

    let ticker = client.get_ticker("BTCUSDT").await?;
    println!("Symbol:     {}", ticker.symbol);
    println!("Last Price: ${:.2}", ticker.last_price);
    println!("Bid:        ${:.2}", ticker.bid_price);
    println!("Ask:        ${:.2}", ticker.ask_price);
    println!("24h Volume: {:.2} BTC", ticker.volume_24h);
    println!("24h Change: {:.2}%", ticker.price_change_percent_24h * 100.0);

    Ok(())
}
