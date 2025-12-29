//! Fetch Data Example
//!
//! Demonstrates how to fetch cryptocurrency data from Bybit.
//!
//! Run with: cargo run --example fetch_data

use online_learning::api::BybitClient;
use std::io::Write;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Bybit Data Fetcher ===\n");

    // Create client
    let client = BybitClient::new();

    // Fetch BTC/USDT hourly data
    let symbol = "BTCUSDT";
    let interval = "1h";
    let limit = 100;

    println!("Fetching {} {} candles for {}...", limit, interval, symbol);

    let candles = client.get_klines(symbol, interval, limit).await?;

    println!("Fetched {} candles\n", candles.len());

    // Display first and last candles
    if let Some(first) = candles.first() {
        println!("First candle:");
        println!("  Timestamp: {}", first.timestamp);
        println!("  Open:      {:.2}", first.open);
        println!("  High:      {:.2}", first.high);
        println!("  Low:       {:.2}", first.low);
        println!("  Close:     {:.2}", first.close);
        println!("  Volume:    {:.2}", first.volume);
        println!();
    }

    if let Some(last) = candles.last() {
        println!("Last candle:");
        println!("  Timestamp: {}", last.timestamp);
        println!("  Open:      {:.2}", last.open);
        println!("  High:      {:.2}", last.high);
        println!("  Low:       {:.2}", last.low);
        println!("  Close:     {:.2}", last.close);
        println!("  Volume:    {:.2}", last.volume);
        println!();
    }

    // Calculate some statistics
    let returns: Vec<f64> = candles
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect();

    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
    let volatility = variance.sqrt();

    println!("Statistics:");
    println!("  Mean return: {:.4}%", mean_return * 100.0);
    println!("  Volatility:  {:.4}%", volatility * 100.0);
    println!("  Max return:  {:.4}%", returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 100.0);
    println!("  Min return:  {:.4}%", returns.iter().cloned().fold(f64::INFINITY, f64::min) * 100.0);
    println!();

    // Fetch other popular pairs
    let pairs = ["ETHUSDT", "SOLUSDT", "XRPUSDT"];

    println!("Fetching current prices for other pairs:");
    for pair in pairs {
        match client.get_ticker(pair).await {
            Ok(price) => println!("  {}: ${:.2}", pair, price),
            Err(e) => println!("  {}: Error - {}", pair, e),
        }
    }

    // Save to CSV
    let csv_path = "data/btcusdt_hourly.csv";
    print!("\nSaving to {}... ", csv_path);
    std::io::stdout().flush()?;

    let mut wtr = csv::Writer::from_path(csv_path)?;
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
    println!("Done!");

    Ok(())
}
