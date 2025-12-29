//! Fetch historical data from Bybit
//!
//! Usage: cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --days 30

use anyhow::Result;
use chrono::{Duration, Utc};
use rust_nn_crypto::data::{BybitClient, BybitConfig};
use std::env;

fn main() -> Result<()> {
    env_logger::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let mut symbol = "BTCUSDT".to_string();
    let mut interval = "60".to_string();
    let mut days = 30i64;
    let mut output_path = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--symbol" | "-s" => {
                symbol = args.get(i + 1).cloned().unwrap_or(symbol);
                i += 2;
            }
            "--interval" | "-i" => {
                interval = args.get(i + 1).cloned().unwrap_or(interval);
                i += 2;
            }
            "--days" | "-d" => {
                days = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(days);
                i += 2;
            }
            "--output" | "-o" => {
                output_path = args.get(i + 1).cloned();
                i += 2;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("Fetching {} data for {} (last {} days)...", interval, symbol, days);

    // Create client
    let config = BybitConfig::default();
    let client = BybitClient::new(config);

    // Calculate time range
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(days);

    // Fetch data
    let series = client.get_historical_klines(&symbol, &interval, start_time, end_time)?;

    println!("Fetched {} candles", series.len());

    if series.is_empty() {
        println!("No data received. Please check symbol and interval.");
        return Ok(());
    }

    // Print summary
    let first = series.data.first().unwrap();
    let last = series.data.last().unwrap();

    println!("\nData Summary:");
    println!("  Symbol: {}", series.symbol);
    println!("  Interval: {}", series.interval);
    println!("  Period: {} to {}", first.timestamp, last.timestamp);
    println!("  Open: {:.2} -> Close: {:.2}", first.open, last.close);
    println!("  Return: {:.2}%", (last.close - first.open) / first.open * 100.0);

    // Price statistics
    let highs: Vec<f64> = series.data.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = series.data.iter().map(|c| c.low).collect();
    let max_high = highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_low = lows.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  Highest: {:.2}", max_high);
    println!("  Lowest: {:.2}", min_low);

    // Volume statistics
    let total_volume: f64 = series.data.iter().map(|c| c.volume).sum();
    println!("  Total Volume: {:.2}", total_volume);

    // Save to CSV if output path specified
    let output = output_path.unwrap_or_else(|| format!("{}_{}.csv", symbol, interval));
    series.save_csv(&output)?;
    println!("\nData saved to: {}", output);

    Ok(())
}

fn print_help() {
    println!("Fetch historical cryptocurrency data from Bybit");
    println!();
    println!("USAGE:");
    println!("    cargo run --bin fetch_data -- [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -s, --symbol <SYMBOL>      Trading pair (default: BTCUSDT)");
    println!("    -i, --interval <INTERVAL>  Candle interval (default: 60)");
    println!("                               Options: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M");
    println!("    -d, --days <DAYS>          Number of days to fetch (default: 30)");
    println!("    -o, --output <PATH>        Output CSV file path");
    println!("    -h, --help                 Print help information");
    println!();
    println!("EXAMPLES:");
    println!("    cargo run --bin fetch_data -- --symbol ETHUSDT --interval 240 --days 90");
    println!("    cargo run --bin fetch_data -- -s SOLUSDT -i D -d 365 -o sol_daily.csv");
}
