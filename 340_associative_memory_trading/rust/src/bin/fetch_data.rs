//! Fetch historical data from Bybit
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --days 30 --output data/btc_hourly.csv

use anyhow::Result;
use associative_memory_trading::data::{intervals, symbols, BybitClient};
use chrono::{Duration, Utc};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "fetch_data")]
#[command(about = "Fetch historical OHLCV data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval (1, 5, 15, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of days to fetch
    #[arg(short, long, default_value = "30")]
    days: i64,

    /// Output CSV file path
    #[arg(short, long, default_value = "data/ohlcv.csv")]
    output: String,

    /// Use testnet
    #[arg(long, default_value = "false")]
    testnet: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    log::info!("Fetching data for {} with interval {}", args.symbol, args.interval);

    let client = BybitClient::public();

    let end_time = Utc::now();
    let start_time = end_time - Duration::days(args.days);

    log::info!("Period: {} to {}", start_time, end_time);

    let data = client.get_historical_klines(&args.symbol, &args.interval, start_time, end_time)?;

    log::info!("Fetched {} candles", data.len());

    // Create output directory if needed
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    data.to_csv(&args.output)?;
    log::info!("Saved to {}", args.output);

    // Print sample
    println!("\nSample data (last 5 candles):");
    println!("{:^25} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume");
    println!("{}", "-".repeat(95));

    for candle in data.data.iter().rev().take(5).rev() {
        println!(
            "{:^25} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            candle.timestamp.format("%Y-%m-%d %H:%M"),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        );
    }

    // Print available symbols and intervals
    println!("\n--- Available Symbols ---");
    for symbol in symbols::major_pairs() {
        print!("{} ", symbol);
    }
    println!("\n\n--- Available Intervals ---");
    println!("Minutes: {} {} {} {} {}",
        intervals::M1, intervals::M3, intervals::M5, intervals::M15, intervals::M30);
    println!("Hours: {} {} {} {} {}",
        intervals::H1, intervals::H2, intervals::H4, intervals::H6, intervals::H12);
    println!("Days/Weeks/Months: {} {} {}",
        intervals::D1, intervals::W1, intervals::MN1);

    Ok(())
}
