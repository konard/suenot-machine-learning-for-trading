//! Fetch historical data from Bybit
//!
//! Usage: cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --days 30

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use crypto_ml::api::{BybitClient, Interval, Symbol};
use crypto_ml::data::Candle;
use std::path::PathBuf;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fetch crypto data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of days to fetch
    #[arg(short, long, default_value = "30")]
    days: i64,

    /// Output file path
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn parse_interval(s: &str) -> Interval {
    match s.to_lowercase().as_str() {
        "1m" => Interval::Min1,
        "5m" => Interval::Min5,
        "15m" => Interval::Min15,
        "30m" => Interval::Min30,
        "1h" => Interval::Hour1,
        "4h" => Interval::Hour4,
        "1d" | "d" => Interval::Day1,
        "1w" | "w" => Interval::Week1,
        _ => Interval::Hour1,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("crypto_ml=info".parse()?),
        )
        .init();

    let args = Args::parse();

    info!("Fetching {} data for {} days", args.symbol, args.days);

    let client = BybitClient::new();
    let symbol = Symbol::new(&args.symbol);
    let interval = parse_interval(&args.interval);

    let end = Utc::now();
    let start = end - Duration::days(args.days);

    let candles = client
        .get_historical_klines(&symbol, interval, start, end)
        .await?;

    info!("Fetched {} candles", candles.len());

    // Print sample data
    println!("\nFirst 5 candles:");
    println!("{:<25} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume");
    println!("{}", "-".repeat(95));

    for candle in candles.iter().take(5) {
        let dt = candle.datetime().map(|d| d.to_string()).unwrap_or_default();
        println!(
            "{:<25} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            dt, candle.open, candle.high, candle.low, candle.close, candle.volume
        );
    }

    println!("\n...\n");
    println!("Last 5 candles:");

    for candle in candles.iter().rev().take(5).rev() {
        let dt = candle.datetime().map(|d| d.to_string()).unwrap_or_default();
        println!(
            "{:<25} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            dt, candle.open, candle.high, candle.low, candle.close, candle.volume
        );
    }

    // Save to file if output path specified
    if let Some(output) = args.output {
        save_candles(&candles, &output)?;
        info!("Saved data to {:?}", output);
    }

    // Print statistics
    println!("\nStatistics:");
    println!("-----------");

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;
    let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;

    println!("Average Price: ${:.2}", avg_price);
    println!("Min Price:     ${:.2}", min_price);
    println!("Max Price:     ${:.2}", max_price);
    println!("Price Range:   ${:.2} ({:.2}%)",
        max_price - min_price,
        (max_price - min_price) / min_price * 100.0);
    println!("Avg Volume:    {:.2}", avg_volume);

    Ok(())
}

fn save_candles(candles: &[Candle], path: &PathBuf) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record(["timestamp", "open", "high", "low", "close", "volume"])?;

    for candle in candles {
        writer.write_record(&[
            candle.timestamp.to_string(),
            candle.open.to_string(),
            candle.high.to_string(),
            candle.low.to_string(),
            candle.close.to_string(),
            candle.volume.to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}
