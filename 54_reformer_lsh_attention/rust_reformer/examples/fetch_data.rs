//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the Bybit API client to fetch
//! historical kline data for cryptocurrency pairs.

use clap::Parser;
use reformer::{BybitClient, Kline};
use std::fs::File;
use std::io::Write;

/// Fetch cryptocurrency data from Bybit
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Trading symbol (e.g., BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval (1, 5, 15, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of klines to fetch
    #[arg(short, long, default_value = "1000")]
    limit: usize,

    /// Output file (optional)
    #[arg(short, long)]
    output: Option<String>,

    /// Fetch extended history (more than 1000 candles)
    #[arg(short, long)]
    extended: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    let args = Args::parse();

    println!("Fetching {} klines for {} (interval: {})", args.limit, args.symbol, args.interval);

    let client = BybitClient::new();

    let klines = if args.extended && args.limit > 1000 {
        println!("Using extended fetch for {} candles...", args.limit);
        client.get_extended_klines(&args.symbol, &args.interval, args.limit).await?
    } else {
        client.get_klines(&args.symbol, &args.interval, args.limit).await?
    };

    println!("Fetched {} klines", klines.len());

    // Print summary
    if let (Some(first), Some(last)) = (klines.first(), klines.last()) {
        let first_date = chrono::DateTime::from_timestamp_millis(first.timestamp as i64)
            .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_default();
        let last_date = chrono::DateTime::from_timestamp_millis(last.timestamp as i64)
            .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_default();

        println!("\nData Range:");
        println!("  First: {} - Open: {:.2}, Close: {:.2}", first_date, first.open, first.close);
        println!("  Last:  {} - Open: {:.2}, Close: {:.2}", last_date, last.open, last.close);

        // Calculate price change
        let price_change = (last.close - first.close) / first.close * 100.0;
        println!("\nPrice Change: {:.2}%", price_change);

        // Calculate average volume
        let avg_volume: f64 = klines.iter().map(|k| k.volume).sum::<f64>() / klines.len() as f64;
        println!("Average Volume: {:.2}", avg_volume);
    }

    // Save to file if requested
    if let Some(output_path) = args.output {
        println!("\nSaving to {}...", output_path);
        save_to_csv(&klines, &output_path)?;
        println!("Saved!");
    }

    Ok(())
}

/// Save klines to CSV file
fn save_to_csv(klines: &[Kline], path: &str) -> anyhow::Result<()> {
    let mut file = File::create(path)?;

    // Write header
    writeln!(file, "timestamp,open,high,low,close,volume,turnover")?;

    // Write data
    for k in klines {
        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            k.timestamp, k.open, k.high, k.low, k.close, k.volume, k.turnover
        )?;
    }

    Ok(())
}
