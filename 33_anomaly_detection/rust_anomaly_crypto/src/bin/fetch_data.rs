//! Fetch cryptocurrency data from Bybit
//!
//! Usage: cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 1000

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use rust_anomaly_crypto::data::{intervals, symbols, BybitClient, BybitConfig};

#[derive(Parser, Debug)]
#[command(author, version, about = "Fetch cryptocurrency data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval (1, 5, 15, 60, 240, D, W)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch (max 1000 per request)
    #[arg(short, long, default_value_t = 500)]
    limit: usize,

    /// Output file path (CSV)
    #[arg(short, long)]
    output: Option<String>,

    /// Use testnet
    #[arg(long)]
    testnet: bool,

    /// Days of historical data to fetch (for pagination)
    #[arg(long)]
    days: Option<i64>,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("Bybit Data Fetcher");
    println!("==================");
    println!("Symbol: {}", args.symbol);
    println!("Interval: {}", args.interval);
    println!("Limit: {}", args.limit);

    // Create client
    let config = if args.testnet {
        BybitConfig::new().testnet()
    } else {
        BybitConfig::new()
    };
    let client = BybitClient::new(config);

    // Fetch data
    let data = if let Some(days) = args.days {
        println!("Fetching {} days of historical data...", days);
        let end_time = Utc::now();
        let start_time = end_time - Duration::days(days);
        client.get_historical_klines(&args.symbol, &args.interval, start_time, end_time)?
    } else {
        println!("Fetching last {} candles...", args.limit);
        client.get_klines(&args.symbol, &args.interval, args.limit, None, None)?
    };

    println!("\nFetched {} candles", data.len());

    if !data.is_empty() {
        let first = data.data.first().unwrap();
        let last = data.data.last().unwrap();

        println!("Time range: {} to {}", first.timestamp, last.timestamp);
        println!(
            "Price range: ${:.2} - ${:.2}",
            data.lows().iter().cloned().fold(f64::INFINITY, f64::min),
            data.highs()
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        );

        // Calculate some statistics
        let returns = data.returns();
        if !returns.is_empty() {
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 =
                returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
            let volatility = variance.sqrt();

            println!("\nStatistics:");
            println!("  Mean return: {:.4}%", mean_return * 100.0);
            println!("  Volatility: {:.4}%", volatility * 100.0);
            println!("  Latest close: ${:.2}", last.close);
            println!("  Latest volume: {:.2}", last.volume);
        }
    }

    // Save to file if output specified
    if let Some(output_path) = args.output {
        data.to_csv(&output_path)?;
        println!("\nSaved to: {}", output_path);
    } else {
        println!("\nTip: Use --output <file.csv> to save data");
    }

    // Show available symbols
    println!("\nAvailable symbols:");
    for symbol in symbols::major_pairs() {
        print!("  {} ", symbol);
    }
    println!();

    println!("\nAvailable intervals:");
    println!(
        "  {} {} {} {} {} {} {} {} {} {} {} {} {}",
        intervals::M1,
        intervals::M3,
        intervals::M5,
        intervals::M15,
        intervals::M30,
        intervals::H1,
        intervals::H2,
        intervals::H4,
        intervals::H6,
        intervals::H12,
        intervals::D1,
        intervals::W1,
        intervals::MN1
    );

    Ok(())
}
