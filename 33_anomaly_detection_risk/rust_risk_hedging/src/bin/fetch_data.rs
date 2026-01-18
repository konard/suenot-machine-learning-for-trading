//! Fetch market data from Bybit
//!
//! Usage: cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 500

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use rust_risk_hedging::data::{intervals, symbols, BybitClient, OHLCVSeries};

#[derive(Parser, Debug)]
#[command(author, version, about = "Fetch cryptocurrency data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval (1, 5, 15, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "200")]
    limit: usize,

    /// Save to CSV file
    #[arg(short, long)]
    output: Option<String>,

    /// Show multiple symbols
    #[arg(long)]
    multi: bool,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("=== Bybit Data Fetcher ===\n");

    let client = BybitClient::public();

    if args.multi {
        // Fetch multiple symbols
        println!("Fetching data for major pairs...\n");

        for symbol in symbols::major_pairs() {
            match client.get_klines(symbol, &args.interval, 10, None, None) {
                Ok(data) => {
                    print_summary(symbol, &data);
                }
                Err(e) => {
                    eprintln!("Error fetching {}: {}", symbol, e);
                }
            }
        }
    } else {
        // Fetch single symbol
        println!("Fetching {} {} candles for {}...\n", args.limit, args.interval, args.symbol);

        let data = client.get_klines(&args.symbol, &args.interval, args.limit, None, None)?;

        print_detailed(&data);

        // Save to CSV if requested
        if let Some(output) = &args.output {
            save_to_csv(&data, output)?;
            println!("\nData saved to {}", output);
        }
    }

    Ok(())
}

fn print_summary(symbol: &str, data: &OHLCVSeries) {
    if let Some(last) = data.data.last() {
        let change = if let Some(first) = data.data.first() {
            (last.close - first.close) / first.close * 100.0
        } else {
            0.0
        };

        println!(
            "{:12} | Price: ${:>10.2} | Change: {:>+6.2}% | Volume: {:>12.0}",
            symbol, last.close, change, last.volume
        );
    }
}

fn print_detailed(data: &OHLCVSeries) {
    println!("Symbol: {} | Interval: {}", data.symbol, data.interval);
    println!("Candles: {}", data.len());

    if let (Some(first), Some(last)) = (data.data.first(), data.data.last()) {
        println!("\nTime Range:");
        println!("  From: {}", first.timestamp);
        println!("  To:   {}", last.timestamp);

        println!("\nLatest Candle:");
        println!("  Open:   ${:.2}", last.open);
        println!("  High:   ${:.2}", last.high);
        println!("  Low:    ${:.2}", last.low);
        println!("  Close:  ${:.2}", last.close);
        println!("  Volume: {:.2}", last.volume);

        let returns = data.returns();
        if !returns.is_empty() {
            let volatility = {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 =
                    returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                variance.sqrt()
            };

            println!("\nStatistics:");
            println!("  Returns:");
            println!("    Min:  {:.2}%", returns.iter().cloned().fold(f64::INFINITY, f64::min));
            println!("    Max:  {:.2}%", returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            println!("    Avg:  {:.2}%", returns.iter().sum::<f64>() / returns.len() as f64);
            println!("    Std:  {:.2}%", volatility);
            println!("  Max Drawdown: {:.2}%", data.max_drawdown());
        }
    }

    // Print last 5 candles
    println!("\nLast 5 Candles:");
    println!("{:>20} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Time", "Open", "High", "Low", "Close", "Volume");
    println!("{}", "-".repeat(84));

    for candle in data.tail(5) {
        println!(
            "{:>20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.0}",
            candle.timestamp.format("%Y-%m-%d %H:%M"),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        );
    }
}

fn save_to_csv(data: &OHLCVSeries, path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;
    writeln!(file, "timestamp,open,high,low,close,volume")?;

    for candle in &data.data {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            candle.timestamp.timestamp(),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        )?;
    }

    Ok(())
}
