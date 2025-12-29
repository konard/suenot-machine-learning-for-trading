//! Example: Fetch cryptocurrency data from Bybit.
//!
//! Usage:
//!   cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --days 90

use anyhow::Result;
use clap::Parser;
use gnn_crypto::data::BybitClient;
use std::fs::{self, File};
use std::io::Write;
use tracing::info;

#[derive(Parser)]
#[command(name = "fetch_data")]
#[command(about = "Fetch cryptocurrency data from Bybit")]
struct Args {
    /// Trading symbols (comma-separated)
    #[arg(short, long, default_value = "BTCUSDT,ETHUSDT,SOLUSDT,AVAXUSDT,MATICUSDT")]
    symbols: String,

    /// Kline interval (1, 5, 15, 30, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of days to fetch
    #[arg(short, long, default_value = "90")]
    days: u32,

    /// Output directory
    #[arg(short, long, default_value = "data")]
    output: String,

    /// Fetch top N symbols by volume
    #[arg(long)]
    top_n: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Create output directory
    fs::create_dir_all(&args.output)?;

    let client = BybitClient::new();

    // Get symbols to fetch
    let symbols: Vec<String> = if let Some(n) = args.top_n {
        info!("Fetching top {} symbols by volume...", n);
        client.get_top_symbols_by_volume(n).await?
    } else {
        args.symbols.split(',').map(|s| s.to_string()).collect()
    };

    info!("Fetching data for {} symbols", symbols.len());
    info!("Interval: {}, Days: {}", args.interval, args.days);

    let mut success_count = 0;
    let mut failed_symbols = Vec::new();

    for symbol in &symbols {
        info!("Fetching {}...", symbol);

        match client
            .fetch_historical_klines(symbol, &args.interval, args.days)
            .await
        {
            Ok(data) => {
                if data.is_empty() {
                    info!("  No data for {}", symbol);
                    failed_symbols.push(symbol.clone());
                    continue;
                }

                // Save to CSV
                let filename = format!("{}/{}_{}.csv", args.output, symbol, args.interval);
                save_to_csv(&data, &filename)?;

                info!(
                    "  Saved {} candles to {}",
                    data.len(),
                    filename
                );
                success_count += 1;
            }
            Err(e) => {
                info!("  Error fetching {}: {}", symbol, e);
                failed_symbols.push(symbol.clone());
            }
        }

        // Small delay to avoid rate limiting
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    info!("\nSummary:");
    info!("  Successfully fetched: {}/{}", success_count, symbols.len());

    if !failed_symbols.is_empty() {
        info!("  Failed symbols: {:?}", failed_symbols);
    }

    // Save symbol list
    let symbols_file = format!("{}/symbols.txt", args.output);
    let successful_symbols: Vec<&String> = symbols
        .iter()
        .filter(|s| !failed_symbols.contains(s))
        .collect();

    let mut file = File::create(&symbols_file)?;
    for symbol in &successful_symbols {
        writeln!(file, "{}", symbol)?;
    }
    info!("  Symbol list saved to {}", symbols_file);

    Ok(())
}

fn save_to_csv(data: &[gnn_crypto::OHLCV], filename: &str) -> Result<()> {
    let mut file = File::create(filename)?;
    writeln!(file, "timestamp,open,high,low,close,volume")?;

    for candle in data {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        )?;
    }

    Ok(())
}
