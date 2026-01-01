//! Example: Fetch historical data from Bybit
//!
//! This example demonstrates how to fetch OHLCV data from Bybit exchange.
//!
//! Usage:
//!   cargo run --example fetch_data -- --symbol BTCUSDT --interval 1h --days 365

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use convnext_trading::data::{BybitClient, Interval};

#[derive(Parser)]
#[command(name = "fetch_data")]
#[command(about = "Fetch historical OHLCV data from Bybit")]
struct Args {
    /// Trading pair symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candlestick interval (1m, 5m, 15m, 1h, 4h, 1d)
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of days to fetch
    #[arg(short, long, default_value = "30")]
    days: u32,

    /// Output file path (optional)
    #[arg(short, long)]
    output: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    info!("=== Bybit Data Fetcher ===");
    info!("Symbol: {}", args.symbol);
    info!("Interval: {}", args.interval);
    info!("Days: {}", args.days);

    // Parse interval
    let interval = Interval::from_str(&args.interval)?;

    // Calculate time range
    let end_time = Utc::now();
    let start_time = end_time - chrono::Duration::days(args.days as i64);

    info!(
        "Fetching data from {} to {}",
        start_time.format("%Y-%m-%d %H:%M"),
        end_time.format("%Y-%m-%d %H:%M")
    );

    // Create client and fetch data
    let client = BybitClient::new();
    let candles = client
        .get_klines(
            &args.symbol,
            interval,
            start_time.timestamp_millis(),
            end_time.timestamp_millis(),
        )
        .await?;

    info!("Fetched {} candles", candles.len());

    // Print sample of data
    if !candles.is_empty() {
        info!("\nFirst 5 candles:");
        for candle in candles.iter().take(5) {
            println!(
                "  {} | O:{:.2} H:{:.2} L:{:.2} C:{:.2} V:{:.0}",
                candle.datetime().format("%Y-%m-%d %H:%M"),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume
            );
        }

        info!("\nLast 5 candles:");
        let start = candles.len().saturating_sub(5);
        for candle in candles.iter().skip(start) {
            println!(
                "  {} | O:{:.2} H:{:.2} L:{:.2} C:{:.2} V:{:.0}",
                candle.datetime().format("%Y-%m-%d %H:%M"),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume
            );
        }

        // Calculate basic statistics
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;

        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;

        info!("\nStatistics:");
        info!("  Price range: ${:.2} - ${:.2}", min_price, max_price);
        info!("  Average price: ${:.2}", avg_price);
        info!("  Average volume: {:.0}", avg_volume);
    }

    // Save to file if output specified
    if let Some(output_path) = args.output {
        // Create parent directories if needed
        if let Some(parent) = std::path::Path::new(&output_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&candles)?;
        std::fs::write(&output_path, json)?;
        info!("\nSaved {} candles to {}", candles.len(), output_path);
    } else {
        let default_path = format!(
            "data/{}_{}.json",
            args.symbol.to_lowercase(),
            args.interval
        );

        if let Some(parent) = std::path::Path::new(&default_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&candles)?;
        std::fs::write(&default_path, json)?;
        info!("\nSaved {} candles to {}", candles.len(), default_path);
    }

    info!("\n=== Done ===");
    Ok(())
}
