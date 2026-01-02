//! Fetch data from Bybit exchange
//!
//! This binary fetches historical OHLCV data from Bybit API.

use anyhow::Result;
use clap::Parser;
use tracing::info;

use inception_time_trading::{BybitClient, setup_logging};

#[derive(Parser)]
#[command(name = "fetch_data")]
#[command(about = "Fetch historical data from Bybit exchange")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    #[arg(short, long, default_value = "15")]
    interval: String,

    /// Number of days of history to fetch
    #[arg(short, long, default_value = "90")]
    days: u32,

    /// Output file path
    #[arg(short, long)]
    output: Option<String>,

    /// Use testnet instead of mainnet
    #[arg(long)]
    testnet: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    setup_logging("info")?;

    println!("\nBybit Data Fetcher");
    println!("═══════════════════════════════════════════════════════════════\n");

    let client = if args.testnet {
        info!("Using Bybit testnet");
        BybitClient::testnet()
    } else {
        BybitClient::new()
    };

    // Calculate time range
    let end_time = chrono::Utc::now().timestamp_millis();
    let start_time = end_time - (args.days as i64 * 24 * 60 * 60 * 1000);

    println!("[INFO] Symbol: {}", args.symbol);
    println!("[INFO] Interval: {}m", args.interval);
    println!("[INFO] Days: {}", args.days);
    println!("[INFO] Fetching data...\n");

    let dataset = client
        .fetch_historical_klines(&args.symbol, &args.interval, start_time, end_time)
        .await?;

    println!("[SUCCESS] Fetched {} candles", dataset.len());

    if let Some((start, end)) = dataset.time_range() {
        println!("[INFO] Time range: {} to {}", start, end);
    }

    // Display sample data
    if let Some(last) = dataset.data.last() {
        println!("\n[LATEST CANDLE]");
        println!("  Timestamp: {}", last.datetime());
        println!("  Open:      ${:.2}", last.open);
        println!("  High:      ${:.2}", last.high);
        println!("  Low:       ${:.2}", last.low);
        println!("  Close:     ${:.2}", last.close);
        println!("  Volume:    {:.4}", last.volume);
    }

    // Save to file
    let output_path = args.output.unwrap_or_else(|| {
        format!(
            "data/{}_{}_{}d.csv",
            args.symbol.to_lowercase(),
            args.interval,
            args.days
        )
    });

    // Create data directory if needed
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    dataset.to_csv(&output_path)?;
    println!("\n[SUCCESS] Saved to {}", output_path);

    println!("\n═══════════════════════════════════════════════════════════════\n");

    Ok(())
}
