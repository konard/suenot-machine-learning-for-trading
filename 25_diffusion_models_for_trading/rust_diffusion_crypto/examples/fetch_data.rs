//! Example: Fetch historical cryptocurrency data from Bybit.
//!
//! Usage:
//! ```bash
//! cargo run --example fetch_data -- --symbol BTCUSDT --interval 60 --days 90
//! ```

use anyhow::Result;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use diffusion_crypto::data::{BybitClient, OHLCVDataset};

#[derive(Parser)]
#[command(name = "fetch_data")]
#[command(about = "Fetch historical cryptocurrency data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval (1, 5, 15, 30, 60, 120, 240, D, W)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of days to fetch
    #[arg(short, long, default_value = "90")]
    days: u32,

    /// Output file path
    #[arg(short, long, default_value = "data/ohlcv.csv")]
    output: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    info!(
        "Fetching {} {} candles for {} days...",
        args.symbol, args.interval, args.days
    );

    // Create client and fetch data
    let client = BybitClient::new();
    let data = client
        .fetch_historical_klines(&args.symbol, &args.interval, args.days)
        .await?;

    info!("Fetched {} candles", data.len());

    // Create dataset
    let dataset = OHLCVDataset::new(data, args.symbol.clone(), args.interval.clone());

    // Create output directory if needed
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Save to CSV
    dataset.to_csv(&args.output)?;

    info!("Data saved to: {}", args.output);
    info!("Symbol: {}", args.symbol);
    info!("Interval: {}", args.interval);
    info!("Records: {}", dataset.len());

    if !dataset.is_empty() {
        let first = &dataset.data[0];
        let last = &dataset.data[dataset.len() - 1];
        info!(
            "Date range: {} to {}",
            chrono::DateTime::from_timestamp_millis(first.timestamp)
                .map(|dt| dt.to_string())
                .unwrap_or_default(),
            chrono::DateTime::from_timestamp_millis(last.timestamp)
                .map(|dt| dt.to_string())
                .unwrap_or_default()
        );
        info!("Price range: ${:.2} - ${:.2}", first.close, last.close);
    }

    Ok(())
}
