//! Standalone binary for fetching data from Bybit
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --days 30

use anyhow::Result;
use clap::Parser;
use tracing::info;
use tracing_subscriber::FmtSubscriber;

use rust_dcgan_bybit::data::BybitClient;

/// Fetch cryptocurrency data from Bybit exchange
#[derive(Parser)]
#[command(name = "fetch_data")]
#[command(about = "Fetch OHLCV data from Bybit API")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of days of historical data to fetch
    #[arg(short, long, default_value = "30")]
    days: u32,

    /// Output CSV file path
    #[arg(short, long)]
    output: Option<String>,

    /// Use Bybit testnet instead of mainnet
    #[arg(long)]
    testnet: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    info!(
        "Fetching {} days of {} interval data for {}",
        args.days, args.interval, args.symbol
    );

    // Create client
    let client = if args.testnet {
        info!("Using Bybit testnet");
        BybitClient::testnet()
    } else {
        BybitClient::new()
    };

    // Calculate time range
    let end_time = chrono::Utc::now().timestamp_millis();
    let start_time = end_time - (args.days as i64 * 24 * 60 * 60 * 1000);

    info!(
        "Time range: {} to {}",
        chrono::DateTime::from_timestamp_millis(start_time)
            .unwrap()
            .format("%Y-%m-%d %H:%M"),
        chrono::DateTime::from_timestamp_millis(end_time)
            .unwrap()
            .format("%Y-%m-%d %H:%M")
    );

    // Fetch data
    let dataset = client
        .fetch_historical_klines(&args.symbol, &args.interval, start_time, end_time)
        .await?;

    info!("Fetched {} klines", dataset.len());

    // Show sample data
    if !dataset.data.is_empty() {
        let first = &dataset.data[0];
        let last = dataset.data.last().unwrap();
        info!(
            "First: {} - O:{:.2} H:{:.2} L:{:.2} C:{:.2}",
            first.datetime().format("%Y-%m-%d %H:%M"),
            first.open,
            first.high,
            first.low,
            first.close
        );
        info!(
            "Last:  {} - O:{:.2} H:{:.2} L:{:.2} C:{:.2}",
            last.datetime().format("%Y-%m-%d %H:%M"),
            last.open,
            last.high,
            last.low,
            last.close
        );
    }

    // Calculate basic statistics
    let closes: Vec<f64> = dataset.close_prices();
    if closes.len() > 1 {
        let returns = dataset.calculate_log_returns();
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std_return = variance.sqrt();

        info!("Statistics:");
        info!("  Mean log return: {:.6}", mean_return);
        info!("  Std log return:  {:.6}", std_return);
        info!(
            "  Annualized vol:  {:.2}%",
            std_return * (365.0 * 24.0_f64).sqrt() * 100.0
        );
    }

    // Save to file
    let output_path = args.output.unwrap_or_else(|| {
        format!(
            "data/{}_{}_{}d.csv",
            args.symbol, args.interval, args.days
        )
    });

    // Create directory if needed
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    dataset.save_csv(&output_path)?;
    info!("Saved data to {}", output_path);

    Ok(())
}
