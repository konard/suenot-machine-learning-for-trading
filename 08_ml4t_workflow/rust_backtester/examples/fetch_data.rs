//! Example: Fetch historical data from Bybit
//!
//! This example demonstrates how to fetch and save historical candlestick data
//! from Bybit exchange.
//!
//! Usage:
//!   cargo run --example fetch_data -- --symbol BTCUSDT --timeframe 1h --days 30

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use rust_backtester::{
    api::BybitClient,
    models::Timeframe,
    utils::{save_candles_csv, save_candles_json},
};
use std::path::PathBuf;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "fetch_data")]
#[command(about = "Fetch historical cryptocurrency data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
    #[arg(short, long, default_value = "1h")]
    timeframe: String,

    /// Number of days to fetch
    #[arg(short, long, default_value_t = 30)]
    days: i64,

    /// Output directory
    #[arg(short, long, default_value = "data")]
    output: PathBuf,

    /// Output format (json, csv)
    #[arg(short, long, default_value = "csv")]
    format: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = Args::parse();

    // Parse timeframe
    let timeframe = Timeframe::from_str(&args.timeframe)
        .ok_or_else(|| anyhow::anyhow!("Invalid timeframe: {}", args.timeframe))?;

    info!(
        "Fetching {} {} data for the last {} days",
        args.symbol, timeframe, args.days
    );

    // Create API client
    let client = BybitClient::new();

    // Calculate time range
    let end = Utc::now();
    let start = end - Duration::days(args.days);

    // Fetch data
    let candles = client
        .get_historical_klines(&args.symbol, timeframe, start, end)
        .await?;

    info!("Fetched {} candles", candles.len());

    if candles.is_empty() {
        println!("No data returned. Please check symbol and timeframe.");
        return Ok(());
    }

    // Print summary
    let first = candles.first().unwrap();
    let last = candles.last().unwrap();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     DATA SUMMARY                             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Symbol:      {:<48} ║", args.symbol);
    println!("║ Timeframe:   {:<48} ║", timeframe);
    println!("║ Candles:     {:<48} ║", candles.len());
    println!("║ From:        {:<48} ║", first.timestamp.format("%Y-%m-%d %H:%M"));
    println!("║ To:          {:<48} ║", last.timestamp.format("%Y-%m-%d %H:%M"));
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ First Close: ${:<47.2} ║", first.close);
    println!("║ Last Close:  ${:<47.2} ║", last.close);
    println!("║ Change:      {:>+47.2}% ║", (last.close / first.close - 1.0) * 100.0);
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Save to file
    let filename = format!(
        "{}_{}_{}d.{}",
        args.symbol.to_lowercase(),
        timeframe,
        args.days,
        args.format
    );
    let output_path = args.output.join(&filename);

    match args.format.as_str() {
        "json" => {
            save_candles_json(&candles, &output_path)?;
        }
        "csv" => {
            save_candles_csv(&candles, &output_path)?;
        }
        _ => {
            anyhow::bail!("Unsupported format: {}", args.format);
        }
    }

    info!("Saved data to {}", output_path.display());
    println!("Data saved to: {}", output_path.display());

    Ok(())
}
