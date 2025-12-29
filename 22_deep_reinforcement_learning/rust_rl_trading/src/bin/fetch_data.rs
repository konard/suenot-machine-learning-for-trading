//! Fetch historical data from Bybit and save to CSV.

use anyhow::Result;
use chrono::{Duration, Utc};
use rust_rl_trading::data::{BybitClient, Interval};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let symbol = args.get(1).map(|s| s.as_str()).unwrap_or("BTCUSDT");
    let days = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(365);
    let interval = args.get(3).map(|s| s.as_str()).unwrap_or("60");

    println!("Fetching {} data for {} (last {} days, {} min interval)...",
             symbol, symbol, days, interval);

    // Create Bybit client
    let client = BybitClient::new();

    // Calculate time range
    let end = Utc::now();
    let start = end - Duration::days(days);

    // Map interval string to enum
    let interval_enum = match interval {
        "1" => Interval::Min1,
        "5" => Interval::Min5,
        "15" => Interval::Min15,
        "30" => Interval::Min30,
        "60" => Interval::Hour1,
        "240" => Interval::Hour4,
        "D" => Interval::Day1,
        _ => Interval::Hour1,
    };

    // Fetch data
    let candles = client
        .get_historical_klines(symbol, interval_enum, start, end)
        .await?;

    println!("Fetched {} candles", candles.len());

    if candles.is_empty() {
        println!("No data fetched. Please check the symbol and try again.");
        return Ok(());
    }

    // Save to CSV
    let filename = format!("data/{}_{}_{}d.csv", symbol, interval, days);

    // Create data directory if it doesn't exist
    std::fs::create_dir_all("data")?;

    let mut writer = csv::Writer::from_path(&filename)?;

    // Write header
    writer.write_record(&[
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ])?;

    // Write data
    for candle in &candles {
        writer.write_record(&[
            candle.timestamp.to_rfc3339(),
            candle.symbol.clone(),
            candle.open.to_string(),
            candle.high.to_string(),
            candle.low.to_string(),
            candle.close.to_string(),
            candle.volume.to_string(),
            candle.turnover.to_string(),
        ])?;
    }

    writer.flush()?;

    println!("Data saved to {}", filename);
    println!("First candle: {} - Open: {:.2}, Close: {:.2}",
             candles.first().unwrap().timestamp,
             candles.first().unwrap().open,
             candles.first().unwrap().close);
    println!("Last candle: {} - Open: {:.2}, Close: {:.2}",
             candles.last().unwrap().timestamp,
             candles.last().unwrap().open,
             candles.last().unwrap().close);

    Ok(())
}
