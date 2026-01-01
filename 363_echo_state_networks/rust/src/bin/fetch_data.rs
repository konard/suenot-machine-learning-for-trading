//! Fetch historical data from Bybit
//!
//! Usage: cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --days 30

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use esn_trading::api::BybitClient;
use std::fs::File;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fetch historical data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Interval in minutes (1, 5, 15, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of days to fetch
    #[arg(short, long, default_value = "30")]
    days: i64,

    /// Output file path
    #[arg(short, long)]
    output: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("Fetching {} {} data for {} days...", args.symbol, args.interval, args.days);

    let client = BybitClient::new();

    // Calculate time range
    let end_time = Utc::now().timestamp_millis();
    let start_time = (Utc::now() - Duration::days(args.days)).timestamp_millis();

    // Fetch data
    let klines = client.get_historical_klines(
        &args.symbol,
        &args.interval,
        start_time,
        end_time,
    ).await?;

    println!("Fetched {} klines", klines.len());

    // Save to CSV
    let output_path = args.output.unwrap_or_else(|| {
        format!("{}_{}_{}d.csv", args.symbol, args.interval, args.days)
    });

    let mut file = File::create(&output_path)?;
    writeln!(file, "timestamp,open,high,low,close,volume,turnover")?;

    for kline in &klines {
        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            kline.start_time,
            kline.open,
            kline.high,
            kline.low,
            kline.close,
            kline.volume,
            kline.turnover
        )?;
    }

    println!("Saved to {}", output_path);

    // Print summary
    if !klines.is_empty() {
        let first = klines.last().unwrap();
        let last = klines.first().unwrap();
        println!("\nData Summary:");
        println!("  Start: {} ({})", first.start_time, format_timestamp(first.start_time));
        println!("  End:   {} ({})", last.start_time, format_timestamp(last.start_time));
        println!("  Open:  {:.2}", first.open);
        println!("  Close: {:.2}", last.close);
        println!("  High:  {:.2}", klines.iter().map(|k| k.high).fold(f64::MIN, f64::max));
        println!("  Low:   {:.2}", klines.iter().map(|k| k.low).fold(f64::MAX, f64::min));
    }

    Ok(())
}

fn format_timestamp(ts: i64) -> String {
    use chrono::{TimeZone, Utc};
    Utc.timestamp_millis_opt(ts)
        .single()
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| "Invalid".to_string())
}
