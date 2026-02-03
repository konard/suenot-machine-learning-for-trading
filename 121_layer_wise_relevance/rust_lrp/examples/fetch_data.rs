//! Example: Fetch market data from Bybit

use clap::Parser;
use rust_lrp::api::BybitClient;
use std::fs::File;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Trading symbols (comma-separated)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbols: String,

    /// Candle interval in minutes
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "1000")]
    limit: u32,

    /// Output file path
    #[arg(short, long, default_value = "market_data.json")]
    output: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    println!("Fetching market data...");
    println!("  Symbols: {}", args.symbols);
    println!("  Interval: {} min", args.interval);
    println!("  Limit: {} candles", args.limit);

    let client = BybitClient::new();
    let symbols: Vec<&str> = args.symbols.split(',').collect();

    let mut all_data = Vec::new();

    for symbol in &symbols {
        println!("\nFetching {}...", symbol);

        match client.fetch_klines(symbol, &args.interval, args.limit, None, None).await {
            Ok(klines) => {
                println!("  Retrieved {} candles", klines.len());

                if let Some(first) = klines.first() {
                    println!("  First: {} @ {:.2}", first.timestamp, first.close);
                }
                if let Some(last) = klines.last() {
                    println!("  Last:  {} @ {:.2}", last.timestamp, last.close);
                }

                all_data.push(serde_json::json!({
                    "symbol": symbol,
                    "interval": &args.interval,
                    "klines": klines,
                }));
            }
            Err(e) => {
                eprintln!("  Error fetching {}: {}", symbol, e);
            }
        }
    }

    // Save to file
    let json = serde_json::to_string_pretty(&all_data)?;
    let mut file = File::create(&args.output)?;
    file.write_all(json.as_bytes())?;

    println!("\nData saved to: {}", args.output);

    Ok(())
}
