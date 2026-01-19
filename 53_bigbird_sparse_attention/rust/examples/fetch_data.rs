//! Fetch Data Example
//!
//! Download market data from Bybit and save to CSV.
//!
//! Usage:
//!   cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --interval 60 --limit 1000

use bigbird_trading::api::{BybitClient, KlineInterval};
use bigbird_trading::data::DataLoader;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Trading symbols (comma-separated)
    #[arg(short, long, default_value = "BTCUSDT,ETHUSDT")]
    symbols: String,

    /// Kline interval in minutes (1, 5, 15, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of klines to fetch
    #[arg(short, long, default_value_t = 1000)]
    limit: u32,

    /// Output directory
    #[arg(short, long, default_value = "data")]
    output: PathBuf,

    /// Use testnet instead of mainnet
    #[arg(long)]
    testnet: bool,
}

fn parse_interval(s: &str) -> KlineInterval {
    match s {
        "1" => KlineInterval::Min1,
        "3" => KlineInterval::Min3,
        "5" => KlineInterval::Min5,
        "15" => KlineInterval::Min15,
        "30" => KlineInterval::Min30,
        "60" | "1h" => KlineInterval::Hour1,
        "120" | "2h" => KlineInterval::Hour2,
        "240" | "4h" => KlineInterval::Hour4,
        "D" | "1d" => KlineInterval::Day1,
        "W" | "1w" => KlineInterval::Week1,
        _ => KlineInterval::Hour1,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== BigBird Trading - Data Fetcher ===\n");

    // Parse symbols
    let symbols: Vec<&str> = args.symbols.split(',').map(|s| s.trim()).collect();
    let interval = parse_interval(&args.interval);

    println!("Configuration:");
    println!("  Symbols:  {:?}", symbols);
    println!("  Interval: {}", interval);
    println!("  Limit:    {}", args.limit);
    println!("  Output:   {:?}", args.output);
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Create client
    let client = if args.testnet {
        BybitClient::with_base_url("https://api-testnet.bybit.com")
    } else {
        BybitClient::new()
    };

    let loader = DataLoader::new();

    // Fetch data for each symbol
    for symbol in symbols {
        println!("Fetching {}...", symbol);

        match client
            .get_klines(symbol, interval, args.limit, None, None)
            .await
        {
            Ok(data) => {
                println!("  Received {} klines", data.len());

                if !data.is_empty() {
                    // Print sample data
                    let first = &data.klines[0];
                    let last = data.klines.last().unwrap();
                    println!(
                        "  Time range: {} to {}",
                        first.timestamp, last.timestamp
                    );
                    println!(
                        "  Price range: {:.2} - {:.2}",
                        data.klines.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
                        data.klines.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max)
                    );

                    // Save to CSV
                    let filename = args.output.join(format!(
                        "{}_{}.csv",
                        symbol.to_lowercase(),
                        interval.as_str()
                    ));
                    loader.save_csv(&data, &filename)?;
                    println!("  Saved to {:?}", filename);
                }
            }
            Err(e) => {
                eprintln!("  Error: {}", e);
            }
        }

        println!();
    }

    println!("Done!");
    Ok(())
}
