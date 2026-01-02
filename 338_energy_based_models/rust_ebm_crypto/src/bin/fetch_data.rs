//! Fetch OHLCV data from Bybit API
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 1000

use clap::Parser;
use log::info;
use rust_ebm_crypto::data::{BybitClient, OhlcvData};

#[derive(Parser, Debug)]
#[command(author, version, about = "Fetch OHLCV data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "1000")]
    limit: usize,

    /// Output CSV file path
    #[arg(short, long)]
    output: Option<String>,

    /// Use testnet
    #[arg(long)]
    testnet: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::parse();

    info!(
        "Fetching {} candles for {} ({} interval)",
        args.limit, args.symbol, args.interval
    );

    // Create client
    let client = if args.testnet {
        BybitClient::testnet()
    } else {
        BybitClient::public()
    };

    // Fetch data
    let data = if args.limit > 1000 {
        info!("Fetching historical data with pagination...");
        client.get_historical_klines(&args.symbol, &args.interval, args.limit, None)?
    } else {
        client.get_klines(&args.symbol, &args.interval, args.limit as u32, None, None)?
    };

    info!("Fetched {} candles", data.len());

    // Print summary
    if !data.is_empty() {
        let first = &data.data[0];
        let last = data.data.last().unwrap();

        println!("\n=== Data Summary ===");
        println!("Symbol:     {}", data.symbol);
        println!("Interval:   {}", data.interval);
        println!("Candles:    {}", data.len());
        println!(
            "Period:     {} to {}",
            first.datetime().format("%Y-%m-%d %H:%M"),
            last.datetime().format("%Y-%m-%d %H:%M")
        );
        println!("Open:       {:.4}", first.open);
        println!("High:       {:.4}", data.highs().iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        println!("Low:        {:.4}", data.lows().iter().cloned().fold(f64::INFINITY, f64::min));
        println!("Close:      {:.4}", last.close);

        let returns = data.returns();
        if !returns.is_empty() {
            let total_return: f64 = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;
            println!("Total Return: {:.2}%", total_return * 100.0);
        }
    }

    // Save to CSV
    if let Some(output_path) = args.output {
        info!("Saving to {}", output_path);
        data.to_csv(&output_path)?;
        println!("\nData saved to: {}", output_path);
    }

    Ok(())
}
