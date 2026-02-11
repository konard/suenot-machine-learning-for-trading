//! Fetch Data Binary
//!
//! Downloads market data from Bybit exchange and saves to CSV.
//! Supports both kline (OHLCV) and tick (trade) data.
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1 --limit 1000
//!   cargo run --bin fetch_data -- --symbol ETHUSDT --data-type ticks --limit 500

use clap::Parser;
use neural_ode_trading::data::{
    generate_synthetic_klines, generate_synthetic_ticks, BybitClient, MarketData, TickData,
};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "fetch_data")]
#[command(about = "Fetch cryptocurrency data from Bybit for Neural ODE trading")]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval (1, 3, 5, 15, 30, 60, 240, D, W)
    #[arg(short, long, default_value = "1")]
    interval: String,

    /// Number of data points to fetch
    #[arg(short, long, default_value_t = 1000)]
    limit: usize,

    /// Data type: klines or ticks
    #[arg(short, long, default_value = "klines")]
    data_type: String,

    /// Output directory
    #[arg(short, long, default_value = "data")]
    output: String,

    /// Use synthetic data instead of API
    #[arg(long)]
    synthetic: bool,
}

fn save_klines_csv(data: &[MarketData], path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["timestamp", "open", "high", "low", "close", "volume", "turnover"])?;

    for candle in data {
        wtr.write_record(&[
            candle.timestamp.to_string(),
            format!("{:.2}", candle.open),
            format!("{:.2}", candle.high),
            format!("{:.2}", candle.low),
            format!("{:.2}", candle.close),
            format!("{:.4}", candle.volume),
            format!("{:.2}", candle.turnover),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

fn save_ticks_csv(data: &[TickData], path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["timestamp_ms", "price", "size", "side", "time_delta_ms"])?;

    for tick in data {
        wtr.write_record(&[
            tick.timestamp_ms.to_string(),
            format!("{:.2}", tick.price),
            format!("{:.6}", tick.size),
            tick.side.clone(),
            format!("{:.1}", tick.time_delta_ms),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("========================================");
    println!("  Neural ODE Trading - Data Fetcher");
    println!("========================================");
    println!("  Symbol:    {}", args.symbol);
    println!("  Type:      {}", args.data_type);
    println!("  Limit:     {}", args.limit);
    println!("  Synthetic: {}", args.synthetic);
    println!("========================================\n");

    // Create output directory
    fs::create_dir_all(&args.output)?;

    match args.data_type.as_str() {
        "klines" => {
            let data = if args.synthetic {
                println!("Generating synthetic kline data...");
                generate_synthetic_klines(&args.symbol, args.limit)
            } else {
                println!("Fetching kline data from Bybit...");
                let client = BybitClient::new();
                match client.fetch_klines(&args.symbol, &args.interval, args.limit) {
                    Ok(data) => {
                        println!("Fetched {} candles from Bybit API", data.len());
                        data
                    }
                    Err(e) => {
                        println!("API error: {}. Using synthetic data.", e);
                        generate_synthetic_klines(&args.symbol, args.limit)
                    }
                }
            };

            let path = PathBuf::from(&args.output)
                .join(format!("{}_{}_klines.csv", args.symbol, args.interval));
            save_klines_csv(&data, &path)?;
            println!("\nSaved {} candles to {:?}", data.len(), path);

            // Print sample
            println!("\nSample data (first 5 rows):");
            println!(
                "{:<15} {:>12} {:>12} {:>12} {:>12} {:>12}",
                "Timestamp", "Open", "High", "Low", "Close", "Volume"
            );
            for candle in data.iter().take(5) {
                println!(
                    "{:<15} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
                    candle.timestamp, candle.open, candle.high, candle.low,
                    candle.close, candle.volume
                );
            }
        }

        "ticks" => {
            let data = if args.synthetic {
                println!("Generating synthetic tick data...");
                generate_synthetic_ticks(&args.symbol, args.limit)
            } else {
                println!("Fetching tick data from Bybit...");
                let client = BybitClient::new();
                match client.fetch_recent_trades(&args.symbol, args.limit) {
                    Ok(data) => {
                        println!("Fetched {} trades from Bybit API", data.len());
                        data
                    }
                    Err(e) => {
                        println!("API error: {}. Using synthetic data.", e);
                        generate_synthetic_ticks(&args.symbol, args.limit)
                    }
                }
            };

            let path =
                PathBuf::from(&args.output).join(format!("{}_ticks.csv", args.symbol));
            save_ticks_csv(&data, &path)?;
            println!("\nSaved {} ticks to {:?}", data.len(), path);

            // Print time gap statistics
            let time_gaps: Vec<f64> = data.iter().skip(1).map(|t| t.time_delta_ms).collect();
            if !time_gaps.is_empty() {
                let mean_gap = time_gaps.iter().sum::<f64>() / time_gaps.len() as f64;
                let min_gap = time_gaps.iter().cloned().fold(f64::MAX, f64::min);
                let max_gap = time_gaps.iter().cloned().fold(f64::MIN, f64::max);

                println!("\nTime gap statistics (irregular timestamps):");
                println!("  Mean gap:  {:.1} ms", mean_gap);
                println!("  Min gap:   {:.1} ms", min_gap);
                println!("  Max gap:   {:.1} ms", max_gap);
                println!(
                    "  Std gap:   {:.1} ms",
                    (time_gaps
                        .iter()
                        .map(|g| (g - mean_gap).powi(2))
                        .sum::<f64>()
                        / time_gaps.len() as f64)
                        .sqrt()
                );
            }
        }

        _ => {
            eprintln!("Unknown data type: {}. Use 'klines' or 'ticks'.", args.data_type);
            std::process::exit(1);
        }
    }

    println!("\nDone!");
    Ok(())
}
