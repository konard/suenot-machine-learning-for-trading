//! Fetch cryptocurrency and synthetic data for operator learning.
//!
//! Fetches OHLCV, order book, and funding rate data from Bybit,
//! and generates synthetic PDE training data.
//!
//! # Usage
//! ```bash
//! cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 500
//! cargo run --bin fetch_data -- --synthetic --samples 2000
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use operator_learning_trading::data;

#[derive(Parser, Debug)]
#[command(name = "fetch_data", about = "Fetch data for operator learning")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval (1, 5, 15, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value_t = 500)]
    limit: usize,

    /// Also fetch order book
    #[arg(long, default_value_t = false)]
    orderbook: bool,

    /// Also fetch funding rates
    #[arg(long, default_value_t = false)]
    funding: bool,

    /// Generate synthetic data instead of fetching
    #[arg(long, default_value_t = false)]
    synthetic: bool,

    /// Number of synthetic samples
    #[arg(long, default_value_t = 1000)]
    samples: usize,

    /// Output directory
    #[arg(short, long, default_value = "data")]
    output: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("{}", "═══════════════════════════════════════════════════════".blue());
    println!("{}", "  Operator Learning Data Fetcher - Chapter 152".blue().bold());
    println!("{}", "═══════════════════════════════════════════════════════".blue());

    std::fs::create_dir_all(&args.output)?;

    if args.synthetic {
        generate_synthetic_data(&args)?;
    } else {
        fetch_bybit_data(&args)?;
    }

    println!("\n{}", "Done!".green().bold());
    Ok(())
}

fn fetch_bybit_data(args: &Args) -> Result<()> {
    // Fetch OHLCV
    println!(
        "\n{} OHLCV for {} (interval={}, limit={})...",
        "Fetching".yellow(),
        args.symbol.cyan(),
        args.interval,
        args.limit
    );

    match data::fetch_bybit_klines(&args.symbol, &args.interval, args.limit) {
        Ok(candles) => {
            println!("  Received {} candles", candles.len());
            if let Some(first) = candles.first() {
                println!("  First: ts={}, O={:.2}, H={:.2}, L={:.2}, C={:.2}",
                    first.timestamp, first.open, first.high, first.low, first.close);
            }
            if let Some(last) = candles.last() {
                println!("  Last:  ts={}, O={:.2}, H={:.2}, L={:.2}, C={:.2}",
                    last.timestamp, last.open, last.high, last.low, last.close);
            }

            // Save to CSV
            let path = format!("{}/{}_klines.csv", args.output, args.symbol);
            let mut wtr = csv::Writer::from_path(&path)?;
            wtr.write_record(["timestamp", "open", "high", "low", "close", "volume"])?;
            for c in &candles {
                wtr.write_record(&[
                    c.timestamp.to_string(),
                    c.open.to_string(),
                    c.high.to_string(),
                    c.low.to_string(),
                    c.close.to_string(),
                    c.volume.to_string(),
                ])?;
            }
            wtr.flush()?;
            println!("  Saved to {}", path.green());

            // Prepare operator learning data
            let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
            if let Some((branch, trunk, targets)) =
                data::prepare_price_operator_data(&prices, 60, 5)
            {
                println!(
                    "\n  {} DeepONet training data: {} samples",
                    "Prepared".yellow(),
                    branch.nrows()
                );
                println!(
                    "    Branch shape: ({}, {})",
                    branch.nrows(),
                    branch.ncols()
                );
                println!(
                    "    Trunk shape:  ({}, {})",
                    trunk.nrows(),
                    trunk.ncols()
                );
            }
        }
        Err(e) => {
            println!("  {} {}", "Error:".red(), e);
        }
    }

    // Fetch order book
    if args.orderbook {
        println!(
            "\n{} order book for {}...",
            "Fetching".yellow(),
            args.symbol.cyan()
        );
        match data::fetch_bybit_orderbook(&args.symbol, 25) {
            Ok(book) => {
                println!("  Mid price: {:.2}", book.mid_price);
                println!("  Bid levels: {}, Ask levels: {}", book.bids.len(), book.asks.len());
                if let Some(best_bid) = book.bids.first() {
                    println!("  Best bid: {:.2} x {:.4}", best_bid.0, best_bid.1);
                }
                if let Some(best_ask) = book.asks.first() {
                    println!("  Best ask: {:.2} x {:.4}", best_ask.0, best_ask.1);
                }
            }
            Err(e) => println!("  {} {}", "Error:".red(), e),
        }
    }

    // Fetch funding rates
    if args.funding {
        println!(
            "\n{} funding rates for {}...",
            "Fetching".yellow(),
            args.symbol.cyan()
        );
        match data::fetch_bybit_funding_rates(&args.symbol, 50) {
            Ok(rates) => {
                println!("  Received {} funding rate entries", rates.len());
                if let Some(&(ts, rate)) = rates.last() {
                    println!("  Latest: ts={}, rate={:.6} ({:.4}%)", ts, rate, rate * 100.0);
                }
                let avg_rate: f64 = rates.iter().map(|&(_, r)| r).sum::<f64>() / rates.len().max(1) as f64;
                println!("  Average rate: {:.6} ({:.4}%)", avg_rate, avg_rate * 100.0);
            }
            Err(e) => println!("  {} {}", "Error:".red(), e),
        }
    }

    Ok(())
}

fn generate_synthetic_data(args: &Args) -> Result<()> {
    println!(
        "\n{} synthetic Black-Scholes data ({} samples)...",
        "Generating".yellow(),
        args.samples
    );

    let (branch, trunk, targets) =
        data::generate_bs_training_data(args.samples, 50, 20);

    println!("  Branch (vol functions): ({}, {})", branch.nrows(), branch.ncols());
    println!("  Trunk (eval points):    ({}, {})", trunk.nrows(), trunk.ncols());
    println!("  Targets (prices):       ({})", targets.len());

    // Statistics
    let target_mean = targets.mean().unwrap_or(0.0);
    let target_std = targets.std(0.0);
    println!("  Target stats: mean={:.4}, std={:.4}", target_mean, target_std);

    println!(
        "\n{} synthetic yield curve data ({} samples)...",
        "Generating".yellow(),
        args.samples
    );

    let (curves_today, curves_future) =
        data::generate_yield_curve_data(args.samples, 30);

    println!("  Curves today:  ({}, {})", curves_today.nrows(), curves_today.ncols());
    println!("  Curves future: ({}, {})", curves_future.nrows(), curves_future.ncols());

    // Yield curve statistics
    let avg_level = curves_today.mean().unwrap_or(0.0);
    let avg_change = (&curves_future - &curves_today).mean().unwrap_or(0.0);
    println!("  Average yield level: {:.4} ({:.2}%)", avg_level, avg_level * 100.0);
    println!("  Average yield change: {:.6} ({:.4}%)", avg_change, avg_change * 100.0);

    Ok(())
}
