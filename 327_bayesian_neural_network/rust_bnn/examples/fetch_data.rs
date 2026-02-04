//! Example: Fetch cryptocurrency data from Bybit.
//!
//! Usage:
//!     cargo run --example fetch_data -- --symbol BTCUSDT --interval 60

use anyhow::Result;
use bayesian_nn_trading::prelude::*;
use clap::Parser;
use colored::Colorize;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fetch OHLCV data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Timeframe interval (1=1min, 5=5min, 60=1hour, D=1day)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "200")]
    limit: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(50).blue());
    println!(
        "{}",
        "Bybit Data Fetcher - Bayesian NN Trading".blue().bold()
    );
    println!("{}", "=".repeat(50).blue());

    // Initialize client
    let client = BybitClient::new();

    println!(
        "\n{} {} {} with interval {}",
        "Fetching".green(),
        args.limit,
        args.symbol.yellow(),
        args.interval.cyan()
    );

    // Fetch klines
    let klines = client
        .get_klines(&args.symbol, &args.interval, args.limit)
        .await?;

    println!("{} {} candles", "Fetched".green(), klines.len());

    if !klines.is_empty() {
        println!("\n{}", "Data Summary:".blue().bold());
        println!("  First candle: {}", klines[0].timestamp);
        println!("  Last candle:  {}", klines.last().unwrap().timestamp);

        // Price statistics
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;

        println!("\n{}", "Price Statistics:".blue().bold());
        println!("  Min:  ${:.2}", min_price);
        println!("  Max:  ${:.2}", max_price);
        println!("  Avg:  ${:.2}", avg_price);
        println!(
            "  Range: {:.2}%",
            (max_price - min_price) / avg_price * 100.0
        );

        // Volume statistics
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let total_volume: f64 = volumes.iter().sum();
        let avg_volume = total_volume / volumes.len() as f64;

        println!("\n{}", "Volume Statistics:".blue().bold());
        println!("  Total: {:.2}", total_volume);
        println!("  Avg:   {:.2}", avg_volume);

        // Show last 5 candles
        println!("\n{}", "Last 5 Candles:".blue().bold());
        for kline in klines.iter().rev().take(5) {
            let change = kline.return_pct() * 100.0;
            let change_str = if change >= 0.0 {
                format!("+{:.2}%", change).green()
            } else {
                format!("{:.2}%", change).red()
            };

            println!(
                "  {} | O: ${:.2} H: ${:.2} L: ${:.2} C: ${:.2} | {}",
                kline.timestamp.format("%Y-%m-%d %H:%M"),
                kline.open,
                kline.high,
                kline.low,
                kline.close,
                change_str
            );
        }
    }

    // Fetch current ticker
    println!("\n{}", "Current Ticker:".blue().bold());
    let ticker = client.get_ticker(&args.symbol).await?;
    println!("  Last Price: ${:.2}", ticker.last_price);
    println!("  24h Change: {}%", ticker.price_change_pct);
    println!("  24h Volume: {:.2}", ticker.volume_24h);
    println!(
        "  Spread: {:.2} bps",
        (ticker.ask_price - ticker.bid_price) / ticker.last_price * 10000.0
    );

    println!("\n{}", "Done!".green().bold());

    Ok(())
}
