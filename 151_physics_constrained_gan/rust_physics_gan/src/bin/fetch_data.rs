//! Fetch cryptocurrency OHLCV data from Bybit API
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 1000
//!   cargo run --bin fetch_data -- --symbol ETHUSDT --interval 240 --limit 5000 --output eth_data.csv

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use physics_constrained_gan::{
    bybit::BybitClient,
    compute_log_returns,
    statistics,
    Candle,
};

#[derive(Parser)]
#[command(name = "fetch_data", about = "Fetch crypto OHLCV data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "1000")]
    limit: usize,

    /// Output CSV file path
    #[arg(short, long, default_value = "data.csv")]
    output: String,

    /// Also print summary statistics
    #[arg(long, default_value = "true")]
    stats: bool,
}

fn save_candles_csv(candles: &[Candle], path: &str) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(&["timestamp", "open", "high", "low", "close", "volume"])?;

    for c in candles {
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
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "═══════════════════════════════════════════════════".blue().bold());
    println!("{}", " Physics-Constrained GAN - Data Fetcher".blue().bold());
    println!("{}", "═══════════════════════════════════════════════════".blue().bold());

    println!("\nFetching {} {} candles for {}...",
             args.limit, args.interval, args.symbol.green());

    let client = BybitClient::new();

    let candles = if args.limit > 1000 {
        client.fetch_extended(&args.symbol, &args.interval, args.limit)?
    } else {
        client.fetch_klines(&args.symbol, &args.interval, args.limit)?
    };

    println!("Fetched {} candles", candles.len().to_string().green());

    if candles.is_empty() {
        println!("{}", "No data returned. Check symbol and network connection.".red());
        return Ok(());
    }

    // Print date range
    let first_ts = candles.first().unwrap().timestamp;
    let last_ts = candles.last().unwrap().timestamp;
    let first_dt = chrono::NaiveDateTime::from_timestamp_millis(first_ts)
        .unwrap_or_default();
    let last_dt = chrono::NaiveDateTime::from_timestamp_millis(last_ts)
        .unwrap_or_default();
    println!("Date range: {} to {}", first_dt, last_dt);

    // Price range
    let min_price = candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    let max_price = candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
    println!("Price range: {:.2} - {:.2}", min_price, max_price);

    // Save to CSV
    save_candles_csv(&candles, &args.output)?;
    println!("Data saved to {}", args.output.green());

    // Compute and print statistics
    if args.stats {
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let returns = compute_log_returns(&prices);

        if returns.is_empty() {
            println!("{}", "Not enough data for statistics".yellow());
            return Ok(());
        }

        let stats = statistics::compute_statistics(&returns);

        println!("\n{}", "Return Statistics:".yellow().bold());
        println!("  Mean return:        {:.8}", stats.mean);
        println!("  Std deviation:      {:.6}", stats.std);
        println!("  Skewness:           {:.4}", stats.skewness);
        println!("  Excess kurtosis:    {:.4}", stats.kurtosis);
        println!("  Min return:         {:.6}", stats.min);
        println!("  Max return:         {:.6}", stats.max);
        println!("  ACF(|r|, lag=1):    {:.4}", stats.acf_abs_lag1);
        println!("  ACF(|r|, lag=5):    {:.4}", stats.acf_abs_lag5);
        println!("  ACF(r^2, lag=1):    {:.4}", stats.acf_sq_lag1);
        println!("  Leverage corr:      {:.4}", stats.leverage_corr);

        // Interpret the stylized facts
        println!("\n{}", "Stylized Facts Check:".yellow().bold());
        let kurt_ok = stats.kurtosis > 0.0;
        let vol_cluster_ok = stats.acf_abs_lag1 > 0.05;
        let leverage_ok = stats.leverage_corr < 0.0;
        let no_acf_ok = statistics::autocorrelation(&returns, 1).abs() < 0.1;

        println!("  Fat tails (kurtosis > 0):     {} (kurtosis = {:.2})",
                 if kurt_ok { "PASS".green() } else { "FAIL".red() },
                 stats.kurtosis);
        println!("  Vol clustering (ACF|r| > 0):  {} (ACF = {:.4})",
                 if vol_cluster_ok { "PASS".green() } else { "FAIL".red() },
                 stats.acf_abs_lag1);
        println!("  Leverage effect (corr < 0):   {} (corr = {:.4})",
                 if leverage_ok { "PASS".green() } else { "FAIL".red() },
                 stats.leverage_corr);
        println!("  No return autocorr (|ACF|<0.1): {} (ACF(1) = {:.4})",
                 if no_acf_ok { "PASS".green() } else { "FAIL".red() },
                 statistics::autocorrelation(&returns, 1));
    }

    println!("\n{}", "Done!".green().bold());
    Ok(())
}
