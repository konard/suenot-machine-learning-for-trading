//! Example: Fetch cryptocurrency data from Bybit and calculate trading features.
//!
//! Usage:
//!   cargo run --example fetch_data -- --symbol BTCUSDT --interval 60 --limit 1000

use anyhow::Result;
use clap::Parser;

use fnet_trading::{calculate_features, BybitClient, FeatureConfig, TradingFeatures};

#[derive(Parser, Debug)]
#[command(name = "fetch_data")]
#[command(about = "Fetch cryptocurrency data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval (1, 5, 15, 60, 240, D, W)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "1000")]
    limit: usize,

    /// Show feature statistics
    #[arg(long, default_value = "true")]
    show_stats: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("FNet Trading - Data Fetcher");
    println!("============================");
    println!("Symbol: {}", args.symbol);
    println!("Interval: {}", args.interval);
    println!("Limit: {}", args.limit);
    println!();

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch klines
    println!("Fetching data from Bybit...");
    let klines = client.fetch_klines(&args.symbol, &args.interval, args.limit)?;

    println!("Fetched {} candles", klines.len());

    if klines.is_empty() {
        println!("No data fetched. Please check your network connection.");
        return Ok(());
    }

    // Show sample data
    println!("\nSample data (last 5 candles):");
    println!("{:>12} {:>12} {:>12} {:>12} {:>12} {:>15}",
             "Open", "High", "Low", "Close", "Volume", "Timestamp");

    for kline in klines.iter().rev().take(5) {
        println!("{:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15}",
                 kline.open, kline.high, kline.low, kline.close,
                 kline.volume, kline.timestamp);
    }

    // Calculate features
    println!("\nCalculating trading features...");
    let config = FeatureConfig::default();
    let features = calculate_features(&klines, &config);

    println!("Generated {} feature samples", features.len());

    if args.show_stats {
        print_feature_stats(&features);
    }

    println!("\nData fetch complete!");
    Ok(())
}

fn print_feature_stats(features: &TradingFeatures) {
    println!("\nFeature Statistics:");
    println!("{:-<60}", "");

    let stats = vec![
        ("Log Returns", &features.log_returns),
        ("Volatility", &features.volatility),
        ("Volume Ratio", &features.volume_ratio),
        ("Momentum (5)", &features.momentum_short),
        ("Momentum (10)", &features.momentum_medium),
        ("Momentum (20)", &features.momentum_long),
        ("RSI Normalized", &features.rsi_normalized),
        ("BB Position", &features.bb_position),
    ];

    println!("{:<20} {:>12} {:>12} {:>12} {:>12}",
             "Feature", "Mean", "Std", "Min", "Max");
    println!("{:-<60}", "");

    for (name, values) in stats {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std = (values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("{:<20} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
                 name, mean, std, min, max);
    }
}
