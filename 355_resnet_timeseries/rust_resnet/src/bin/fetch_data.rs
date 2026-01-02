//! Fetch cryptocurrency data from Bybit
//!
//! This binary fetches historical kline data from Bybit and saves it to CSV.
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1 --limit 10000

use anyhow::Result;
use chrono::{DateTime, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use rust_resnet::api::BybitClient;
use std::fs::File;
use std::io::Write;

/// Configuration for data fetching
struct Config {
    symbol: String,
    interval: String,
    limit: usize,
    output_path: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            interval: "1".to_string(), // 1 minute
            limit: 10000,
            output_path: "data".to_string(),
        }
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--symbol" | "-s" => {
                if i + 1 < args.len() {
                    config.symbol = args[i + 1].clone();
                    i += 1;
                }
            }
            "--interval" | "-i" => {
                if i + 1 < args.len() {
                    config.interval = args[i + 1].clone();
                    i += 1;
                }
            }
            "--limit" | "-l" => {
                if i + 1 < args.len() {
                    config.limit = args[i + 1].parse().unwrap_or(10000);
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    config.output_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("Fetch cryptocurrency data from Bybit\n");
                println!("Usage: fetch_data [OPTIONS]\n");
                println!("Options:");
                println!("  -s, --symbol <SYMBOL>      Trading pair (default: BTCUSDT)");
                println!("  -i, --interval <INTERVAL>  Kline interval: 1,3,5,15,30,60,120,240,360,720,D,W,M (default: 1)");
                println!("  -l, --limit <LIMIT>        Number of candles to fetch (default: 10000)");
                println!("  -o, --output <PATH>        Output directory (default: data)");
                println!("  -h, --help                 Show this help message");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    config
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let config = parse_args();

    println!("=== Bybit Data Fetcher ===\n");
    println!("Symbol:   {}", config.symbol);
    println!("Interval: {} minute(s)", config.interval);
    println!("Limit:    {} candles", config.limit);
    println!();

    // Create client
    let client = BybitClient::new();

    // Create progress bar
    let pb = ProgressBar::new(config.limit as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    println!("Fetching data from Bybit...");

    // Fetch data
    let candles = client
        .fetch_klines_paginated(&config.symbol, &config.interval, config.limit)
        .await?;

    pb.finish_with_message("Done!");

    if candles.is_empty() {
        println!("\nNo data received!");
        return Ok(());
    }

    // Print summary
    let first_time = DateTime::from_timestamp_millis(candles.first().unwrap().timestamp)
        .unwrap_or_else(|| DateTime::<Utc>::MIN_UTC);
    let last_time = DateTime::from_timestamp_millis(candles.last().unwrap().timestamp)
        .unwrap_or_else(|| DateTime::<Utc>::MIN_UTC);

    println!("\n=== Data Summary ===");
    println!("Candles fetched: {}", candles.len());
    println!("First candle:    {}", first_time.format("%Y-%m-%d %H:%M:%S UTC"));
    println!("Last candle:     {}", last_time.format("%Y-%m-%d %H:%M:%S UTC"));
    println!(
        "Price range:     ${:.2} - ${:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
    );

    // Create output directory
    std::fs::create_dir_all(&config.output_path)?;

    // Save to CSV
    let filename = format!(
        "{}/{}_{}_{}candles.csv",
        config.output_path,
        config.symbol,
        config.interval,
        candles.len()
    );

    let mut file = File::create(&filename)?;

    // Write header
    writeln!(file, "timestamp,datetime,open,high,low,close,volume,turnover")?;

    // Write data
    for candle in &candles {
        let dt = DateTime::from_timestamp_millis(candle.timestamp)
            .unwrap_or_else(|| DateTime::<Utc>::MIN_UTC);
        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            candle.timestamp,
            dt.format("%Y-%m-%d %H:%M:%S"),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.turnover
        )?;
    }

    println!("\nData saved to: {}", filename);

    // Calculate some basic statistics
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let returns: Vec<f64> = closes
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
        / returns.len() as f64;
    let std_return = variance.sqrt();

    println!("\n=== Basic Statistics ===");
    println!("Mean return:     {:.6}%", mean_return * 100.0);
    println!("Std deviation:   {:.6}%", std_return * 100.0);
    println!("Annualized vol:  {:.2}%", std_return * (525600.0_f64).sqrt() * 100.0);

    let positive = returns.iter().filter(|&&r| r > 0.0).count();
    println!(
        "Up candles:      {:.1}%",
        100.0 * positive as f64 / returns.len() as f64
    );

    println!("\nDone!");

    Ok(())
}
