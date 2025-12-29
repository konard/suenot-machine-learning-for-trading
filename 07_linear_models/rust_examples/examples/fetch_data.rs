//! Example: Fetching cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the Bybit API client
//! to fetch OHLCV data for various cryptocurrencies.

use chrono::{Duration, Utc};
use linear_models_crypto::api::bybit::{BybitClient, Interval};

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    println!("===========================================");
    println!("  Bybit Cryptocurrency Data Fetcher");
    println!("===========================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Define symbols to fetch
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    // Fetch current ticker info
    println!("Current Market Prices:");
    println!("-----------------------");

    for symbol in &symbols {
        match client.get_ticker(symbol) {
            Ok(ticker) => {
                println!(
                    "{}: ${} (24h: {}%)",
                    ticker.symbol, ticker.last_price, ticker.price_24h_pcnt
                );
            }
            Err(e) => {
                println!("{}: Error fetching ticker - {}", symbol, e);
            }
        }
    }

    println!("\n");

    // Fetch historical OHLCV data for BTC
    println!("Fetching historical data for BTCUSDT...");
    println!("----------------------------------------");

    let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(100), None, None)?;

    println!("Fetched {} hourly candles\n", klines.len());

    // Display last 10 candles
    println!("Last 10 candles:");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume"
    );
    println!("{}", "-".repeat(90));

    for kline in klines.iter().rev().take(10) {
        println!(
            "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            kline.datetime().format("%Y-%m-%d %H:%M"),
            kline.open,
            kline.high,
            kline.low,
            kline.close,
            kline.volume
        );
    }

    // Calculate basic statistics
    println!("\nBasic Statistics:");
    println!("-----------------");

    let returns: Vec<f64> = klines.windows(2).map(|w| w[1].return_pct()).collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance =
        returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();

    let max_return = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_return = returns.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("Mean hourly return:  {:.4}%", mean_return * 100.0);
    println!("Std deviation:       {:.4}%", std_dev * 100.0);
    println!("Max return:          {:.4}%", max_return * 100.0);
    println!("Min return:          {:.4}%", min_return * 100.0);

    // Fetch longer history
    println!("\nFetching extended history (last 7 days)...");

    let end_time = Utc::now().timestamp_millis();
    let start_time = (Utc::now() - Duration::days(7)).timestamp_millis();

    let history = client.get_klines_history("BTCUSDT", Interval::Hour1, start_time, end_time)?;

    println!("Fetched {} hourly candles over 7 days", history.len());

    // Calculate 7-day performance
    if history.len() >= 2 {
        let first_price = history.first().map(|k| k.close).unwrap_or(0.0);
        let last_price = history.last().map(|k| k.close).unwrap_or(0.0);
        let performance = (last_price / first_price - 1.0) * 100.0;

        println!("7-day performance: {:.2}%", performance);
        println!("Start price: ${:.2}", first_price);
        println!("End price:   ${:.2}", last_price);
    }

    // Demonstrate different intervals
    println!("\nAvailable intervals demonstration:");
    println!("-----------------------------------");

    let intervals = vec![
        (Interval::Min15, "15 minutes"),
        (Interval::Hour4, "4 hours"),
        (Interval::Day1, "1 day"),
    ];

    for (interval, name) in intervals {
        match client.get_klines("BTCUSDT", interval, Some(5), None, None) {
            Ok(data) => {
                println!(
                    "{} interval: {} candles fetched, latest close: ${:.2}",
                    name,
                    data.len(),
                    data.last().map(|k| k.close).unwrap_or(0.0)
                );
            }
            Err(e) => {
                println!("{} interval: Error - {}", name, e);
            }
        }
    }

    println!("\nDone!");

    Ok(())
}
