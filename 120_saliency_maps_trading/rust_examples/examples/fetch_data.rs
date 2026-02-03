//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the Bybit API client
//! to fetch historical OHLCV data.

use saliency_maps_trading::api::bybit::{BybitClient, Interval};
use chrono::Utc;

fn main() {
    env_logger::init();

    println!("Saliency Maps Trading - Data Fetcher");
    println!("=====================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch recent klines for BTC/USDT
    let symbol = "BTCUSDT";
    let interval = Interval::Hour1;

    println!("Fetching {} {} klines...", symbol, "1h");

    match client.get_klines(symbol, interval, Some(100), None, None) {
        Ok(klines) => {
            println!("Fetched {} klines\n", klines.len());

            // Display last 5 candles
            println!("Last 5 candles:");
            println!("{:-<80}", "");
            println!(
                "{:<20} {:>12} {:>12} {:>12} {:>12}",
                "Timestamp", "Open", "High", "Low", "Close"
            );
            println!("{:-<80}", "");

            for kline in klines.iter().rev().take(5) {
                println!(
                    "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
                    kline.datetime().format("%Y-%m-%d %H:%M"),
                    kline.open,
                    kline.high,
                    kline.low,
                    kline.close
                );
            }

            println!("{:-<80}", "");

            // Compute some statistics
            let returns: Vec<f64> = klines.iter().map(|k| k.return_pct()).collect();
            let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let volatility: f64 = {
                let variance: f64 = returns
                    .iter()
                    .map(|r| (r - avg_return).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                variance.sqrt()
            };

            println!("\nStatistics:");
            println!("  Average hourly return: {:.4}%", avg_return * 100.0);
            println!("  Hourly volatility: {:.4}%", volatility * 100.0);
            println!(
                "  Price range: ${:.2} - ${:.2}",
                klines.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
                klines.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max)
            );
        }
        Err(e) => {
            eprintln!("Error fetching klines: {}", e);
        }
    }

    // Also try fetching ETH data
    println!("\n\nFetching ETHUSDT daily klines...");

    match client.get_klines("ETHUSDT", Interval::Day1, Some(30), None, None) {
        Ok(klines) => {
            println!("Fetched {} daily klines for ETH\n", klines.len());

            let first_price = klines.first().map(|k| k.close).unwrap_or(0.0);
            let last_price = klines.last().map(|k| k.close).unwrap_or(0.0);
            let period_return = (last_price / first_price - 1.0) * 100.0;

            println!("30-day performance:");
            println!("  Start price: ${:.2}", first_price);
            println!("  End price: ${:.2}", last_price);
            println!("  Period return: {:.2}%", period_return);
        }
        Err(e) => {
            eprintln!("Error fetching ETH klines: {}", e);
        }
    }
}
