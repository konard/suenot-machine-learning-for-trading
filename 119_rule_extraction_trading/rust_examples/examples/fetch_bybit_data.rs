//! Example: Fetching cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the BybitClient to fetch
//! OHLCV data and current ticker information.
//!
//! Run with: cargo run --example fetch_bybit_data

use anyhow::Result;
use rule_extraction_trading::data::{BybitClient, Interval};

fn main() -> Result<()> {
    println!("=== Bybit Data Fetcher ===\n");

    let client = BybitClient::new();

    // Fetch Bitcoin hourly data
    println!("Fetching BTCUSDT hourly data...");
    let btc_klines = client.get_klines("BTCUSDT", Interval::Hour1.as_str(), Some(24))?;

    println!("\nBTCUSDT - Last 24 Hours:");
    println!("{:-<90}", "");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Time", "Open", "High", "Low", "Close", "Volume"
    );
    println!("{:-<90}", "");

    for kline in btc_klines.iter() {
        println!(
            "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.4}",
            kline.timestamp.format("%Y-%m-%d %H:%M"),
            kline.open,
            kline.high,
            kline.low,
            kline.close,
            kline.volume
        );
    }

    // Calculate some statistics
    if !btc_klines.is_empty() {
        let returns: Vec<f64> = btc_klines.iter().map(|k| k.returns()).collect();
        let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let volatility = {
            let variance = returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
                / returns.len() as f64;
            variance.sqrt()
        };

        println!("\n=== Statistics ===");
        println!("Average hourly return: {:.4}%", avg_return * 100.0);
        println!("Hourly volatility: {:.4}%", volatility * 100.0);
        println!(
            "Annualized volatility: {:.2}%",
            volatility * (24.0 * 365.0_f64).sqrt() * 100.0
        );
    }

    // Fetch current ticker
    println!("\n=== Current Ticker ===");
    let ticker = client.get_ticker("BTCUSDT")?;
    println!("Symbol: {}", ticker.symbol);
    println!("Last Price: ${:.2}", ticker.last_price);
    println!("24h High: ${:.2}", ticker.high_price_24h);
    println!("24h Low: ${:.2}", ticker.low_price_24h);
    println!("24h Change: {:.2}%", ticker.price_change_24h * 100.0);
    println!("24h Volume: {:.2} BTC", ticker.volume_24h);

    // Fetch Ethereum data
    println!("\n=== Ethereum Data ===");
    let eth_ticker = client.get_ticker("ETHUSDT")?;
    println!("ETH Price: ${:.2}", eth_ticker.last_price);
    println!("24h Change: {:.2}%", eth_ticker.price_change_24h * 100.0);

    // Fetch multiple symbols
    println!("\n=== Multiple Symbols ===");
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    for symbol in &symbols {
        match client.get_ticker(symbol) {
            Ok(t) => {
                println!(
                    "{}: ${:.2} ({:+.2}%)",
                    symbol,
                    t.last_price,
                    t.price_change_24h * 100.0
                );
            }
            Err(e) => {
                println!("{}: Error - {}", symbol, e);
            }
        }
    }

    Ok(())
}
