//! Example: Fetching cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the Bybit API client
//! to fetch OHLCV (candlestick) data for cryptocurrency trading pairs.
//!
//! Run with: cargo run --example fetch_data

use conformal_prediction_trading::api::bybit::{BybitClient, Interval};

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    println!("=== Fetching Cryptocurrency Data from Bybit ===\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch recent klines for BTC/USDT
    println!("Fetching 100 hourly candles for BTCUSDT...");
    let btc_klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(100), None, None)?;

    println!("Received {} candles for BTCUSDT", btc_klines.len());
    if let Some(first) = btc_klines.first() {
        println!(
            "First candle: {} - Open: {:.2}, Close: {:.2}",
            first.datetime(),
            first.open,
            first.close
        );
    }
    if let Some(last) = btc_klines.last() {
        println!(
            "Last candle:  {} - Open: {:.2}, Close: {:.2}",
            last.datetime(),
            last.open,
            last.close
        );
    }

    // Calculate some statistics
    let closes: Vec<f64> = btc_klines.iter().map(|k| k.close).collect();
    let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

    let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let volatility = {
        let variance = returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
            / returns.len() as f64;
        variance.sqrt()
    };

    println!("\n--- Statistics ---");
    println!("Average hourly return: {:.4}%", avg_return * 100.0);
    println!("Hourly volatility: {:.4}%", volatility * 100.0);
    println!(
        "Annualized volatility: {:.2}%",
        volatility * (252.0 * 24.0_f64).sqrt() * 100.0
    );

    // Fetch data for ETH/USDT
    println!("\n\nFetching 50 daily candles for ETHUSDT...");
    let eth_klines = client.get_klines("ETHUSDT", Interval::Day1, Some(50), None, None)?;

    println!("Received {} candles for ETHUSDT", eth_klines.len());
    if let Some(last) = eth_klines.last() {
        println!(
            "Latest: Open: {:.2}, High: {:.2}, Low: {:.2}, Close: {:.2}",
            last.open, last.high, last.low, last.close
        );
        println!("Volume: {:.2}, Return: {:.2}%", last.volume, last.return_pct() * 100.0);
    }

    // Get current ticker
    println!("\n\nFetching current ticker for SOLUSDT...");
    let ticker = client.get_ticker("SOLUSDT")?;
    println!("SOL/USDT:");
    println!("  Last Price: ${}", ticker.last_price);
    println!("  24h High: ${}", ticker.high_price_24h);
    println!("  24h Low: ${}", ticker.low_price_24h);
    println!("  24h Change: {}%", ticker.price_24h_pcnt);

    println!("\n=== Done! ===");

    Ok(())
}
