//! Example: Fetching market data from Bybit
//!
//! This example demonstrates how to fetch cryptocurrency market data
//! from the Bybit API.

use lime_trading::api::bybit::BybitClient;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    println!("Bybit Data Fetcher");
    println!("==================\n");

    // Create client
    let client = BybitClient::new();

    // Fetch BTCUSDT data
    println!("Fetching BTCUSDT hourly data...");
    let btc_candles = client.fetch_klines("BTCUSDT", "60", 100)?;

    println!("Fetched {} candles\n", btc_candles.len());
    println!("Latest candles:");
    println!("{:^25} {:^12} {:^12} {:^12} {:^12} {:^15}",
             "Timestamp", "Open", "High", "Low", "Close", "Volume");
    println!("{}", "-".repeat(90));

    for candle in btc_candles.iter().rev().take(10) {
        println!("{:^25} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
                 candle.timestamp.format("%Y-%m-%d %H:%M"),
                 candle.open,
                 candle.high,
                 candle.low,
                 candle.close,
                 candle.volume);
    }

    println!("\n");

    // Fetch ETHUSDT data
    println!("Fetching ETHUSDT daily data...");
    let eth_candles = client.fetch_klines("ETHUSDT", "D", 30)?;

    println!("Fetched {} candles\n", eth_candles.len());

    // Calculate some statistics
    let closes: Vec<f64> = eth_candles.iter().map(|c| c.close).collect();
    let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;
    let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("ETH Statistics (last 30 days):");
    println!("  Average close: ${:.2}", avg_price);
    println!("  Max close: ${:.2}", max_price);
    println!("  Min close: ${:.2}", min_price);
    println!("  Range: ${:.2}", max_price - min_price);

    // Calculate daily returns
    let mut returns: Vec<f64> = Vec::new();
    for i in 1..closes.len() {
        returns.push((closes[i] / closes[i - 1]) - 1.0);
    }

    let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>() / returns.len() as f64;
    let volatility = variance.sqrt() * (365.0_f64).sqrt(); // Annualized

    println!("  Avg daily return: {:.4}%", avg_return * 100.0);
    println!("  Annualized volatility: {:.2}%", volatility * 100.0);

    println!("\nDone!");
    Ok(())
}
