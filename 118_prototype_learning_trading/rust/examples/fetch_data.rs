//! Example: Fetching cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the Bybit API client
//! to fetch OHLCV data for analysis.

use prototype_learning::{BybitClient, Interval};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("Prototype Learning - Data Fetching Example");
    println!("==========================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch recent BTC/USDT klines
    println!("Fetching BTC/USDT hourly data...");
    let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(100), None, None)?;

    println!("Fetched {} klines\n", klines.len());

    // Display sample data
    println!("Sample data (last 5 candles):");
    println!(
        "{:<25} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume"
    );
    println!("{}", "-".repeat(95));

    for kline in klines.iter().rev().take(5) {
        println!(
            "{:<25} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            kline.datetime().format("%Y-%m-%d %H:%M:%S"),
            kline.open,
            kline.high,
            kline.low,
            kline.close,
            kline.volume
        );
    }

    // Calculate some basic statistics
    println!("\nBasic Statistics:");
    println!("-----------------");

    let returns: Vec<f64> = klines.iter().map(|k| k.return_pct()).collect();
    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let volatility: f64 = {
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        variance.sqrt()
    };

    println!("Mean candle return: {:.4}%", mean_return * 100.0);
    println!("Volatility (per candle): {:.4}%", volatility * 100.0);

    let max_price = klines
        .iter()
        .map(|k| k.high)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_price = klines
        .iter()
        .map(|k| k.low)
        .fold(f64::INFINITY, f64::min);

    println!("Price range: ${:.2} - ${:.2}", min_price, max_price);

    // Get current ticker
    println!("\nCurrent Ticker:");
    println!("---------------");
    let ticker = client.get_ticker("BTCUSDT")?;
    println!("Symbol: {}", ticker.symbol);
    println!("Last Price: ${}", ticker.last_price);
    println!("24h Change: {}%", ticker.price_24h_pcnt);
    println!("24h Volume: {}", ticker.volume_24h);

    println!("\nDone!");
    Ok(())
}
