//! Fetch market data from Bybit
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --limit 1000

use anyhow::Result;
use chrono::{Duration, Utc};
use rust_marl_trading::api::{BybitClient, TimeFrame};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse arguments (simple manual parsing)
    let args: Vec<String> = std::env::args().collect();

    let symbol = args
        .iter()
        .position(|a| a == "--symbol")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("BTCUSDT");

    let interval_str = args
        .iter()
        .position(|a| a == "--interval")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("1h");

    let limit: u32 = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let interval = match interval_str {
        "1m" => TimeFrame::Min1,
        "5m" => TimeFrame::Min5,
        "15m" => TimeFrame::Min15,
        "1h" | "60" => TimeFrame::Hour1,
        "4h" => TimeFrame::Hour4,
        "1d" | "D" => TimeFrame::Day1,
        _ => TimeFrame::Hour1,
    };

    println!("=== Bybit Data Fetcher ===");
    println!("Symbol: {}", symbol);
    println!("Interval: {:?}", interval);
    println!("Limit: {}", limit);
    println!();

    // Create client
    let client = BybitClient::new();

    // Fetch klines
    println!("Fetching kline data...");
    let data = client.get_klines(symbol, interval, Some(limit), None, None).await?;

    println!("Received {} candles", data.candles.len());
    println!();

    // Display last 10 candles
    println!("Last 10 candles:");
    println!("{:-<80}", "");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume"
    );
    println!("{:-<80}", "");

    for candle in data.candles.iter().rev().take(10).rev() {
        println!(
            "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
            candle.timestamp.format("%Y-%m-%d %H:%M"),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        );
    }
    println!("{:-<80}", "");
    println!();

    // Calculate some basic statistics
    if !data.candles.is_empty() {
        let returns = data.returns();
        let closes = data.closes();

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len().max(1) as f64;
        let volatility = variance.sqrt();

        let first_price = closes.first().copied().unwrap_or(0.0);
        let last_price = closes.last().copied().unwrap_or(0.0);
        let total_return = (last_price - first_price) / first_price;

        println!("=== Statistics ===");
        println!("First price: ${:.2}", first_price);
        println!("Last price: ${:.2}", last_price);
        println!("Total return: {:.2}%", total_return * 100.0);
        println!("Mean return: {:.4}%", mean_return * 100.0);
        println!("Volatility: {:.4}%", volatility * 100.0);
        println!();

        // Calculate SMAs
        let sma_20 = data.sma(20);
        let sma_50 = data.sma(50);

        if let (Some(&sma20), Some(&sma50)) = (sma_20.last(), sma_50.last()) {
            println!("=== Technical Indicators ===");
            println!("SMA(20): ${:.2}", sma20);
            println!("SMA(50): ${:.2}", sma50);

            if sma20 > sma50 {
                println!("Signal: BULLISH (SMA20 > SMA50)");
            } else {
                println!("Signal: BEARISH (SMA20 < SMA50)");
            }
        }
    }

    // Fetch order book
    println!();
    println!("=== Order Book ===");
    let orderbook = client.get_orderbook(symbol, Some(5)).await?;

    println!("Bids:");
    for (price, qty) in orderbook.bids.iter().take(5) {
        println!("  ${:.2} x {:.4}", price, qty);
    }

    println!("Asks:");
    for (price, qty) in orderbook.asks.iter().take(5) {
        println!("  ${:.2} x {:.4}", price, qty);
    }

    if let Some(mid) = orderbook.mid_price() {
        println!();
        println!("Mid price: ${:.2}", mid);
        println!("Spread: ${:.2} ({:.4}%)", orderbook.spread().unwrap_or(0.0), orderbook.spread_pct().unwrap_or(0.0));
        println!("Imbalance: {:.4}", orderbook.imbalance());
    }

    // Fetch ticker
    println!();
    println!("=== Ticker ===");
    let ticker = client.get_ticker(symbol).await?;
    println!("Last price: ${:.2}", ticker.last_price);
    println!("24h high: ${:.2}", ticker.high_24h);
    println!("24h low: ${:.2}", ticker.low_24h);
    println!("24h change: {:.2}%", ticker.price_change_pct_24h);
    println!("24h volume: {:.2}", ticker.volume_24h);

    Ok(())
}
