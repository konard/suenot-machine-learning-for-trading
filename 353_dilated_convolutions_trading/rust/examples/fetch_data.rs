//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to:
//! 1. Connect to Bybit API
//! 2. Fetch kline (candlestick) data
//! 3. Display basic statistics

use dilated_conv_trading::api::{Category, Interval};
use dilated_conv_trading::BybitClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Bybit Data Fetcher ===\n");

    // Create API client (using linear perpetuals by default)
    let client = BybitClient::new().with_category(Category::Linear);

    // Fetch klines for BTCUSDT
    let symbol = "BTCUSDT";
    let interval = Interval::Hour1;
    let limit = 200;

    println!("Fetching {} {} klines for {}...", limit, "1h", symbol);

    let klines = client
        .get_klines_with_interval(symbol, interval, limit)
        .await?;

    println!("Fetched {} klines\n", klines.len());

    // Display first and last klines
    if let Some(first) = klines.first() {
        println!("First kline:");
        println!("  Time: {}", first.timestamp);
        println!("  Open: {:.2}", first.open);
        println!("  High: {:.2}", first.high);
        println!("  Low: {:.2}", first.low);
        println!("  Close: {:.2}", first.close);
        println!("  Volume: {:.2}", first.volume);
    }

    if let Some(last) = klines.last() {
        println!("\nLast kline:");
        println!("  Time: {}", last.timestamp);
        println!("  Open: {:.2}", last.open);
        println!("  High: {:.2}", last.high);
        println!("  Low: {:.2}", last.low);
        println!("  Close: {:.2}", last.close);
        println!("  Volume: {:.2}", last.volume);
    }

    // Calculate basic statistics
    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let returns: Vec<f64> = closes
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
    let volatility = variance.sqrt();

    println!("\n=== Statistics ===");
    println!("Price range: {:.2} - {:.2}",
        closes.iter().cloned().fold(f64::INFINITY, f64::min),
        closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!("Mean hourly return: {:.4}%", mean_return * 100.0);
    println!("Hourly volatility: {:.4}%", volatility * 100.0);
    println!("Annualized volatility: {:.2}%", volatility * (24.0 * 365.0_f64).sqrt() * 100.0);

    // Fetch ticker for additional info
    println!("\n=== Current Ticker ===");
    let ticker = client.get_ticker(symbol).await?;
    println!("Last price: {:.2}", ticker.last_price);
    println!("24h change: {:.2}%", ticker.price_change_percent_24h);
    println!("24h volume: {:.2}", ticker.volume_24h);
    println!("Bid: {:.2} x {:.4}", ticker.bid_price, ticker.bid_size);
    println!("Ask: {:.2} x {:.4}", ticker.ask_price, ticker.ask_size);

    // Fetch order book
    println!("\n=== Order Book (top 5) ===");
    let orderbook = client.get_orderbook(symbol, 5).await?;

    println!("Asks:");
    for ask in orderbook.asks.iter().take(5).rev() {
        println!("  {:.2} x {:.4}", ask.price, ask.size);
    }
    println!("--- Spread: {:.2} ---", orderbook.spread().unwrap_or(0.0));
    println!("Bids:");
    for bid in orderbook.bids.iter().take(5) {
        println!("  {:.2} x {:.4}", bid.price, bid.size);
    }

    println!("\nImbalance: {:.4}", orderbook.imbalance());

    Ok(())
}
