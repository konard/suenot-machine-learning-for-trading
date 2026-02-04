//! Example: Fetch market data from Bybit

use ensemble_uncertainty_trading::prelude::*;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=".repeat(60));
    println!("Fetching Market Data from Bybit");
    println!("=".repeat(60));

    // Create client
    let client = BybitClient::new();

    // Fetch klines for BTC
    println!("\n--- Fetching BTCUSDT Klines ---");
    let klines = client.get_klines("BTCUSDT", "60", 100).await?;

    println!("Fetched {} candles", klines.len());
    if let Some(first) = klines.first() {
        println!("First candle: {} O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
            first.datetime(),
            first.open, first.high, first.low, first.close, first.volume);
    }
    if let Some(last) = klines.last() {
        println!("Last candle:  {} O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
            last.datetime(),
            last.open, last.high, last.low, last.close, last.volume);
    }

    // Fetch ticker
    println!("\n--- Fetching BTCUSDT Ticker ---");
    let ticker = client.get_ticker("BTCUSDT").await?;

    println!("Last Price: ${:.2}", ticker.last_price);
    println!("Bid: ${:.2}, Ask: ${:.2}", ticker.bid_price, ticker.ask_price);
    println!("24h Volume: {:.2}", ticker.volume_24h);
    println!("24h Change: {:.2}%", ticker.price_change_pct_24h);

    // Fetch order book
    println!("\n--- Fetching BTCUSDT Order Book ---");
    let order_book = client.get_order_book("BTCUSDT", 10).await?;

    if let (Some(bid), Some(ask)) = (order_book.best_bid(), order_book.best_ask()) {
        println!("Best Bid: ${:.2}", bid);
        println!("Best Ask: ${:.2}", ask);
    }
    if let Some(mid) = order_book.mid_price() {
        println!("Mid Price: ${:.2}", mid);
    }
    if let Some(spread) = order_book.spread_bps() {
        println!("Spread: {:.2} bps", spread);
    }
    println!("Volume Imbalance: {:.4}", order_book.volume_imbalance());

    // Fetch multiple symbols
    println!("\n--- Fetching Multiple Symbols ---");
    let symbols = &["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let multi_data = client.get_multiple_klines(symbols, "60", 10).await?;

    for (symbol, klines) in multi_data {
        if let Some(last) = klines.last() {
            println!("{}: ${:.2} ({} candles)", symbol, last.close, klines.len());
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Data fetch complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}
