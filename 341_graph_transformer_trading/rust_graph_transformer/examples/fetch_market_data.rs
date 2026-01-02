//! Example: Fetch market data from Bybit
//!
//! This example demonstrates how to fetch cryptocurrency market data
//! from the Bybit exchange using the API client.

use graph_transformer_trading::prelude::*;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    env_logger::init();

    println!("=== Fetching Market Data from Bybit ===\n");

    // Create API client
    let client = BybitClient::new();

    // Define symbols to fetch
    let symbols = DEFAULT_SYMBOLS;
    println!("Fetching data for {} symbols...\n", symbols.len());

    // Fetch tickers
    println!("1. Fetching tickers...");
    let tickers = client.get_tickers_filtered(symbols).await?;

    println!("\nTop Cryptocurrencies:");
    println!("{:-<70}", "");
    println!("{:<12} {:>12} {:>10} {:>15}", "Symbol", "Price", "Change%", "Volume (24h)");
    println!("{:-<70}", "");

    for ticker in &tickers {
        println!(
            "{:<12} {:>12.2} {:>9.2}% {:>15.0}",
            ticker.symbol,
            ticker.last_price,
            ticker.price_change_24h * 100.0,
            ticker.volume_24h
        );
    }

    // Fetch klines for BTC
    println!("\n2. Fetching BTCUSDT klines (1h, last 100 candles)...");
    let klines = client.get_klines("BTCUSDT", "60", 100).await?;

    println!("\nRecent BTCUSDT Candles:");
    println!("{:-<80}", "");
    println!("{:<20} {:>10} {:>10} {:>10} {:>10} {:>12}",
             "Time", "Open", "High", "Low", "Close", "Volume");
    println!("{:-<80}", "");

    for kline in klines.iter().take(5) {
        let time = chrono::DateTime::from_timestamp_millis(kline.start_time)
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_default();

        println!(
            "{:<20} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>12.4}",
            time, kline.open, kline.high, kline.low, kline.close, kline.volume
        );
    }

    // Fetch order book
    println!("\n3. Fetching BTCUSDT order book (depth 10)...");
    let orderbook = client.get_orderbook("BTCUSDT", 10).await?;

    println!("\nOrder Book:");
    println!("{:-<50}", "");
    println!("Mid Price: ${:.2}", orderbook.mid_price().unwrap_or(0.0));
    println!("Spread: {:.2} bps", orderbook.spread_bps().unwrap_or(0.0));
    println!("Depth Imbalance (L5): {:.4}", orderbook.depth_imbalance(5));

    println!("\nAsks (top 5):");
    for level in orderbook.asks.iter().take(5) {
        println!("  ${:.2} - {:.6} BTC", level.price, level.size);
    }

    println!("\nBids (top 5):");
    for level in orderbook.bids.iter().take(5) {
        println!("  ${:.2} - {:.6} BTC", level.price, level.size);
    }

    // Fetch recent trades
    println!("\n4. Fetching recent trades for BTCUSDT...");
    let trades = client.get_recent_trades("BTCUSDT", 10).await?;

    println!("\nRecent Trades:");
    println!("{:-<60}", "");
    for trade in trades.iter().take(5) {
        let side = if trade.is_buy() { "BUY " } else { "SELL" };
        println!(
            "  {} ${:.2} x {:.6} BTC = ${:.2}",
            side, trade.price, trade.size, trade.value()
        );
    }

    println!("\n=== Data fetching complete! ===");

    Ok(())
}
