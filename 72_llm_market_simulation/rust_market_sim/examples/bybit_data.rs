//! Bybit Data Integration Example
//!
//! Fetches real cryptocurrency data from Bybit exchange and uses it
//! to initialize a market simulation.

use llm_market_sim::data::{BybitClient, CRYPTO_UNIVERSE};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Bybit Data Integration Example");
    println!("==============================");

    let client = BybitClient::new();

    // Fetch ticker for BTCUSDT
    println!("\nFetching BTCUSDT ticker...");
    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            println!("  Symbol: {}", ticker.symbol);
            println!("  Last Price: ${}", ticker.last_price);
            println!("  24h High: ${}", ticker.high_price_24h);
            println!("  24h Low: ${}", ticker.low_price_24h);
            println!("  24h Volume: {}", ticker.volume_24h);
            println!("  24h Change: {}%", ticker.price_24h_pcnt);
        }
        Err(e) => {
            println!("  Error: {}", e);
            println!("  (This is expected if running without network access)");
        }
    }

    // Fetch order book
    println!("\nFetching order book...");
    match client.get_orderbook("BTCUSDT", 5).await {
        Ok(orderbook) => {
            println!("  Top 5 Bids:");
            for (price, qty) in orderbook.bids.iter().take(5) {
                println!("    ${:.2} x {:.4}", price, qty);
            }
            println!("  Top 5 Asks:");
            for (price, qty) in orderbook.asks.iter().take(5) {
                println!("    ${:.2} x {:.4}", price, qty);
            }
            if let (Some(bid), Some(ask)) = (orderbook.bids.first(), orderbook.asks.first()) {
                let spread = ask.0 - bid.0;
                println!("  Spread: ${:.2} ({:.4}%)", spread, spread / bid.0 * 100.0);
            }
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    // Fetch klines
    println!("\nFetching recent klines...");
    match client.get_klines("BTCUSDT", "60", None, None, Some(5)).await {
        Ok(series) => {
            println!("  Symbol: {}", series.symbol);
            println!("  Interval: {}", series.interval);
            println!("  Candles: {}", series.len());
            for candle in series.candles.iter().take(5) {
                println!(
                    "    {}: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
                    candle.timestamp.format("%Y-%m-%d %H:%M"),
                    candle.open, candle.high, candle.low, candle.close, candle.volume
                );
            }
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    // List available pairs
    println!("\nAvailable Crypto Universe:");
    for (i, symbol) in CRYPTO_UNIVERSE.iter().enumerate() {
        print!("  {}", symbol);
        if (i + 1) % 4 == 0 {
            println!();
        }
    }
    println!();

    println!("\nDone!");
    Ok(())
}
