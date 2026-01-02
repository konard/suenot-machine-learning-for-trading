//! Fetch market data from Bybit
//!
//! This example demonstrates how to fetch cryptocurrency data
//! for use with Graph Attention Networks.
//!
//! Run with: cargo run --example fetch_data

use anyhow::Result;
use gat_trading::api::BybitClient;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Bybit Data Fetcher ===\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Symbols to fetch
    let symbols = vec![
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOGEUSDT",
        "NEARUSDT", "MATICUSDT", "ARBUSDT", "LINKUSDT", "UNIUSDT",
    ];

    println!("Fetching 1-hour candles for {} symbols...\n", symbols.len());

    for symbol in &symbols {
        match client.get_klines(symbol, "1h", 100).await {
            Ok(candles) => {
                let latest = candles.last().unwrap();
                println!(
                    "{}: {} candles | Latest: O:{:.2} H:{:.2} L:{:.2} C:{:.2} V:{:.0}",
                    symbol,
                    candles.len(),
                    latest.open,
                    latest.high,
                    latest.low,
                    latest.close,
                    latest.volume
                );
            }
            Err(e) => {
                println!("{}: Error - {}", symbol, e);
            }
        }
    }

    println!("\n=== Order Book Example ===\n");

    // Fetch order book
    match client.get_orderbook("BTCUSDT", 10).await {
        Ok(ob) => {
            println!("BTCUSDT Order Book:");
            println!("  Mid price: {:.2}", ob.mid_price().unwrap_or(0.0));
            println!("  Spread: {:.2}", ob.spread().unwrap_or(0.0));
            println!("  Spread %: {:.4}%", ob.spread_pct().unwrap_or(0.0) * 100.0);
            println!("  Imbalance (5 levels): {:.3}", ob.imbalance(5));
            println!("\n  Top 3 Bids:");
            for bid in ob.bids.iter().take(3) {
                println!("    {:.2} @ {:.4}", bid.price, bid.quantity);
            }
            println!("  Top 3 Asks:");
            for ask in ob.asks.iter().take(3) {
                println!("    {:.2} @ {:.4}", ask.price, ask.quantity);
            }
        }
        Err(e) => println!("Error fetching order book: {}", e),
    }

    println!("\n=== Recent Trades Example ===\n");

    // Fetch recent trades
    match client.get_recent_trades("BTCUSDT", 10).await {
        Ok(trades) => {
            println!("BTCUSDT Recent Trades:");
            for trade in trades.iter().take(5) {
                println!(
                    "  {:?} {:.4} BTC @ {:.2}",
                    trade.side, trade.quantity, trade.price
                );
            }
        }
        Err(e) => println!("Error fetching trades: {}", e),
    }

    println!("\n=== Done ===");

    Ok(())
}
