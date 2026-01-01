//! Bybit Data Feed Example
//!
//! Demonstrates fetching market data from Bybit exchange.

use neuromorphic_trading::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Bybit Data Feed Example ===\n");

    // Create Bybit client (testnet)
    let config = BybitConfig {
        testnet: true,
        ..Default::default()
    };
    let client = BybitClient::new(config);

    let symbol = "BTCUSDT";

    // Fetch ticker
    println!("Fetching ticker for {}...", symbol);
    match client.get_ticker(symbol).await {
        Ok(ticker) => {
            println!("Ticker:");
            println!("  Last Price: ${:.2}", ticker.last_price);
            println!("  24h High: ${:.2}", ticker.high_24h);
            println!("  24h Low: ${:.2}", ticker.low_24h);
            println!("  24h Volume: {:.2}", ticker.volume_24h);
            println!("  24h Change: {:.2}%", ticker.price_change_pct);
        }
        Err(e) => {
            println!("Failed to fetch ticker: {}", e);
        }
    }

    println!();

    // Fetch orderbook
    println!("Fetching orderbook for {}...", symbol);
    match client.get_orderbook(symbol, 5).await {
        Ok(orderbook) => {
            println!("Orderbook:");
            println!("  Best Bid: ${:.2}", orderbook.best_bid().unwrap_or(0.0));
            println!("  Best Ask: ${:.2}", orderbook.best_ask().unwrap_or(0.0));
            println!("  Mid Price: ${:.2}", orderbook.mid_price().unwrap_or(0.0));
            println!("  Spread: ${:.2}", orderbook.spread().unwrap_or(0.0));
            println!("  Spread (bps): {:.2}", orderbook.spread_bps().unwrap_or(0.0));

            println!("\n  Bids:");
            for (i, bid) in orderbook.bids.iter().take(5).enumerate() {
                println!("    {}: ${:.2} x {:.4}", i + 1, bid.price, bid.quantity);
            }

            println!("\n  Asks:");
            for (i, ask) in orderbook.asks.iter().take(5).enumerate() {
                println!("    {}: ${:.2} x {:.4}", i + 1, ask.price, ask.quantity);
            }
        }
        Err(e) => {
            println!("Failed to fetch orderbook: {}", e);
        }
    }

    println!();

    // Fetch recent trades
    println!("Fetching recent trades for {}...", symbol);
    match client.get_trades(symbol, 5).await {
        Ok(trades) => {
            println!("Recent Trades:");
            for trade in trades.iter().take(5) {
                println!(
                    "  {:?} ${:.2} x {:.4}",
                    trade.side, trade.price, trade.quantity
                );
            }
        }
        Err(e) => {
            println!("Failed to fetch trades: {}", e);
        }
    }

    println!("\n=== Example Complete ===");

    Ok(())
}
