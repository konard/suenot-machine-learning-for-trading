//! # Fetch Order Book Example
//!
//! Demonstrates fetching order book data from Bybit exchange.
//!
//! Run with: `cargo run --example fetch_orderbook`

use anyhow::Result;
use order_flow_imbalance::BybitClient;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Get symbol from args or use default
    let symbol = env::args().nth(1).unwrap_or_else(|| "BTCUSDT".to_string());

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           Order Book Fetcher - Bybit Exchange             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Create client
    let client = BybitClient::new();

    println!("Fetching order book for {}...", symbol);
    println!();

    // Fetch order book with 50 levels
    let orderbook = client.get_orderbook(&symbol, 50).await?;

    // Display order book
    println!("═══════════════════════════════════════════════════════════");
    println!("                     ORDER BOOK: {}                         ", symbol);
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Show asks (reversed for display)
    println!("  ASKS (Sellers)");
    println!("  ─────────────────────────────────────────");
    for (i, level) in orderbook.asks.iter().take(10).enumerate().rev() {
        println!(
            "  {:>2}. ${:>12.2}  │  {:>12.4} {}",
            i + 1,
            level.price,
            level.size,
            symbol.replace("USDT", "")
        );
    }

    println!();
    println!("  ═════════════════════════════════════════");
    if let Some(spread) = orderbook.spread() {
        println!(
            "  SPREAD: ${:.2} ({:.2} bps)",
            spread,
            orderbook.spread_bps().unwrap_or(0.0)
        );
    }
    if let Some(mid) = orderbook.mid_price() {
        println!("  MID PRICE: ${:.2}", mid);
    }
    println!("  ═════════════════════════════════════════");
    println!();

    // Show bids
    println!("  BIDS (Buyers)");
    println!("  ─────────────────────────────────────────");
    for (i, level) in orderbook.bids.iter().take(10).enumerate() {
        println!(
            "  {:>2}. ${:>12.2}  │  {:>12.4} {}",
            i + 1,
            level.price,
            level.size,
            symbol.replace("USDT", "")
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");

    // Show summary statistics
    println!();
    println!("SUMMARY STATISTICS");
    println!("───────────────────────────────────────────────────────────");
    println!("  Bid Levels:        {}", orderbook.bid_levels());
    println!("  Ask Levels:        {}", orderbook.ask_levels());
    println!("  Bid Depth (L5):    {:.4}", orderbook.bid_depth(5));
    println!("  Ask Depth (L5):    {:.4}", orderbook.ask_depth(5));
    println!("  Depth Imbalance:   {:.4}", orderbook.depth_imbalance(5));

    if let Some(imb) = orderbook.weighted_depth_imbalance(10) {
        println!("  Weighted Imbalance:{:.4}", imb);
    }

    // Price impact
    let test_size = 1.0;
    if let Some(impact) = orderbook.price_impact_buy(test_size) {
        println!(
            "  Buy {} Impact:    ${:.2}",
            test_size,
            impact - orderbook.mid_price().unwrap_or(0.0)
        );
    }
    if let Some(impact) = orderbook.price_impact_sell(test_size) {
        println!(
            "  Sell {} Impact:   ${:.2}",
            test_size,
            orderbook.mid_price().unwrap_or(0.0) - impact
        );
    }

    println!();
    println!("Timestamp: {}", orderbook.timestamp);
    println!();

    Ok(())
}
