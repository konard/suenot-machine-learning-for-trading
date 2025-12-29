//! Example: Fetching trades from Bybit
//!
//! This example demonstrates how to use the Bybit API client
//! to fetch recent trades and market data.
//!
//! Run with: cargo run --example fetch_trades

use crypto_embeddings::BybitClient;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("=== Bybit API Example ===\n");

    let client = BybitClient::new();

    // 1. Get ticker information
    println!("1. Fetching BTCUSDT ticker...");
    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            println!("   Symbol: {}", ticker.symbol);
            println!("   Last Price: ${:.2}", ticker.last_price);
            println!("   24h High: ${:.2}", ticker.high_price_24h);
            println!("   24h Low: ${:.2}", ticker.low_price_24h);
            println!("   24h Change: {:.2}%", ticker.price_change_24h * 100.0);
            println!("   24h Volume: {:.2} BTC", ticker.volume_24h);
        }
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // 2. Get recent trades
    println!("2. Fetching last 10 trades...");
    match client.get_recent_trades("BTCUSDT", 10).await {
        Ok(trades) => {
            println!("   {:>8} {:>12} {:>10} {:>6}", "Time", "Price", "Qty", "Side");
            println!("   {}", "-".repeat(40));
            for trade in &trades {
                let time = chrono::DateTime::from_timestamp_millis(trade.time)
                    .map(|t| t.format("%H:%M:%S").to_string())
                    .unwrap_or_default();
                println!(
                    "   {:>8} {:>12.2} {:>10.6} {:>6}",
                    time, trade.price, trade.qty, trade.side
                );
            }
        }
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // 3. Get order book
    println!("3. Fetching order book (top 5 levels)...");
    match client.get_orderbook("BTCUSDT", 5).await {
        Ok(orderbook) => {
            println!("   Asks (Sell orders):");
            for ask in orderbook.asks.iter().rev() {
                println!("      ${:.2} - {:.6} BTC", ask.price, ask.qty);
            }
            println!("   -------- Spread --------");
            println!("   Bids (Buy orders):");
            for bid in &orderbook.bids {
                println!("      ${:.2} - {:.6} BTC", bid.price, bid.qty);
            }

            if let (Some(best_ask), Some(best_bid)) = (orderbook.asks.first(), orderbook.bids.first()) {
                let spread = best_ask.price - best_bid.price;
                let spread_pct = (spread / best_bid.price) * 100.0;
                println!("   Spread: ${:.2} ({:.4}%)", spread, spread_pct);
            }
        }
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // 4. Get klines (candlesticks)
    println!("4. Fetching last 5 hourly candles...");
    match client.get_klines("BTCUSDT", "60", 5).await {
        Ok(klines) => {
            println!("   {:>12} {:>10} {:>10} {:>10} {:>10}", "Time", "Open", "High", "Low", "Close");
            println!("   {}", "-".repeat(56));
            for kline in klines.iter().rev() {
                let time = chrono::DateTime::from_timestamp_millis(kline.start_time)
                    .map(|t| t.format("%H:%M").to_string())
                    .unwrap_or_default();
                println!(
                    "   {:>12} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
                    time, kline.open, kline.high, kline.low, kline.close
                );
            }
        }
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // 5. Analyze trades for simple metrics
    println!("5. Analyzing last 100 trades...");
    match client.get_recent_trades("BTCUSDT", 100).await {
        Ok(trades) => {
            let buys: Vec<_> = trades.iter().filter(|t| t.side == "Buy").collect();
            let sells: Vec<_> = trades.iter().filter(|t| t.side == "Sell").collect();

            let buy_volume: f64 = buys.iter().map(|t| t.qty * t.price).sum();
            let sell_volume: f64 = sells.iter().map(|t| t.qty * t.price).sum();

            let total_volume = buy_volume + sell_volume;
            let buy_ratio = (buy_volume / total_volume) * 100.0;

            println!("   Buy trades: {} ({:.1}% volume)", buys.len(), buy_ratio);
            println!("   Sell trades: {} ({:.1}% volume)", sells.len(), 100.0 - buy_ratio);
            println!("   Total volume: ${:.2}", total_volume);

            if let (Some(first), Some(last)) = (trades.last(), trades.first()) {
                let price_change = last.price - first.price;
                let price_change_pct = (price_change / first.price) * 100.0;
                println!("   Price change: ${:.2} ({:.4}%)", price_change, price_change_pct);
            }
        }
        Err(e) => println!("   Error: {}", e),
    }

    println!("\n=== Done ===");
    Ok(())
}
