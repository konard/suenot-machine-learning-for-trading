//! Fetch market data from Bybit
//!
//! This example demonstrates how to fetch OHLCV data from Bybit exchange.

use efficientnet_trading::api::BybitClient;
use tracing_subscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Bybit Data Fetcher ===\n");

    // Create client
    let client = BybitClient::new();

    // Fetch recent klines
    println!("Fetching BTCUSDT 5-minute candles...");
    let candles = client.fetch_klines("BTCUSDT", "5", 100).await?;

    println!("Fetched {} candles\n", candles.len());

    // Display last 5 candles
    println!("Last 5 candles:");
    println!("{:<15} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume");
    println!("{}", "-".repeat(85));

    for candle in candles.iter().rev().take(5).rev() {
        println!("{:<15} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.4}",
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        );
    }

    // Fetch order book
    println!("\nFetching BTCUSDT order book...");
    let orderbook = client.fetch_orderbook("BTCUSDT", 10).await?;

    println!("\nOrder Book (Top 5 levels):");
    println!("{:>15} {:>15}  |  {:>15} {:>15}",
        "Bid Price", "Bid Qty", "Ask Price", "Ask Qty");
    println!("{}", "-".repeat(70));

    for i in 0..5 {
        let bid = orderbook.bids.get(i);
        let ask = orderbook.asks.get(i);

        let (bid_price, bid_qty) = bid
            .map(|b| (format!("{:.2}", b.price), format!("{:.4}", b.quantity)))
            .unwrap_or(("-".to_string(), "-".to_string()));

        let (ask_price, ask_qty) = ask
            .map(|a| (format!("{:.2}", a.price), format!("{:.4}", a.quantity)))
            .unwrap_or(("-".to_string(), "-".to_string()));

        println!("{:>15} {:>15}  |  {:>15} {:>15}",
            bid_price, bid_qty, ask_price, ask_qty);
    }

    // Calculate some statistics
    if let (Some(bid), Some(ask)) = (orderbook.best_bid(), orderbook.best_ask()) {
        println!("\nMarket Statistics:");
        println!("  Best Bid:       ${:.2}", bid);
        println!("  Best Ask:       ${:.2}", ask);
        println!("  Mid Price:      ${:.2}", orderbook.mid_price().unwrap_or(0.0));
        println!("  Spread:         ${:.2}", orderbook.spread().unwrap_or(0.0));
        println!("  Spread (bps):   {:.2}", orderbook.spread_bps().unwrap_or(0.0));
        println!("  Vol Imbalance:  {:.4}", orderbook.volume_imbalance());
    }

    // Fetch recent trades
    println!("\nFetching recent trades...");
    let trades = client.fetch_trades("BTCUSDT", 10).await?;

    println!("\nRecent Trades:");
    println!("{:<20} {:>10} {:>12} {:>12}",
        "ID", "Side", "Price", "Quantity");
    println!("{}", "-".repeat(60));

    for trade in trades.iter().take(10) {
        let side = if trade.is_buy() { "BUY" } else { "SELL" };
        println!("{:<20} {:>10} {:>12.2} {:>12.4}",
            &trade.id[..trade.id.len().min(18)],
            side,
            trade.price,
            trade.quantity
        );
    }

    println!("\nDone!");
    Ok(())
}
