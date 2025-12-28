//! Example: Fetching market data from Bybit
//!
//! Run with: cargo run --example fetch_data

use ml_crypto::api::BybitClient;
use ml_crypto::data::DataLoader;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Bybit Data Fetching Example ===\n");

    let client = BybitClient::new();

    // Fetch BTCUSDT hourly candles
    println!("Fetching BTCUSDT 1h candles...");
    let candles = client.get_klines("BTCUSDT", "1h", 100).await?;

    println!("Fetched {} candles\n", candles.len());

    // Display first few candles
    println!("First 5 candles:");
    println!("{:>24} {:>12} {:>12} {:>12} {:>12} {:>15}",
             "Time", "Open", "High", "Low", "Close", "Volume");
    println!("{:-<95}", "");

    for candle in candles.iter().take(5) {
        println!(
            "{:>24} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            candle.datetime().format("%Y-%m-%d %H:%M:%S"),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        );
    }

    // Calculate some statistics
    println!("\n=== Statistics ===");

    let avg_volume = candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64;
    let avg_range = candles.iter().map(|c| c.range()).sum::<f64>() / candles.len() as f64;
    let bullish_count = candles.iter().filter(|c| c.is_bullish()).count();

    println!("Average volume: {:.2}", avg_volume);
    println!("Average range: {:.2}", avg_range);
    println!("Bullish candles: {} ({:.1}%)",
             bullish_count,
             bullish_count as f64 / candles.len() as f64 * 100.0);

    // Fetch order book
    println!("\n=== Order Book ===");
    let orderbook = client.get_orderbook("BTCUSDT", 5).await?;

    println!("Best bid: {:.2} @ {:.4}",
             orderbook.bids[0].price,
             orderbook.bids[0].quantity);
    println!("Best ask: {:.2} @ {:.4}",
             orderbook.asks[0].price,
             orderbook.asks[0].quantity);
    println!("Spread: {:.2} ({:.4}%)",
             orderbook.spread().unwrap_or(0.0),
             orderbook.spread_pct().unwrap_or(0.0));
    println!("Imbalance: {:.4}", orderbook.imbalance());

    // Fetch recent trades
    println!("\n=== Recent Trades ===");
    let trades = client.get_recent_trades("BTCUSDT", 10).await?;

    println!("Last 5 trades:");
    for trade in trades.iter().take(5) {
        println!("  {:?} {:.4} @ {:.2}", trade.side, trade.quantity, trade.price);
    }

    // Save data to file
    println!("\n=== Saving Data ===");
    DataLoader::save_candles(&candles, "btcusdt_1h.csv")?;
    println!("Saved candles to btcusdt_1h.csv");

    // Also save as JSON
    DataLoader::save_candles_json(&candles, "btcusdt_1h.json")?;
    println!("Saved candles to btcusdt_1h.json");

    Ok(())
}
