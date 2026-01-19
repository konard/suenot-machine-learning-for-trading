//! Bybit API client example
//!
//! This example demonstrates how to fetch cryptocurrency data from Bybit
//! and combine it with earnings call analysis.

use earnings_call_analyzer::api::BybitClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Bybit API Client Example ===\n");

    let client = BybitClient::new();

    // Fetch ticker data
    println!("--- Fetching BTCUSDT Ticker ---\n");
    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            println!("Symbol: {}", ticker.symbol);
            println!("Last Price: ${:.2}", ticker.last_price);
            println!("Bid: ${:.2}", ticker.bid_price);
            println!("Ask: ${:.2}", ticker.ask_price);
            println!("24h Volume: {:.2}", ticker.volume_24h);
            println!("24h Change: {:.2}%", ticker.price_change_24h);
        }
        Err(e) => {
            println!("Error fetching ticker: {}", e);
            println!("(This is expected if not connected to the internet)");
        }
    }

    // Fetch candlestick data
    println!("\n--- Fetching Recent Candles ---\n");
    match client.get_klines("BTCUSDT", "60", 5).await {
        Ok(candles) => {
            println!("Fetched {} candles:\n", candles.len());
            for (i, candle) in candles.iter().enumerate() {
                println!("Candle {}: Open=${:.2} High=${:.2} Low=${:.2} Close=${:.2} Vol={:.2}",
                    i + 1,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume
                );
            }
        }
        Err(e) => {
            println!("Error fetching candles: {}", e);
            println!("(This is expected if not connected to the internet)");
        }
    }

    // Fetch order book
    println!("\n--- Fetching Order Book ---\n");
    match client.get_orderbook("BTCUSDT", 5).await {
        Ok(orderbook) => {
            println!("Top 5 Bids:");
            for bid in orderbook.bids.iter().take(5) {
                println!("  ${:.2} x {:.4}", bid.price, bid.quantity);
            }
            println!("\nTop 5 Asks:");
            for ask in orderbook.asks.iter().take(5) {
                println!("  ${:.2} x {:.4}", ask.price, ask.quantity);
            }
        }
        Err(e) => {
            println!("Error fetching orderbook: {}", e);
            println!("(This is expected if not connected to the internet)");
        }
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
