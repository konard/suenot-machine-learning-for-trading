//! Fetch market data from Bybit
//!
//! Example: cargo run --bin fetch_bybit_data

use equivariant_gnn_trading::BybitClient;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== Bybit Data Fetcher for E-GNN Trading ===\n");

    let client = BybitClient::new();
    let symbols = &["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"];

    println!("Fetching kline data for {} symbols...\n", symbols.len());

    let mut all_candles = HashMap::new();

    for symbol in symbols {
        match client.get_klines(symbol, "60", 168, None, None).await {
            Ok(candles) => {
                println!("{}: {} candles fetched", symbol, candles.len());
                if let Some(last) = candles.last() {
                    println!("  Latest: O={:.2} H={:.2} L={:.2} C={:.2} V={:.0}",
                        last.open, last.high, last.low, last.close, last.volume);
                }
                all_candles.insert(symbol.to_string(), candles);
            }
            Err(e) => println!("{}: Error - {}", symbol, e),
        }
    }

    println!("\n=== Ticker Information ===\n");

    for symbol in symbols {
        if let Ok(ticker) = client.get_ticker(symbol).await {
            println!("{}: ${:.2} (24h: {:.2}%, Funding: {:.4}%)",
                ticker.symbol, ticker.last_price,
                ticker.price_change_24h * 100.0,
                ticker.funding_rate * 100.0);
        }
    }

    println!("\n=== Order Book Snapshot ===\n");

    if let Ok(ob) = client.get_orderbook("BTCUSDT", 5).await {
        println!("BTCUSDT Order Book:");
        println!("  Best Bid: ${:.2}", ob.best_bid().unwrap_or(0.0));
        println!("  Best Ask: ${:.2}", ob.best_ask().unwrap_or(0.0));
        println!("  Spread: {:.4}%", ob.spread_pct().unwrap_or(0.0) * 100.0);
        println!("  Imbalance (5 levels): {:.2}", ob.order_imbalance(5));
    }

    println!("\nTotal candles fetched: {}", all_candles.values().map(|c| c.len()).sum::<usize>());
    println!("\nDone!");

    Ok(())
}
