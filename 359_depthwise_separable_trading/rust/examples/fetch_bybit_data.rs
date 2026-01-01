//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the BybitClient to fetch
//! historical kline data and display basic statistics.
//!
//! Run with: cargo run --example fetch_bybit_data

use dsc_trading::data::{BybitClient, CandleSeries, Timeframe};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    env_logger::init();

    println!("===========================================");
    println!("  Bybit Data Fetching Example");
    println!("===========================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Symbols to fetch
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for symbol in symbols {
        println!("Fetching {} hourly data...", symbol);

        // Fetch 500 hourly candles
        match client.get_klines(symbol, "60", 500).await {
            Ok(candles) => {
                println!("  Received {} candles", candles.len());

                if let Some(series) = CandleSeries::new(candles) {
                    // Calculate statistics
                    let close = series.close_prices();
                    let returns = series.returns();

                    let first_price = close[0];
                    let last_price = close[close.len() - 1];
                    let price_change = (last_price - first_price) / first_price * 100.0;

                    let avg_return = returns.mean().unwrap_or(0.0) * 100.0;
                    let volatility = returns
                        .mapv(|x| x.powi(2))
                        .mean()
                        .unwrap_or(0.0)
                        .sqrt()
                        * 100.0;

                    println!("  Price range: ${:.2} - ${:.2}", first_price, last_price);
                    println!("  Period change: {:.2}%", price_change);
                    println!("  Avg hourly return: {:.4}%", avg_return);
                    println!("  Hourly volatility: {:.4}%", volatility);
                }
            }
            Err(e) => {
                eprintln!("  Error fetching {}: {}", symbol, e);
            }
        }

        println!();
    }

    // Fetch order book for BTC
    println!("Fetching BTCUSDT order book...");
    match client.get_orderbook("BTCUSDT", 25).await {
        Ok(ob) => {
            println!("  Best Bid: ${:.2}", ob.best_bid().unwrap_or(0.0));
            println!("  Best Ask: ${:.2}", ob.best_ask().unwrap_or(0.0));
            println!("  Spread: ${:.2}", ob.spread().unwrap_or(0.0));
            println!("  Spread (bps): {:.2}", ob.spread_bps().unwrap_or(0.0));
            println!("  Bid Volume: {:.4} BTC", ob.total_bid_volume());
            println!("  Ask Volume: {:.4} BTC", ob.total_ask_volume());
            println!("  Imbalance: {:.4}", ob.imbalance());
        }
        Err(e) => {
            eprintln!("  Error fetching order book: {}", e);
        }
    }

    println!();

    // Fetch ticker
    println!("Fetching BTCUSDT ticker...");
    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            println!("  Last Price: ${:.2}", ticker.last_price);
            println!("  24h Change: {:.2}%", ticker.price_change_pct);
            println!("  24h High: ${:.2}", ticker.high_24h);
            println!("  24h Low: ${:.2}", ticker.low_24h);
            println!("  24h Volume: {:.2} BTC", ticker.volume_24h);
            println!("  24h Turnover: ${:.2}M", ticker.turnover_24h / 1_000_000.0);
        }
        Err(e) => {
            eprintln!("  Error fetching ticker: {}", e);
        }
    }

    println!("\n===========================================");
    println!("  Data fetching complete!");
    println!("===========================================");

    Ok(())
}
