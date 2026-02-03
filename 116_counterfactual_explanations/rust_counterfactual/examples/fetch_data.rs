//! Example: Fetch market data from Bybit
//!
//! This example demonstrates how to fetch cryptocurrency data from
//! the Bybit exchange API.
//!
//! Run with: cargo run --example fetch_data

use rust_counterfactual::api::BybitClient;

fn main() {
    println!("Fetching market data from Bybit...");
    println!("==================================\n");

    let client = BybitClient::new();

    // Fetch BTCUSDT hourly candles
    match client.get_klines("BTCUSDT", "60", 50) {
        Ok(candles) => {
            println!("Fetched {} candles for BTCUSDT\n", candles.len());

            println!("Last 5 candles:");
            println!("{:<15} {:>12} {:>12} {:>12} {:>12} {:>15}",
                     "Timestamp", "Open", "High", "Low", "Close", "Volume");
            println!("{}", "-".repeat(80));

            for candle in candles.iter().rev().take(5).rev() {
                println!("{:<15} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
                         candle.timestamp,
                         candle.open,
                         candle.high,
                         candle.low,
                         candle.close,
                         candle.volume);
            }
        }
        Err(e) => {
            eprintln!("Error fetching data: {}", e);
            println!("\nNote: This example requires network access to Bybit API.");
            println!("Using sample data instead...\n");

            // Use sample data
            let (features, labels, names) = rust_counterfactual::data::get_sample_data();
            println!("Generated {} samples with {} features", features.len(), names.len());
            println!("Features: {:?}", names);
        }
    }

    // Also try ETH
    println!("\n\nFetching ETHUSDT data...");
    match client.get_klines("ETHUSDT", "60", 10) {
        Ok(candles) => {
            println!("Fetched {} candles for ETHUSDT", candles.len());
            if let Some(last) = candles.last() {
                println!("Latest close price: ${:.2}", last.close);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    println!("\nDone!");
}
