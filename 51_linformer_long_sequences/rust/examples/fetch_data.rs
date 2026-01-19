//! Example: Fetch market data from Bybit API.
//!
//! Run with: cargo run --example fetch_data

use linformer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("Linformer Data Fetching Example");
    println!("================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch kline data
    println!("Fetching BTCUSDT 1h klines from Bybit...");
    let klines = client
        .get_klines("BTCUSDT", "60", 500, None, None)
        .await?;

    println!("Fetched {} klines", klines.len());

    if let Some(first) = klines.first() {
        println!(
            "\nFirst kline: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={:.2}",
            first.open, first.high, first.low, first.close, first.volume
        );
    }

    if let Some(last) = klines.last() {
        println!(
            "Last kline:  Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={:.2}",
            last.open, last.high, last.low, last.close, last.volume
        );
    }

    // Load into DataLoader
    println!("\nProcessing data...");
    let mut loader = DataLoader::new();
    loader.from_klines(&klines)?;

    // Get prices for feature calculation
    if let Some(prices) = loader.get_prices("close") {
        println!("Close prices shape: {}", prices.len());

        // Calculate technical features
        let features = TechnicalFeatures::calculate_all(&prices);
        println!("Features matrix shape: {:?}", features.dim());

        // Normalize features
        let normalized = TechnicalFeatures::normalize_zscore(&features);
        println!("Normalized features shape: {:?}", normalized.dim());

        // Calculate some statistics
        let price_mean = prices.mean().unwrap_or(0.0);
        let price_min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let price_max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\nPrice Statistics:");
        println!("  Mean:  ${:.2}", price_mean);
        println!("  Min:   ${:.2}", price_min);
        println!("  Max:   ${:.2}", price_max);
        println!("  Range: ${:.2}", price_max - price_min);
    }

    println!("\nDone!");
    Ok(())
}
