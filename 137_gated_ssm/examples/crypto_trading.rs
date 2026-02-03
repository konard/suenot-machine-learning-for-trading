//! Crypto trading example using Gated SSM with Bybit data.
//!
//! Fetches BTC/USDT kline data from Bybit and generates trading signals.

use gated_ssm::data::{compute_features, fetch_bybit_klines};
use gated_ssm::trading::{GatedSSMStrategy, SignalDirection};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Gated SSM Crypto Trading Example ===\n");

    // Fetch BTC daily data from Bybit
    println!("Fetching BTCUSDT daily klines from Bybit...");
    let klines = fetch_bybit_klines("BTCUSDT", "D", 200).await?;
    println!("Fetched {} klines\n", klines.len());

    // Compute features
    let features = compute_features(&klines);
    println!("Computed {} feature vectors\n", features.len());

    // Initialize strategy
    let mut strategy = GatedSSMStrategy::new(4, 32, 0.05, 60);

    // Generate signals
    let mut long_count = 0;
    let mut short_count = 0;
    let mut neutral_count = 0;

    println!("{:<12} {:>10} {:>8} {:>10}", "Date", "Close", "Signal", "Confidence");
    println!("{}", "-".repeat(45));

    for (i, feat) in features.iter().enumerate() {
        let signal = strategy.on_bar(feat);
        let kline = &klines[i + 1]; // +1 because features start from index 1

        match signal.direction {
            SignalDirection::Long => long_count += 1,
            SignalDirection::Short => short_count += 1,
            SignalDirection::Neutral => neutral_count += 1,
        }

        // Print last 20 signals
        if i >= features.len().saturating_sub(20) {
            let dir_str = match signal.direction {
                SignalDirection::Long => "LONG",
                SignalDirection::Short => "SHORT",
                SignalDirection::Neutral => "NEUTRAL",
            };
            println!(
                "{:<12} {:>10.2} {:>8} {:>10.4}",
                kline.open_time, kline.close, dir_str, signal.confidence
            );
        }
    }

    println!("\n--- Signal Summary ---");
    println!("Long:    {}", long_count);
    println!("Short:   {}", short_count);
    println!("Neutral: {}", neutral_count);

    Ok(())
}
