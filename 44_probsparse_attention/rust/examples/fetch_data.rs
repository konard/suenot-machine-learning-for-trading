//! Example: Fetching data from Bybit API
//!
//! Run with: cargo run --example fetch_data

use informer_probsparse::{BybitClient, DataLoader};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    println!("=== Informer ProbSparse: Data Fetching Example ===\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch Bitcoin hourly data
    println!("Fetching BTCUSDT hourly data...");
    let klines = client.get_klines("BTCUSDT", "60", 500).await?;

    println!("Fetched {} klines", klines.len());
    println!("Date range: {} to {}",
        klines.first().map(|k| k.datetime().to_string()).unwrap_or_default(),
        klines.last().map(|k| k.datetime().to_string()).unwrap_or_default()
    );

    // Show sample data
    println!("\nSample data (first 5 klines):");
    for k in klines.iter().take(5) {
        println!("  {} - O:{:.2} H:{:.2} L:{:.2} C:{:.2} V:{:.2}",
            k.datetime().format("%Y-%m-%d %H:%M"),
            k.open, k.high, k.low, k.close, k.volume
        );
    }

    // Prepare dataset
    println!("\n--- Preparing Dataset ---\n");

    let loader = DataLoader::new();
    let dataset = loader.prepare_dataset(&klines, 96, 24)?;

    println!("Dataset prepared:");
    println!("  Samples: {}", dataset.n_samples);
    println!("  Sequence length: {}", dataset.seq_len);
    println!("  Prediction horizon: {}", dataset.pred_len);
    println!("  Features: {}", dataset.n_features);

    // Split data
    let (train, val, test) = dataset.split(0.7, 0.15);

    println!("\nData split:");
    println!("  Train samples: {}", train.n_samples);
    println!("  Validation samples: {}", val.n_samples);
    println!("  Test samples: {}", test.n_samples);

    // Fetch multiple symbols
    println!("\n--- Fetching Multiple Symbols ---\n");

    let symbols = &["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let multi_data = client.get_multi_klines(symbols, "60", 100).await?;

    for (symbol, klines) in &multi_data {
        if let Some(latest) = klines.last() {
            println!("{}: {} klines, latest close: ${:.2}",
                symbol, klines.len(), latest.close);
        } else {
            println!("{}: No data available", symbol);
        }
    }

    println!("\n=== Done ===");

    Ok(())
}
