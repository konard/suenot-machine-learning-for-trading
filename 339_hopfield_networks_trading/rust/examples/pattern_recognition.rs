//! Pattern Recognition Example
//!
//! This example demonstrates how to use Hopfield Networks for
//! cryptocurrency market pattern recognition using Bybit data.
//!
//! Run with: cargo run --example pattern_recognition

use hopfield_trading::data::bybit::BybitClient;
use hopfield_trading::hopfield::classical::ClassicalHopfield;
use hopfield_trading::hopfield::modern::ModernHopfield;
use hopfield_trading::hopfield::HopfieldNetwork;
use hopfield_trading::trading::patterns::{PatternEncoder, PatternEncoderConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("===========================================");
    println!("  Hopfield Network Pattern Recognition");
    println!("  Cryptocurrency Market Analysis");
    println!("===========================================\n");

    // Initialize Bybit client
    let client = BybitClient::new();

    println!("Fetching BTC/USDT hourly data from Bybit...\n");

    // Fetch historical candle data
    let candles = client.get_klines("BTCUSDT", "60", 200).await?;

    println!("Fetched {} candles", candles.len());
    println!(
        "Latest price: ${:.2}",
        candles.first().map(|c| c.close).unwrap_or(0.0)
    );
    println!();

    // Configure pattern encoder
    let config = PatternEncoderConfig {
        window_size: 12, // 12-hour patterns
        include_volume: true,
        include_indicators: false, // Keep it simple for demo
        ..Default::default()
    };

    let encoder = PatternEncoder::new(config);

    // Encode patterns
    println!("Encoding patterns with window size {}...", encoder.config().window_size);
    let patterns = encoder.encode_candles(&candles);
    println!("Created {} patterns\n", patterns.len());

    if patterns.is_empty() {
        println!("Not enough data to create patterns!");
        return Ok(());
    }

    // Get pattern dimension
    let dimension = patterns[0].len();
    println!("Pattern dimension: {}", dimension);

    // Split patterns into training and test sets
    let split_idx = patterns.len() * 80 / 100;
    let (train_patterns, test_patterns) = patterns.split_at(split_idx);

    println!("Training patterns: {}", train_patterns.len());
    println!("Test patterns: {}\n", test_patterns.len());

    // ===================================
    // Classical Hopfield Network
    // ===================================
    println!("--- Classical Hopfield Network ---");

    let mut classical = ClassicalHopfield::new(dimension);
    let capacity = classical.capacity();
    println!("Theoretical capacity: {} patterns", capacity);

    // Store limited patterns (classical has low capacity)
    let store_count = capacity.min(train_patterns.len());
    classical.store_patterns(&train_patterns[..store_count].to_vec());
    println!("Stored {} patterns", store_count);

    // Test retrieval
    if let Some(test_pattern) = test_patterns.first() {
        let result = classical.retrieve_detailed(test_pattern);
        println!("\nRetrieval test:");
        println!("  Converged: {}", result.converged);
        println!("  Iterations: {}", result.iterations);
        println!("  Energy: {:.4}", result.energy);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
        if let Some(idx) = result.matched_index {
            println!("  Matched pattern index: {}", idx);
        }
    }

    // ===================================
    // Modern Hopfield Network
    // ===================================
    println!("\n--- Modern Hopfield Network ---");

    let mut modern = ModernHopfield::with_beta(dimension, 5.0);
    println!("Capacity: {} patterns", modern.capacity());

    // Store all training patterns (modern has exponential capacity)
    modern.store_patterns(&train_patterns.to_vec());
    println!("Stored {} patterns", train_patterns.len());

    // Test retrieval on multiple test patterns
    println!("\nRetrieval tests on {} test patterns:", test_patterns.len().min(5));

    for (i, test_pattern) in test_patterns.iter().take(5).enumerate() {
        let result = modern.retrieve_detailed(test_pattern);

        println!(
            "  Test {}: Conf={:.2}%, Iter={}, Match={}",
            i + 1,
            result.confidence * 100.0,
            result.iterations,
            result.matched_index.map(|x| x.to_string()).unwrap_or("None".to_string())
        );
    }

    // ===================================
    // Pattern Similarity Analysis
    // ===================================
    println!("\n--- Pattern Similarity Analysis ---");

    // Get attention weights for a query
    if let Some(query) = test_patterns.last() {
        let attention = modern.attention_weights(query);

        // Find top 3 most similar patterns
        let mut indexed: Vec<(usize, f64)> = attention.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Top 3 most similar stored patterns:");
        for (rank, (idx, weight)) in indexed.iter().take(3).enumerate() {
            println!("  {}. Pattern {}: {:.2}% attention", rank + 1, idx, weight * 100.0);
        }
    }

    // ===================================
    // Summary
    // ===================================
    println!("\n===========================================");
    println!("  Summary");
    println!("===========================================");
    println!("Symbol: BTC/USDT");
    println!("Timeframe: 1 hour");
    println!("Pattern window: {} hours", encoder.config().window_size);
    println!("Total patterns analyzed: {}", patterns.len());
    println!("Network dimension: {}", dimension);
    println!(
        "\nClassical Hopfield: {} patterns stored",
        classical.pattern_count()
    );
    println!(
        "Modern Hopfield: {} patterns stored",
        modern.pattern_count()
    );
    println!("\nNote: Modern Hopfield can store exponentially more patterns");
    println!("and provides attention-weighted retrieval for better matching.");

    Ok(())
}
