//! Market Regime Detection Example
//!
//! This example demonstrates how to use Hopfield Networks to detect
//! and classify market regimes (bull, bear, ranging, volatile).
//!
//! Run with: cargo run --example regime_detection

use hopfield_trading::data::bybit::BybitClient;
use hopfield_trading::data::Candle;
use hopfield_trading::hopfield::modern::ModernHopfield;
use hopfield_trading::hopfield::HopfieldNetwork;
use hopfield_trading::trading::patterns::{PatternEncoder, PatternEncoderConfig};
use hopfield_trading::trading::MarketRegime;

/// Create synthetic regime patterns for training
fn create_regime_patterns(size: usize) -> Vec<(Vec<f64>, MarketRegime)> {
    let mut patterns = Vec::new();

    // Bull trend pattern: increasing values
    let bull: Vec<f64> = (0..size).map(|i| (i as f64 / size as f64) * 2.0 - 1.0 + 0.3).collect();
    patterns.push((normalize(&bull), MarketRegime::BullTrend));

    // Bear trend pattern: decreasing values
    let bear: Vec<f64> = (0..size).map(|i| 1.0 - (i as f64 / size as f64) * 2.0 - 0.3).collect();
    patterns.push((normalize(&bear), MarketRegime::BearTrend));

    // Ranging pattern: oscillating around zero
    let range: Vec<f64> = (0..size)
        .map(|i| ((i as f64 * 0.5).sin() * 0.3))
        .collect();
    patterns.push((normalize(&range), MarketRegime::Ranging));

    // High volatility pattern: large swings
    let volatile: Vec<f64> = (0..size)
        .map(|i| ((i as f64 * 1.5).sin() * 0.8))
        .collect();
    patterns.push((normalize(&volatile), MarketRegime::HighVolatility));

    // Low volatility pattern: small changes
    let calm: Vec<f64> = (0..size)
        .map(|i| ((i as f64 * 0.2).sin() * 0.1))
        .collect();
    patterns.push((normalize(&calm), MarketRegime::LowVolatility));

    patterns
}

fn normalize(v: &[f64]) -> Vec<f64> {
    let mean: f64 = v.iter().sum::<f64>() / v.len() as f64;
    let std: f64 = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    let std = std.max(1e-10);
    v.iter().map(|x| (x - mean) / std).collect()
}

fn encode_candles_for_regime(candles: &[Candle]) -> Vec<f64> {
    // Simple encoding: normalized close prices
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    normalize(&closes)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("===========================================");
    println!("  Market Regime Detection");
    println!("  Using Hopfield Networks");
    println!("===========================================\n");

    // Initialize client
    let client = BybitClient::new();

    // Fetch data for multiple symbols
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    println!("Fetching data from Bybit...\n");

    let window_size = 24; // 24-hour windows for regime detection

    // Create regime patterns
    let regime_patterns = create_regime_patterns(window_size);

    // Create Modern Hopfield Network
    let mut network = ModernHopfield::with_beta(window_size, 8.0);

    // Store regime patterns
    let patterns: Vec<Vec<f64>> = regime_patterns.iter().map(|(p, _)| p.clone()).collect();
    network.store_patterns(&patterns);

    println!("Stored {} regime patterns:", patterns.len());
    for (_, regime) in &regime_patterns {
        println!("  - {:?}", regime);
    }
    println!();

    // Analyze each symbol
    for symbol in &symbols {
        println!("--- {} Analysis ---", symbol);

        match client.get_klines(symbol, "60", 100).await {
            Ok(candles) => {
                if candles.len() < window_size {
                    println!("Not enough data for {}\n", symbol);
                    continue;
                }

                // Get latest price info
                let latest = &candles[0];
                println!("Current price: ${:.2}", latest.close);

                // Encode recent window
                let recent_candles: Vec<Candle> = candles.iter().take(window_size).cloned().collect();
                let current_pattern = encode_candles_for_regime(&recent_candles);

                // Get retrieval result
                let result = network.retrieve_detailed(&current_pattern);

                // Get attention weights to determine regime
                let attention = network.attention_weights(&current_pattern);

                // Find best matching regime
                let mut best_idx = 0;
                let mut best_weight = 0.0;

                for (i, &weight) in attention.iter().enumerate() {
                    if weight > best_weight {
                        best_weight = weight;
                        best_idx = i;
                    }
                }

                let detected_regime = &regime_patterns[best_idx].1;

                println!("Detected regime: {:?}", detected_regime);
                println!("Confidence: {:.2}%", best_weight * 100.0);
                println!("Pattern match quality: {:.2}%", result.confidence * 100.0);

                // Show regime probabilities
                println!("\nRegime probabilities:");
                for (i, (_, regime)) in regime_patterns.iter().enumerate() {
                    let prob = attention.get(i).unwrap_or(&0.0);
                    let bar_len = (prob * 30.0) as usize;
                    let bar = "█".repeat(bar_len);
                    println!("  {:?}: {:5.1}% {}", regime, prob * 100.0, bar);
                }

                // Trading recommendation based on regime
                println!("\nTrading recommendation:");
                match detected_regime {
                    MarketRegime::BullTrend => {
                        println!("  Strategy: Trend following (LONG bias)");
                        println!("  Position: Consider scaling into longs");
                    }
                    MarketRegime::BearTrend => {
                        println!("  Strategy: Trend following (SHORT bias)");
                        println!("  Position: Consider scaling into shorts or hedging");
                    }
                    MarketRegime::Ranging => {
                        println!("  Strategy: Mean reversion / Grid trading");
                        println!("  Position: Buy support, sell resistance");
                    }
                    MarketRegime::HighVolatility => {
                        println!("  Strategy: Reduce position size");
                        println!("  Position: Widen stops, smaller positions");
                    }
                    MarketRegime::LowVolatility => {
                        println!("  Strategy: Prepare for breakout");
                        println!("  Position: Consider straddle/strangle");
                    }
                    _ => {
                        println!("  Strategy: Wait for clarity");
                        println!("  Position: Reduce exposure");
                    }
                }

                println!();
            }
            Err(e) => {
                println!("Error fetching {}: {}\n", symbol, e);
            }
        }
    }

    // ===================================
    // Regime Transition Analysis
    // ===================================
    println!("===========================================");
    println!("  Historical Regime Analysis (BTC/USDT)");
    println!("===========================================\n");

    // Fetch more data for historical analysis
    let btc_candles = client.get_klines("BTCUSDT", "60", 200).await?;

    if btc_candles.len() >= window_size * 2 {
        println!("Analyzing {} hours of data...\n", btc_candles.len());

        let mut regime_history = Vec::new();

        // Slide window through history
        for i in 0..btc_candles.len() - window_size {
            let window: Vec<Candle> = btc_candles[i..i + window_size].to_vec();
            let pattern = encode_candles_for_regime(&window);
            let attention = network.attention_weights(&pattern);

            // Find dominant regime
            let mut best_idx = 0;
            let mut best_weight = 0.0;
            for (j, &w) in attention.iter().enumerate() {
                if w > best_weight {
                    best_weight = w;
                    best_idx = j;
                }
            }

            regime_history.push((i, regime_patterns[best_idx].1, best_weight));
        }

        // Count regime occurrences
        let mut regime_counts = std::collections::HashMap::new();
        for (_, regime, _) in &regime_history {
            *regime_counts.entry(*regime).or_insert(0) += 1;
        }

        println!("Regime distribution over {} windows:", regime_history.len());
        for (regime, count) in &regime_counts {
            let pct = *count as f64 / regime_history.len() as f64 * 100.0;
            println!("  {:?}: {} ({:.1}%)", regime, count, pct);
        }

        // Detect regime changes
        println!("\nRecent regime transitions (last 10):");
        let mut prev_regime = None;
        let mut transitions = Vec::new();

        for (i, regime, confidence) in &regime_history {
            if let Some(prev) = prev_regime {
                if prev != regime {
                    transitions.push((i, prev, regime, confidence));
                }
            }
            prev_regime = Some(regime);
        }

        for (i, from, to, conf) in transitions.iter().rev().take(10) {
            println!("  Hour {}: {:?} -> {:?} (conf: {:.1}%)", i, from, to, conf * 100.0);
        }
    }

    println!("\n===========================================");
    println!("  Regime Detection Complete");
    println!("===========================================");

    Ok(())
}
