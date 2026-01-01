//! Live Bybit Data Integration Example
//!
//! This example demonstrates fetching live market data from Bybit
//! and analyzing it with SE-enhanced feature attention.

use se_trading::prelude::*;
use se_trading::data::bybit::{BybitClient, generate_sample_data};
use se_trading::data::features::FEATURE_NAMES;

#[tokio::main]
async fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Bybit Live Data with SE Analysis                         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Try to fetch live data, fall back to sample data if API unavailable
    let klines = match fetch_live_data().await {
        Ok(data) => {
            println!("✓ Successfully fetched live data from Bybit!\n");
            data
        }
        Err(e) => {
            println!("⚠ Could not fetch live data: {}", e);
            println!("  Using generated sample data instead.\n");
            generate_sample_data(200, 50000.0)
        }
    };

    println!("Data summary:");
    println!("  - Total candles: {}", klines.len());
    if let (Some(first), Some(last)) = (klines.first(), klines.last()) {
        println!("  - Price range: {:.2} - {:.2}", first.close, last.close);
        println!("  - Last close: {:.2}", last.close);
    }
    println!();

    // Compute features
    let feature_engine = FeatureEngine::default();
    let features = feature_engine.compute_features(&klines);

    println!("═══════════════════════════════════════════════════════════");
    println!("                   FEATURE COMPUTATION                      ");
    println!("═══════════════════════════════════════════════════════════\n");

    // Display last row of features
    let last_idx = features.nrows() - 1;
    println!("Current feature values (last candle):\n");

    for (i, &name) in FEATURE_NAMES.iter().enumerate() {
        let value = features[[last_idx, i]];
        let bar_len = ((value.abs() * 20.0).min(20.0)) as usize;
        let bar = if value >= 0.0 {
            format!("{:>20}│{}",
                    "",
                    "█".repeat(bar_len))
        } else {
            format!("{:>20}│{}",
                    "█".repeat(bar_len),
                    "")
        };

        println!("  {:<15} {:>8.4} {}", name, value, bar);
    }

    // Apply SE block analysis
    println!("\n═══════════════════════════════════════════════════════════");
    println!("                 SE ATTENTION ANALYSIS                      ");
    println!("═══════════════════════════════════════════════════════════\n");

    let se_block = SEBlock::new(FEATURE_NAMES.len(), 4);
    let attention = se_block.get_attention_weights(&features);

    // Sort by attention weight
    let mut feature_attention: Vec<(&str, f64)> = FEATURE_NAMES
        .iter()
        .zip(attention.iter())
        .map(|(&name, &weight)| (name, weight))
        .collect();

    feature_attention.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Feature attention ranking:\n");
    for (i, (name, weight)) in feature_attention.iter().enumerate() {
        let bar_len = (weight * 50.0) as usize;
        let bar = "█".repeat(bar_len);
        let rank_indicator = match i {
            0 => " ← HIGHEST",
            1 => " ← 2nd",
            2 => " ← 3rd",
            _ => "",
        };

        println!(
            "  {:2}. {:<15} {:.4} │{}│{}",
            i + 1,
            name,
            weight,
            bar,
            rank_indicator
        );
    }

    // Generate trading signal
    println!("\n═══════════════════════════════════════════════════════════");
    println!("                   TRADING SIGNAL                           ");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut strategy = SEMomentumStrategy::default_strategy();

    // Get signal
    if let Some(signal) = strategy.generate_signal(&klines) {
        let direction_str = match signal.direction {
            Direction::Long => "🟢 LONG (BUY)",
            Direction::Short => "🔴 SHORT (SELL)",
            Direction::Neutral => "⚪ NEUTRAL (WAIT)",
        };

        println!("  Signal: {}", direction_str);
        println!("  Strength: {:.2}%", signal.strength * 100.0);
        println!("  Confidence: {:.2}%", signal.confidence * 100.0);
        println!("  Raw value: {:.4}", signal.raw_signal);

        // Display top features contributing to this signal
        if let Some(ref attn) = signal.feature_attention {
            println!("\n  Top features for this signal:");
            let top = signal.top_features(FEATURE_NAMES, 3);
            for (name, weight) in top {
                println!("    - {} ({:.3})", name, weight);
            }
        }
    } else {
        println!("  No signal generated (filtered or insufficient data)");
    }

    // Market regime analysis
    println!("\n═══════════════════════════════════════════════════════════");
    println!("                  MARKET REGIME ANALYSIS                    ");
    println!("═══════════════════════════════════════════════════════════\n");

    if let Some(analysis) = strategy.analyze_attention(&klines) {
        let top_features = analysis.top_features(5);

        println!("  Attention entropy: {:.4}", analysis.attention_entropy);
        println!("  Focus level: {}",
                 if analysis.is_focused(1.5) { "HIGH (concentrated)" }
                 else { "LOW (distributed)" });
        println!("\n  Most attended features:");

        for (name, weight) in top_features {
            println!("    - {}: {:.3}", name, weight);
        }

        // Interpret market regime based on top features
        println!("\n  Regime interpretation:");
        let top_feature = &analysis.feature_weights[0].0;

        match top_feature.as_str() {
            "atr" | "volatility" => {
                println!("    📊 HIGH VOLATILITY REGIME");
                println!("       The model is focusing on volatility indicators.");
                println!("       Consider tighter stops and smaller positions.");
            }
            "rsi" | "momentum" => {
                println!("    📈 MOMENTUM REGIME");
                println!("       The model is focusing on momentum indicators.");
                println!("       Good for trend-following strategies.");
            }
            "macd" | "macd_signal" => {
                println!("    🔄 TREND TRANSITION REGIME");
                println!("       The model is focusing on trend change indicators.");
                println!("       Watch for potential reversals.");
            }
            "bollinger_pct" => {
                println!("    ↔️ MEAN REVERSION REGIME");
                println!("       The model is focusing on band indicators.");
                println!("       Market may be ranging/consolidating.");
            }
            "volume_ma_ratio" | "obv_normalized" => {
                println!("    📦 VOLUME-DRIVEN REGIME");
                println!("       The model is focusing on volume indicators.");
                println!("       Pay attention to breakouts with volume.");
            }
            _ => {
                println!("    ❓ MIXED REGIME");
                println!("       No dominant indicator detected.");
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("                      SUMMARY                               ");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("  The SE block dynamically adjusts feature importance");
    println!("  based on current market conditions, allowing the model");
    println!("  to focus on the most relevant indicators.\n");

    println!("✓ Analysis complete!");
}

/// Attempt to fetch live data from Bybit
async fn fetch_live_data() -> Result<Vec<se_trading::data::bybit::Kline>, String> {
    let client = BybitClient::new();

    client
        .get_klines("BTCUSDT", "15", 200)
        .await
        .map_err(|e| e.to_string())
}
