//! Live data example: Fetch data from Bybit and predict regimes
//!
//! This example demonstrates fetching real market data from Bybit
//! and using the zero-shot model to analyze multiple assets.
//!
//! Run with: cargo run --example bybit_live

use zero_shot_trading::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    println!("Zero-Shot Trading: Bybit Live Data Example");
    println!("==========================================\n");

    // Create Bybit client
    let client = BybitClient::new();
    println!("✓ Bybit client initialized\n");

    // Define assets to analyze
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let interval = "15"; // 15-minute klines
    let limit = 200;

    // Create model and feature generator
    let model_config = zero_shot_trading::model::network::ModelConfig::default();
    let model = ZeroShotModel::new(model_config);
    let feature_generator = FeatureGenerator::default();

    println!("Analyzing {} assets...\n", symbols.len());

    // Analyze each asset
    for symbol in &symbols {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  {}", symbol);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Fetch klines
        match client.fetch_klines(symbol, interval, limit).await {
            Ok(klines) => {
                println!("  Fetched {} klines", klines.len());

                // Get current price
                if let Some(last) = klines.last() {
                    println!("  Current price: ${:.2}", last.close);
                }

                // Generate features
                match feature_generator.generate(&klines) {
                    Ok(features) => {
                        println!("  Features: {} timesteps x {} features", features.len(), features[0].len());

                        // Predict regime with scores
                        match model.predict_regime_with_scores(&features) {
                            Ok((regime, scores)) => {
                                // Calculate softmax probabilities
                                let sum_exp: f64 = scores.iter().map(|s| s.exp()).sum();
                                let confidence = scores.iter()
                                    .map(|s| s.exp() / sum_exp)
                                    .fold(f64::NEG_INFINITY, f64::max);

                                println!("\n  📊 Prediction:");
                                println!("     Regime: {:?}", regime);
                                println!("     Confidence: {:.1}%", confidence * 100.0);

                                // Show all probabilities
                                let regimes = MarketRegime::all();
                                println!("\n  📈 Regime Probabilities:");
                                for (i, &score) in scores.iter().enumerate() {
                                    let prob = score.exp() / sum_exp;
                                    let bar_len = (prob * 20.0) as usize;
                                    let bar = "█".repeat(bar_len);
                                    println!("     {:18} {:>5.1}% {}",
                                        format!("{:?}", regimes[i]),
                                        prob * 100.0,
                                        bar);
                                }
                            }
                            Err(e) => println!("  ✗ Prediction error: {}", e),
                        }
                    }
                    Err(e) => println!("  ✗ Feature error: {}", e),
                }
            }
            Err(e) => {
                println!("  ✗ Failed to fetch data: {}", e);
            }
        }

        println!();
    }

    // Also fetch ticker info for display
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Current Market Tickers");
    println!("═══════════════════════════════════════════════════════════════\n");

    for symbol in &symbols {
        match client.fetch_ticker(symbol).await {
            Ok(ticker) => {
                let change_pct = ticker.price_24h_pct;
                let direction = if change_pct >= 0.0 { "↑" } else { "↓" };
                println!(
                    "  {} | ${:.2} | {} {:.2}% | Vol: ${:.0}M",
                    symbol,
                    ticker.last_price,
                    direction,
                    change_pct.abs() * 100.0,
                    ticker.volume_24h / 1_000_000.0
                );
            }
            Err(e) => {
                println!("  {} | Error: {}", symbol, e);
            }
        }
    }

    println!("\n✓ Analysis complete!");
    Ok(())
}
