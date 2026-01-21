//! Basic example: Zero-shot regime prediction
//!
//! This example demonstrates how to use the zero-shot model
//! to predict market regimes from historical kline data.
//!
//! Run with: cargo run --example basic_prediction

use zero_shot_trading::prelude::*;

fn main() -> Result<()> {
    println!("Zero-Shot Trading: Basic Prediction Example");
    println!("============================================\n");

    // Create model with default configuration
    let config = zero_shot_trading::model::network::ModelConfig::default();
    let model = ZeroShotModel::new(config);

    println!("Model created with:");
    println!("  - Input features: 11");
    println!("  - Embedding dimension: 64");
    println!("  - Output classes: 5 market regimes\n");

    // Create synthetic market data for demonstration
    let features = create_synthetic_features(50, 11);
    println!("Generated {} timesteps of synthetic features", features.len());

    // Predict market regime
    match model.predict_regime(&features) {
        Ok(regime) => {
            println!("\nPredicted Regime: {:?}", regime);
            println!("Regime Description: {}", describe_regime(regime));
        }
        Err(e) => {
            println!("Prediction error: {}", e);
        }
    }

    // Predict with confidence scores
    match model.predict_regime_with_scores(&features) {
        Ok((regime, scores)) => {
            println!("\n--- Detailed Prediction ---");
            println!("Predicted: {:?}", regime);
            println!("\nRegime Scores:");
            let regimes = MarketRegime::all();
            for (i, &score) in scores.iter().enumerate() {
                let softmax = score.exp() / scores.iter().map(|s| s.exp()).sum::<f64>();
                println!("  {:?}: {:.2}%", regimes[i], softmax * 100.0);
            }
        }
        Err(e) => {
            println!("Prediction error: {}", e);
        }
    }

    println!("\n✓ Example completed successfully!");
    Ok(())
}

/// Create synthetic features for demonstration
fn create_synthetic_features(timesteps: usize, num_features: usize) -> Vec<Vec<f64>> {
    let mut features = Vec::with_capacity(timesteps);

    // Create a trending pattern (simulating uptrend)
    for i in 0..timesteps {
        let mut row = Vec::with_capacity(num_features);
        let trend = (i as f64 / timesteps as f64) * 0.1;

        // Returns (positive for uptrend)
        row.push(0.002 + trend * 0.01 + (rand::random::<f64>() - 0.5) * 0.01);
        // Log returns
        row.push(0.002 + trend * 0.01 + (rand::random::<f64>() - 0.5) * 0.01);
        // High-low range
        row.push(0.02 + (rand::random::<f64>() - 0.5) * 0.01);
        // Close position
        row.push(0.6 + (rand::random::<f64>() - 0.5) * 0.2);
        // Volume ratio
        row.push(1.0 + (rand::random::<f64>() - 0.5) * 0.5);
        // Volatility 20
        row.push(0.015 + (rand::random::<f64>() - 0.5) * 0.005);
        // Volatility 5
        row.push(0.02 + (rand::random::<f64>() - 0.5) * 0.01);
        // SMA ratio
        row.push(1.01 + trend * 0.05);
        // RSI (normalized)
        row.push(0.55 + trend * 0.3);
        // MACD
        row.push(0.001 + trend * 0.002);
        // Bollinger position
        row.push(0.6 + trend * 0.2);

        features.push(row);
    }

    features
}

/// Get human-readable description of regime
fn describe_regime(regime: MarketRegime) -> &'static str {
    match regime {
        MarketRegime::StrongUptrend => "Strong bullish momentum - consider long positions",
        MarketRegime::WeakUptrend => "Mild bullish bias - cautious long exposure",
        MarketRegime::Sideways => "Range-bound market - avoid directional trades",
        MarketRegime::WeakDowntrend => "Mild bearish bias - reduce long exposure",
        MarketRegime::StrongDowntrend => "Strong bearish momentum - consider short positions",
    }
}
