//! Basic Concept Bottleneck Model example.
//!
//! Demonstrates the core functionality of the CBM for trading.

use concept_bottleneck_trading::api::bybit::generate_sample_klines;
use concept_bottleneck_trading::models::cbm::ConceptBottleneckTrader;

fn main() {
    println!("=================================================");
    println!("  Concept Bottleneck Model - Basic Example");
    println!("=================================================\n");

    // Generate sample data
    println!("Generating sample market data...");
    let klines = generate_sample_klines(100);
    println!("Generated {} candlesticks\n", klines.len());

    // Create the model
    let model = ConceptBottleneckTrader::new();

    // Make a prediction
    println!("Making prediction...\n");
    match model.predict_with_explanation(&klines) {
        Some(prediction) => {
            println!("=== PREDICTION ===");
            println!("Signal: {:?}", prediction.signal);
            println!("Confidence: {:.2}%\n", prediction.confidence * 100.0);

            println!("=== EXTRACTED CONCEPTS ===");
            let concepts = &prediction.concepts;
            println!("  Trend Direction: {:?}", concepts.trend_direction);
            println!("  Trend Strength: {:.4}", concepts.trend_strength);
            println!("  Volatility Regime: {:?}", concepts.volatility_regime);
            println!("  Volatility Value: {:.4}", concepts.volatility_value);
            println!("  Momentum Signal: {:?}", concepts.momentum_signal);
            println!("  Momentum Value: {:.4}", concepts.momentum_value);
            println!("  Volume Profile: {:?}", concepts.volume_profile);
            println!("  Volume Ratio: {:.4}", concepts.volume_ratio);
            println!("  RSI: {:.2}", concepts.rsi);
            println!("  Support Distance: {:.4}", concepts.support_distance);
            println!("  Resistance Distance: {:.4}", concepts.resistance_distance);

            println!("\n=== CONCEPT CONTRIBUTIONS ===");
            let mut contributions: Vec<_> = prediction.contributions.iter().collect();
            contributions.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

            for (name, contrib) in contributions.iter().take(5) {
                let sign = if **contrib >= 0.0 { "+" } else { "" };
                println!("  {}: {}{:.4}", name, sign, contrib);
            }
        }
        None => {
            println!("Error: Could not make prediction. Not enough data.");
        }
    }

    println!("\n=================================================");
    println!("  Example complete!");
    println!("=================================================");
}
