//! Example: Simple prediction with rule-based model
//!
//! This example demonstrates the prediction pipeline using
//! a rule-based model (since actual neural network inference
//! requires ONNX Runtime or LibTorch integration).

use efficientnet_trading::prelude::*;
use efficientnet_trading::model::config::ModelConfig;
use efficientnet_trading::model::inference::{ModelInference, RuleBasedModel, DummyModel};

fn main() -> anyhow::Result<()> {
    println!("EfficientNet Trading - Simple Prediction Example");
    println!("=================================================\n");

    // Generate sample price data
    let prices: Vec<f64> = (0..120)
        .map(|i| 45000.0 + 500.0 * (i as f64 * 0.1).sin() + 100.0 * (i as f64 * 0.3).cos())
        .collect();

    println!("Generated {} price points", prices.len());
    println!("Price range: ${:.2} - ${:.2}\n",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Create image generator
    let generator = ImageStackGenerator::new(64);

    // Create multi-transform image
    let image = generator.create_multi_transform_stack(&prices);
    println!("Created image with shape: {:?}", image.dim());

    // Create model configuration
    let config = ModelConfig::b2();
    println!("\nModel Configuration:");
    println!("  Variant: {:?}", config.variant);
    println!("  Image Size: {}", config.image_size);
    println!("  Num Classes: {}", config.num_classes);
    println!("  Predict Magnitude: {}", config.predict_magnitude);

    // Test with Dummy Model (random predictions)
    println!("\n--- Dummy Model (Random Predictions) ---");
    let dummy_model = DummyModel::new(config.clone());

    for i in 0..3 {
        let pred = dummy_model.predict(&image)?;
        println!("\nPrediction {}:", i + 1);
        println!("  P(Down):    {:.3}", pred.prob_down);
        println!("  P(Neutral): {:.3}", pred.prob_neutral);
        println!("  P(Up):      {:.3}", pred.prob_up);
        println!("  Magnitude:  {:+.4}", pred.magnitude.unwrap_or(0.0));
        println!("  Predicted:  {:?}", match pred.predicted_class() {
            0 => "DOWN",
            1 => "NEUTRAL",
            2 => "UP",
            _ => "UNKNOWN",
        });
    }

    // Test with Rule-Based Model
    println!("\n--- Rule-Based Model ---");
    let rule_model = RuleBasedModel::new(config.clone());
    let pred = rule_model.predict(&image)?;

    println!("\nRule-based Prediction:");
    println!("  P(Down):    {:.3}", pred.prob_down);
    println!("  P(Neutral): {:.3}", pred.prob_neutral);
    println!("  P(Up):      {:.3}", pred.prob_up);
    println!("  Magnitude:  {:+.4}", pred.magnitude.unwrap_or(0.0));
    println!("  Confidence: {:.1}%", pred.confidence() * 100.0);

    // Create Signal Generator
    println!("\n--- Signal Generation ---");
    let signal_gen = SignalGenerator::with_threshold(0.5);

    let signal = signal_gen.generate(
        pred.prob_down,
        pred.prob_neutral,
        pred.prob_up,
        pred.magnitude,
    );

    println!("\nGenerated Signal:");
    println!("  Type:       {}", signal.signal_type);
    println!("  Confidence: {:.1}%", signal.confidence * 100.0);
    if let Some(ret) = signal.expected_return {
        println!("  Expected:   {:+.2}%", ret * 100.0);
    }
    println!("  Actionable: {}", signal.is_actionable());

    // Batch prediction example
    println!("\n--- Batch Predictions ---");

    // Create multiple windows of price data
    let windows: Vec<Vec<f64>> = (0..5)
        .map(|offset| {
            prices[offset * 10..(offset * 10 + 60)].to_vec()
        })
        .collect();

    let images: Vec<_> = windows
        .iter()
        .map(|w| generator.create_multi_transform_stack(w))
        .collect();

    let predictions = rule_model.predict_batch(&images)?;

    println!("\nBatch of {} predictions:", predictions.len());
    for (i, pred) in predictions.iter().enumerate() {
        let class = match pred.predicted_class() {
            0 => "DOWN",
            1 => "NEUTRAL",
            2 => "UP",
            _ => "?",
        };
        println!(
            "  Window {}: {} (confidence: {:.0}%, magnitude: {:+.3}%)",
            i + 1,
            class,
            pred.confidence() * 100.0,
            pred.magnitude.unwrap_or(0.0) * 100.0
        );
    }

    // Convert predictions to signals
    let signals: Vec<Signal> = predictions
        .iter()
        .map(|p| signal_gen.generate(p.prob_down, p.prob_neutral, p.prob_up, p.magnitude))
        .collect();

    println!("\nGenerated Signals:");
    for (i, signal) in signals.iter().enumerate() {
        println!("  Window {}: {} (confidence: {:.0}%)",
            i + 1,
            signal.signal_type,
            signal.confidence * 100.0
        );
    }

    let actionable = signals.iter().filter(|s| s.is_actionable()).count();
    println!("\nActionable signals: {} out of {}", actionable, signals.len());

    println!("\nPrediction example complete!");

    Ok(())
}
