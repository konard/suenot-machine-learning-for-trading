//! Example: MC Dropout Inference
//!
//! This example demonstrates how to use MC Dropout for uncertainty
//! estimation in predictions.

use mc_dropout_trading::prelude::*;
use ndarray::Array1;
use rand::Rng;

fn main() {
    println!("=== MC Dropout Inference Example ===\n");

    // Create model configuration
    let config = MCDropoutConfig::new(20)
        .with_hidden_dims(vec![64, 32, 16])
        .with_dropout_rate(0.2)
        .with_mc_samples(100);

    println!("Model Configuration:");
    println!("  Input dim:     {}", config.input_dim);
    println!("  Hidden dims:   {:?}", config.hidden_dims);
    println!("  Dropout rate:  {}", config.dropout_rate);
    println!("  MC samples:    {}", config.n_mc_samples);

    // Create model
    let model = MCDropoutModel::new(config);
    println!("\nModel created with {} parameters", model.num_parameters());

    // Create sample input
    let mut rng = rand::thread_rng();
    let input: Vec<f64> = (0..20).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let input = Array1::from_vec(input);

    println!("\nRunning MC Dropout inference with 100 forward passes...");

    // Run MC Dropout inference
    let result = model.predict_with_uncertainty(&input, Some(100));

    println!("\nPrediction Results:");
    println!("  Mean:     {:.6}", result.mean);
    println!("  Std:      {:.6}", result.std);
    println!("  Variance: {:.6}", result.variance);
    println!("  Z-score:  {:.2}", result.z_score());

    // Get confidence intervals
    let (lower_90, upper_90) = result.confidence_interval(0.90);
    let (lower_95, upper_95) = result.confidence_interval(0.95);

    println!("\nConfidence Intervals:");
    println!("  90% CI: [{:.6}, {:.6}]", lower_90, upper_90);
    println!("  95% CI: [{:.6}, {:.6}]", lower_95, upper_95);

    // Show sample distribution
    println!("\nSample Distribution:");
    println!("  Min:    {:.6}", result.percentile(0.0));
    println!("  25th:   {:.6}", result.percentile(25.0));
    println!("  Median: {:.6}", result.percentile(50.0));
    println!("  75th:   {:.6}", result.percentile(75.0));
    println!("  Max:    {:.6}", result.percentile(100.0));

    // Demonstrate multiple predictions with varying confidence
    println!("\n=== Multiple Predictions Demo ===\n");

    for i in 0..5 {
        // Create slightly different inputs
        let input: Vec<f64> = (0..20).map(|_| rng.gen_range(-1.0..1.0) * (1.0 + i as f64 * 0.5)).collect();
        let input = Array1::from_vec(input);

        let result = model.predict_with_uncertainty(&input, Some(50));

        let confidence = if result.is_confident(1.5) { "HIGH" } else { "LOW" };

        println!(
            "Sample {}: Mean={:+.4}, Std={:.4}, Z={:.2}, Confidence={}",
            i + 1,
            result.mean,
            result.std,
            result.z_score(),
            confidence
        );
    }

    println!("\n=== MC Dropout inference complete! ===");
}
