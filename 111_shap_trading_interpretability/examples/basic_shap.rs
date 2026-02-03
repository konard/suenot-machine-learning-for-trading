//! Basic SHAP example demonstrating feature importance explanation.
//!
//! This example shows how to:
//! 1. Create a simple linear model
//! 2. Initialize a SHAP explainer
//! 3. Compute SHAP values for a prediction
//! 4. Identify top contributing features

use shap_trading::model::shap::{LinearModel, ShapExplainer, ShapValues};

fn main() {
    println!("=== SHAP Trading Interpretability Demo ===\n");

    // Define feature names
    let feature_names = vec![
        "returns_1".to_string(),
        "returns_5".to_string(),
        "returns_10".to_string(),
        "volatility".to_string(),
        "rsi".to_string(),
        "volume_ratio".to_string(),
        "price_ma_ratio".to_string(),
    ];

    // Create a simple linear model
    // Positive weights mean feature contributes to bullish signal
    let weights = vec![
        0.5,  // returns_1: Recent positive returns → bullish
        0.3,  // returns_5: 5-period momentum
        0.2,  // returns_10: 10-period momentum
        -0.4, // volatility: High vol → reduce position
        0.1,  // rsi: Normalized RSI
        0.15, // volume_ratio: High volume confirms signal
        0.25, // price_ma_ratio: Price above MA → bullish
    ];

    let model = LinearModel::new(weights.clone(), 0.0);

    // Generate some background data for the explainer
    let background_data: Vec<Vec<f64>> = (0..50)
        .map(|i| {
            vec![
                0.01 * (i as f64 % 10.0 - 5.0),   // returns vary around 0
                0.02 * (i as f64 % 10.0 - 5.0),   // 5-period returns
                0.03 * (i as f64 % 10.0 - 5.0),   // 10-period returns
                0.02 + 0.005 * (i as f64 % 8.0),  // volatility
                0.5 + 0.1 * (i as f64 % 10.0 - 5.0), // normalized RSI
                1.0 + 0.1 * (i as f64 % 10.0 - 5.0), // volume ratio
                1.0 + 0.02 * (i as f64 % 10.0 - 5.0), // price/MA ratio
            ]
        })
        .collect();

    // Compute base value (expected prediction on background)
    let base_value: f64 = background_data
        .iter()
        .map(|x| model.predict(x))
        .sum::<f64>()
        / background_data.len() as f64;

    println!("Base value (expected prediction): {:.4}", base_value);

    // Create SHAP explainer
    let explainer = ShapExplainer::new(
        feature_names.clone(),
        base_value,
        background_data,
        100, // number of samples for KernelSHAP
    );

    // Example 1: Bullish signal (positive returns, low volatility)
    println!("\n--- Example 1: Bullish Signal ---");
    let bullish_features = vec![
        0.02,  // Strong positive recent return
        0.03,  // Positive 5-period momentum
        0.02,  // Positive 10-period momentum
        0.015, // Low volatility
        0.65,  // RSI above average
        1.3,   // High volume
        1.03,  // Price above 20-MA
    ];

    let explanation = explainer.explain(|x| model.predict(x), &bullish_features);
    println!("Input features: {:?}", bullish_features);
    println!("Prediction: {:.4}", explanation.prediction);
    println!("\nSHAP values (feature contributions):");
    for (name, value) in explanation.shap_values.top_contributors(7) {
        let direction = if value > 0.0 { "+" } else { "" };
        println!("  {}: {}{:.4}", name, direction, value);
    }

    // Example 2: Bearish signal (negative returns, high volatility)
    println!("\n--- Example 2: Bearish Signal ---");
    let bearish_features = vec![
        -0.03, // Strong negative recent return
        -0.02, // Negative 5-period momentum
        -0.01, // Negative 10-period momentum
        0.04,  // High volatility
        0.35,  // RSI below average (oversold)
        0.8,   // Low volume
        0.97,  // Price below 20-MA
    ];

    let explanation = explainer.explain(|x| model.predict(x), &bearish_features);
    println!("Input features: {:?}", bearish_features);
    println!("Prediction: {:.4}", explanation.prediction);
    println!("\nSHAP values (feature contributions):");
    for (name, value) in explanation.shap_values.top_contributors(7) {
        let direction = if value > 0.0 { "+" } else { "" };
        println!("  {}: {}{:.4}", name, direction, value);
    }

    // Verify efficiency property
    println!("\n--- SHAP Efficiency Property ---");
    let sum: f64 = explanation.shap_values.values.iter().sum();
    println!(
        "Sum of SHAP values: {:.4}",
        sum
    );
    println!(
        "Prediction - Base: {:.4}",
        explanation.prediction - base_value
    );
    println!(
        "Efficiency verified: {}",
        explanation.verify_efficiency(0.1)
    );

    // Example 3: Neutral signal (mixed features)
    println!("\n--- Example 3: Neutral Signal ---");
    let neutral_features = vec![
        0.001, // Almost no recent return
        0.0,   // No 5-period momentum
        0.0,   // No 10-period momentum
        0.025, // Average volatility
        0.5,   // Neutral RSI
        1.0,   // Average volume
        1.0,   // Price at 20-MA
    ];

    let explanation = explainer.explain(|x| model.predict(x), &neutral_features);
    println!("Input features: {:?}", neutral_features);
    println!("Prediction: {:.4}", explanation.prediction);
    println!("\nTop contributing features:");
    for (name, value) in explanation.shap_values.top_contributors(3) {
        let direction = if value > 0.0 { "+" } else { "" };
        println!("  {}: {}{:.4}", name, direction, value);
    }

    println!("\n=== Demo Complete ===");
}
