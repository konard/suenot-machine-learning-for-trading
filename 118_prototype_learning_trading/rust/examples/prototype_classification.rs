//! Example: Market state classification using prototype learning
//!
//! This example demonstrates how to:
//! 1. Fetch market data from Bybit
//! 2. Compute technical features
//! 3. Classify market state using prototype network
//! 4. Get interpretable explanations

use prototype_learning::{
    BybitClient, FeatureEngineering, Interval, MarketState, PrototypeNetwork,
};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("Prototype Learning - Market Classification Example");
    println!("==================================================\n");

    // Create Bybit client and fetch data
    println!("1. Fetching market data...");
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(200), None, None)?;
    println!("   Fetched {} klines\n", klines.len());

    // Compute features
    println!("2. Computing technical features...");
    let fe = FeatureEngineering::new(60);
    let features = fe.compute_features(&klines)?;
    println!("   Feature matrix shape: {:?}", features.dim());
    println!("   Features: {:?}\n", fe.get_feature_names());

    // Create prototype network
    println!("3. Creating prototype network...");
    let n_features = fe.get_feature_names().len();
    let model = PrototypeNetwork::new(
        n_features, // 13 features
        128,        // latent dimension
        5,          // 5 market states
        2,          // 2 prototypes per class
    );
    println!("   Number of prototypes: {}\n", model.num_prototypes());

    // Prepare input for classification
    // We'll use the last 60 periods of data
    println!("4. Classifying current market state...");

    let lookback = fe.lookback();
    let start_idx = features.dim().0 - lookback;

    // Flatten features for the lookback window
    let mut input_features = Vec::with_capacity(n_features * lookback);
    for t in start_idx..features.dim().0 {
        for f in 0..n_features {
            let val = features[[t, f]];
            // Replace NaN with 0 for simplicity
            input_features.push(if val.is_nan() { 0.0 } else { val });
        }
    }
    let input = Array1::from_vec(input_features);

    // Get prediction and explanation
    let explanation = model.explain(&input)?;
    println!("\n{}", explanation.explain());

    // Display class probabilities
    println!("\nClass Similarities:");
    for (state, sim) in &explanation.class_similarities {
        let bar_len = ((sim.abs() * 50.0) as usize).min(50);
        let bar = if *sim >= 0.0 {
            format!("{:>50}", "#".repeat(bar_len))
        } else {
            format!("{:<50}", "-".repeat(bar_len))
        };
        println!("  {:<15} [{:>50}] {:.4}", state.name(), bar, sim);
    }

    // Demonstrate batch prediction
    println!("\n5. Batch prediction on recent data...");
    println!("\n{:<10} {:<15} {:>10}", "Period", "State", "Confidence");
    println!("{}", "-".repeat(40));

    // Predict for the last 10 available periods
    let n_predictions = 10;
    let available_periods = features.dim().0.saturating_sub(lookback);

    for i in (available_periods.saturating_sub(n_predictions))..available_periods {
        let mut input_vec = Vec::with_capacity(n_features * lookback);
        for t in i..(i + lookback).min(features.dim().0) {
            for f in 0..n_features {
                let val = features[[t, f]];
                input_vec.push(if val.is_nan() { 0.0 } else { val });
            }
        }

        // Pad if necessary
        while input_vec.len() < n_features * lookback {
            input_vec.push(0.0);
        }

        let input = Array1::from_vec(input_vec);
        if let Ok((state, confidence, _)) = model.predict_single(&input) {
            println!(
                "{:<10} {:<15} {:>10.2}%",
                format!("t-{}", available_periods - i - 1),
                state.name(),
                confidence * 100.0
            );
        }
    }

    // Trading signal generation
    println!("\n6. Generating trading signal...");
    let confidence_threshold = 0.25; // Low threshold for demo with random model

    let action = if explanation.confidence >= confidence_threshold {
        explanation.predicted_state.trading_action()
    } else {
        prototype_learning::models::prototype::TradingAction::Hold
    };

    println!(
        "   Current state: {} (confidence: {:.2}%)",
        explanation.predicted_state.name(),
        explanation.confidence * 100.0
    );
    println!("   Confidence threshold: {:.2}%", confidence_threshold * 100.0);
    println!(
        "   Trading action: {:?}",
        action
    );

    // Summary
    println!("\n7. Summary");
    println!("   --------");
    println!("   - Prototype learning provides interpretable market classification");
    println!("   - Each prediction explains which prototype pattern was most similar");
    println!("   - Confidence scores help filter low-quality signals");
    println!("   - The model classifies markets into 5 states:");
    for i in 0..5 {
        if let Some(state) = MarketState::from_int(i) {
            println!("     * {}: {:?}", state.name(), state.trading_action());
        }
    }

    println!("\nNote: This example uses a randomly initialized model.");
    println!("For production use, load pre-trained weights from Python training.");

    Ok(())
}
