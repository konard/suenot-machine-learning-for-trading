//! Example: Generating live trading signals
//!
//! This example demonstrates how to use the Deep Ensemble model
//! for generating live trading signals with uncertainty-aware filtering.

use deep_ensembles_trading::api::client::BybitClient;
use deep_ensembles_trading::ensemble::config::EnsembleConfig;
use deep_ensembles_trading::ensemble::model::DeepEnsemble;
use deep_ensembles_trading::features::engine::FeatureEngine;
use deep_ensembles_trading::strategy::signal::SignalType;
use deep_ensembles_trading::strategy::uncertainty::{RegimeDetector, UncertaintyStrategy};
use ndarray::{Array1, Array2};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Live Trading Signals Example ===\n");

    // Step 1: Fetch historical data
    println!("1. Fetching historical data from Bybit...");
    let client = BybitClient::new();
    let symbol = "BTCUSDT";

    let klines = match client.get_klines(symbol, "60", 500).await {
        Ok(k) => k,
        Err(e) => {
            println!("   Warning: Could not fetch live data: {}", e);
            println!("   Using synthetic data instead...\n");
            return run_with_synthetic_data();
        }
    };

    println!("   Fetched {} klines for {}", klines.len(), symbol);

    // Step 2: Prepare features
    println!("\n2. Preparing features...");
    let features = FeatureEngine::compute_features(&klines);
    let targets = FeatureEngine::compute_targets(&klines, 5);

    println!("   Features shape: {}x{}", features.nrows(), features.ncols());

    // Standardize features
    let (features_std, means, stds) = FeatureEngine::standardize(&features);

    // Step 3: Split data for training
    let train_size = (features_std.nrows() as f64 * 0.8) as usize;
    let x_train = features_std.slice(ndarray::s![..train_size, ..]).to_owned();
    let y_train = targets.slice(ndarray::s![..train_size]).to_owned();

    // Step 4: Train ensemble
    println!("\n3. Training Deep Ensemble...");
    let config = EnsembleConfig {
        num_models: 5,
        input_dim: features_std.ncols(),
        hidden_dims: vec![64, 32],
        learning_rate: 0.005,
        batch_size: 32,
        max_epochs: 50,
        early_stopping_patience: 10,
        ..EnsembleConfig::default()
    };

    let mut ensemble = DeepEnsemble::new(config);
    ensemble.train(&x_train, &y_train, None, None, false);
    println!("   Training complete!");

    // Step 5: Create strategy and regime detector
    println!("\n4. Creating trading strategy...");
    let strategy = UncertaintyStrategy::new(0.6, 0.5, 1.2, 0.1);
    let mut regime_detector = RegimeDetector::default();

    // Step 6: Generate signal for latest data
    println!("\n5. Generating current trading signal...");

    // Get latest features
    let latest_idx = features_std.nrows() - 1;
    let latest_features = features_std
        .slice(ndarray::s![latest_idx..latest_idx + 1, ..])
        .to_owned();

    // Get prediction
    let predictions = ensemble.predict(&latest_features);

    // Update regime detector (simulate with historical data)
    for i in train_size..features_std.nrows() {
        let feat = features_std.slice(ndarray::s![i..i + 1, ..]).to_owned();
        let pred = ensemble.predict(&feat);
        regime_detector.update(&pred);
    }

    let regime = regime_detector.detect();

    // Generate signal
    let signals = strategy.generate_signals(&predictions, &[symbol.to_string()]);
    let signal = &signals[0];

    // Step 7: Display results
    println!("\n{}", "=".repeat(60));
    println!("CURRENT TRADING SIGNAL FOR {}", symbol);
    println!("{}", "=".repeat(60));

    println!("\nMarket Regime: {:?}", regime);

    println!("\nPrediction:");
    println!("   Mean Return:            {:.4}%", predictions.mean[0] * 100.0);
    println!("   Total Uncertainty:      {:.4}", predictions.total_std[0]);
    println!("   Epistemic Uncertainty:  {:.4}", predictions.epistemic_std[0]);
    println!("   Aleatoric Uncertainty:  {:.4}", predictions.aleatoric_std[0]);

    let epistemic_ratio = predictions.epistemic_std[0] / predictions.total_std[0];
    println!("   Epistemic Ratio:        {:.2}%", epistemic_ratio * 100.0);

    println!("\nSignal:");
    println!("   Type:       {:?}", signal.signal_type);
    println!("   Confidence: {:.2}%", signal.confidence * 100.0);

    match signal.signal_type {
        SignalType::Long => {
            let position = strategy.calculate_position_size(signal, 100000.0);
            println!("\n   RECOMMENDATION: BUY");
            println!("   Suggested position: ${:.2}", position);
        }
        SignalType::Short => {
            let position = strategy.calculate_position_size(signal, 100000.0);
            println!("\n   RECOMMENDATION: SELL/SHORT");
            println!("   Suggested position: ${:.2}", position.abs());
        }
        SignalType::Hold => {
            println!("\n   RECOMMENDATION: HOLD / NO TRADE");
            if epistemic_ratio > 0.5 {
                println!("   Reason: High model uncertainty - not confident in prediction");
            } else {
                println!("   Reason: Signal strength too weak");
            }
        }
    }

    // Show individual model predictions
    println!("\nIndividual Model Predictions:");
    for i in 0..predictions.num_models() {
        println!(
            "   Model {}: {:.4}% +/- {:.4}",
            i + 1,
            predictions.individual_means[i][0] * 100.0,
            predictions.individual_stds[i][0]
        );
    }

    println!("\n{}", "=".repeat(60));
    println!("DISCLAIMER: This is for educational purposes only.");
    println!("Do not trade based on this signal without proper risk management.");
    println!("{}", "=".repeat(60));

    Ok(())
}

/// Run with synthetic data when API is unavailable
fn run_with_synthetic_data() -> anyhow::Result<()> {
    println!("Running with synthetic data...\n");

    // Generate synthetic features and targets
    let n = 500;
    let d = 10;

    let features = Array2::from_shape_fn((n, d), |_| rand::random::<f64>() * 2.0 - 1.0);
    let targets = Array1::from_shape_fn(n, |i| {
        let sum: f64 = features.row(i).iter().take(3).sum();
        sum * 0.1 + rand::random::<f64>() * 0.05 - 0.025
    });

    // Train ensemble
    println!("Training ensemble on synthetic data...");
    let config = EnsembleConfig {
        num_models: 5,
        input_dim: d,
        hidden_dims: vec![32, 16],
        max_epochs: 30,
        ..EnsembleConfig::default()
    };

    let mut ensemble = DeepEnsemble::new(config);
    ensemble.train(&features, &targets, None, None, false);

    // Generate signal
    let strategy = UncertaintyStrategy::default();
    let latest = features.slice(ndarray::s![-1.., ..]).to_owned();
    let predictions = ensemble.predict(&latest);
    let signals = strategy.generate_signals(&predictions, &["SYNTHETIC".to_string()]);

    println!("\nSynthetic Signal:");
    println!("   Type: {:?}", signals[0].signal_type);
    println!("   Confidence: {:.2}%", signals[0].confidence * 100.0);
    println!("   Predicted Return: {:.4}%", signals[0].predicted_return * 100.0);

    Ok(())
}
