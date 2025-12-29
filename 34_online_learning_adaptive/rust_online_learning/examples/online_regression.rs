//! Online Regression Example
//!
//! Demonstrates online linear regression with streaming cryptocurrency data.
//!
//! Run with: cargo run --example online_regression

use online_learning::api::BybitClient;
use online_learning::features::MomentumFeatures;
use online_learning::models::{OnlineLinearRegression, OnlineModel};
use online_learning::streaming::StreamSimulator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Online Linear Regression Demo ===\n");

    // Fetch data
    let client = BybitClient::new();
    let symbol = "BTCUSDT";

    println!("Fetching {} data from Bybit...", symbol);
    let candles = client.get_klines(symbol, "1h", 500).await?;
    println!("Fetched {} candles\n", candles.len());

    // Setup
    let periods = vec![12, 24, 48, 96];
    let learning_rate = 0.01;

    println!("Configuration:");
    println!("  Momentum periods: {:?}", periods);
    println!("  Learning rate: {}", learning_rate);
    println!();

    // Create stream simulator
    let mut simulator = StreamSimulator::new(candles, periods.clone());

    // Create online model
    let mut model = OnlineLinearRegression::new(periods.len(), learning_rate);

    // Metrics tracking
    let mut predictions = Vec::new();
    let mut actuals = Vec::new();
    let mut cumulative_error = 0.0;

    println!("Starting online learning...\n");

    // Stream and learn
    let mut count = 0;
    while let Some(obs) = simulator.next() {
        // Predict before learning
        let prediction = model.predict(&obs.features);

        // Record
        predictions.push(prediction);
        actuals.push(obs.target);

        let error = (prediction - obs.target).abs();
        cumulative_error += error;

        // Learn from observation
        model.learn(&obs.features, obs.target);

        count += 1;

        // Progress update every 50 samples
        if count % 50 == 0 {
            let avg_error = cumulative_error / count as f64;
            println!(
                "  Processed {} samples, Avg MAE: {:.6}, Current prediction: {:.6}",
                count, avg_error, prediction
            );
        }
    }

    println!("\n=== Results ===\n");

    // Calculate metrics
    let n = predictions.len() as f64;

    // MSE
    let mse: f64 = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>()
        / n;

    // MAE
    let mae: f64 = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>()
        / n;

    // Direction accuracy
    let direction_correct: usize = predictions
        .iter()
        .zip(actuals.iter())
        .filter(|(p, a)| p.signum() == a.signum())
        .count();

    let direction_accuracy = direction_correct as f64 / n;

    // Correlation
    let pred_mean: f64 = predictions.iter().sum::<f64>() / n;
    let actual_mean: f64 = actuals.iter().sum::<f64>() / n;

    let covariance: f64 = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - pred_mean) * (a - actual_mean))
        .sum::<f64>()
        / n;

    let pred_std = (predictions.iter().map(|p| (p - pred_mean).powi(2)).sum::<f64>() / n).sqrt();
    let actual_std = (actuals.iter().map(|a| (a - actual_mean).powi(2)).sum::<f64>() / n).sqrt();

    let correlation = if pred_std > 0.0 && actual_std > 0.0 {
        covariance / (pred_std * actual_std)
    } else {
        0.0
    };

    println!("Prediction Metrics:");
    println!("  Total samples: {}", predictions.len());
    println!("  MSE: {:.8}", mse);
    println!("  MAE: {:.8}", mae);
    println!("  Direction Accuracy: {:.2}%", direction_accuracy * 100.0);
    println!("  Correlation: {:.4}", correlation);
    println!();

    // Model weights
    println!("Final Model Weights:");
    let weights = model.get_params();
    let feature_names = ["mom_12h", "mom_24h", "mom_48h", "mom_96h", "bias"];

    for (name, weight) in feature_names.iter().zip(weights.iter()) {
        println!("  {}: {:.6}", name, weight);
    }

    // Trading simulation
    println!("\n=== Simple Trading Simulation ===\n");

    let threshold = 0.001;
    let mut total_pnl = 0.0;
    let mut trades = 0;
    let mut wins = 0;

    for (pred, actual) in predictions.iter().zip(actuals.iter()) {
        let signal = if *pred > threshold {
            1.0
        } else if *pred < -threshold {
            -1.0
        } else {
            0.0
        };

        if signal != 0.0 {
            let pnl = signal * actual;
            total_pnl += pnl;
            trades += 1;
            if pnl > 0.0 {
                wins += 1;
            }
        }
    }

    let win_rate = if trades > 0 { wins as f64 / trades as f64 } else { 0.0 };

    println!("Trading Results:");
    println!("  Total trades: {}", trades);
    println!("  Win rate: {:.2}%", win_rate * 100.0);
    println!("  Total PnL: {:.4}%", total_pnl * 100.0);
    println!("  Avg PnL per trade: {:.6}%", if trades > 0 { total_pnl / trades as f64 * 100.0 } else { 0.0 });

    Ok(())
}
