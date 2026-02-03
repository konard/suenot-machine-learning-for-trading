//! Example: Make predictions with MambaTS
//!
//! This example demonstrates how to use the MambaTS model
//! to make price predictions.
//!
//! Run with: cargo run --example predict

use chrono::{Duration, Utc};
use mambats_time_series::{
    BybitClient, Interval, MambaTS,
    prepare_features,
    data::features::normalize_features,
};
use mambats_time_series::model::mambats::MambaTSConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("MambaTS - Prediction Example");
    println!("============================\n");

    // Create model with default config
    let config = MambaTSConfig {
        n_variables: 17,  // Number of features from prepare_features
        d_model: 64,
        n_layers: 2,
        patch_size: 8,
        forecast_horizon: 6,  // Predict 6 hours ahead
        d_state: 8,
    };

    println!("Model Configuration:");
    println!("  Variables:        {}", config.n_variables);
    println!("  Model Dimension:  {}", config.d_model);
    println!("  Layers:           {}", config.n_layers);
    println!("  Patch Size:       {}", config.patch_size);
    println!("  Forecast Horizon: {} hours", config.forecast_horizon);
    println!();

    // Create model (with random weights for demonstration)
    let model = MambaTS::new(config.clone());

    // Try to fetch real data
    println!("Fetching market data from Bybit...");
    let client = BybitClient::new();
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(3);

    match client.get_klines("BTCUSDT", Interval::OneHour, start_time, end_time).await {
        Ok(candles) if candles.len() >= 50 => {
            println!("Fetched {} candles\n", candles.len());

            // Prepare features
            let features = prepare_features(&candles);
            println!("Feature matrix: {} x {}", features.nrows(), features.ncols());

            // Normalize features
            let (normalized, _stats) = normalize_features(&features);

            // Remove NaN rows and take last lookback window
            let lookback = 40;
            let valid_start = normalized.nrows().saturating_sub(lookback);
            let input = normalized.slice(ndarray::s![valid_start.., ..]).to_owned();

            println!("Input shape: {} x {}", input.nrows(), input.ncols());

            // Make prediction
            match model.predict(&input) {
                Ok(prediction) => {
                    println!("\nPrediction for next {} hours:", config.forecast_horizon);
                    println!("{:-<50}", "");

                    // Get current price for context
                    let current_price = candles.last().map(|c| c.close).unwrap_or(0.0);
                    println!("Current Price: ${:.2}", current_price);
                    println!();

                    // Show predictions (note: with random weights, these are not meaningful)
                    println!("Note: Using random weights for demonstration.");
                    println!("      Train the model for actual predictions.\n");

                    for h in 0..config.forecast_horizon {
                        let pred_value = prediction[[h, 0]];
                        // Interpret as normalized return
                        let direction = if pred_value > 0.0 { "UP" } else { "DOWN" };
                        let confidence = pred_value.abs().min(1.0) * 100.0;
                        println!(
                            "  Hour +{}: {} (confidence: {:.1}%)",
                            h + 1,
                            direction,
                            confidence
                        );
                    }
                }
                Err(e) => {
                    println!("Prediction error: {}", e);
                }
            }
        }
        Ok(candles) => {
            println!("Not enough data: only {} candles", candles.len());
            demonstrate_with_synthetic_data(&model, &config);
        }
        Err(e) => {
            println!("Could not fetch data: {}", e);
            println!("Demonstrating with synthetic data...\n");
            demonstrate_with_synthetic_data(&model, &config);
        }
    }

    Ok(())
}

fn demonstrate_with_synthetic_data(model: &MambaTS, config: &MambaTSConfig) {
    use ndarray::Array2;
    use rand::Rng;

    println!("Generating synthetic data for demonstration...\n");

    let mut rng = rand::thread_rng();
    let n_samples = 50;

    // Generate synthetic features
    let features = Array2::from_shape_fn((n_samples, config.n_variables), |(i, j)| {
        let base = (i as f64 * 0.1).sin();
        let noise: f64 = rng.gen::<f64>() * 0.1;
        base + noise + j as f64 * 0.01
    });

    println!("Synthetic feature matrix: {} x {}", features.nrows(), features.ncols());

    match model.predict(&features) {
        Ok(prediction) => {
            println!("\nPrediction (synthetic data):");
            println!("{:-<50}", "");
            println!("Note: Random weights + synthetic data = random predictions\n");

            for h in 0..config.forecast_horizon {
                let pred_value = prediction[[h, 0]];
                println!("  Hour +{}: {:.4}", h + 1, pred_value);
            }
        }
        Err(e) => {
            println!("Prediction error: {}", e);
        }
    }
}
