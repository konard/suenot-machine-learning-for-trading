//! Example: Train MambaTS model (demonstration)
//!
//! Note: For actual training, use the Python implementation with PyTorch.
//! This example demonstrates the workflow and data preparation.
//!
//! Run with: cargo run --example train_model

use chrono::{Duration, Utc};
use indicatif::{ProgressBar, ProgressStyle};

use mambats_time_series::{
    BybitClient, Interval,
    prepare_features,
    data::processor::{DataProcessor, TimeSeriesDataset, remove_nan_rows},
};
use mambats_time_series::model::mambats::{MambaTS, MambaTSConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("MambaTS - Training Demonstration");
    println!("=================================\n");

    println!("Note: For actual training, use the Python implementation.");
    println!("      This example demonstrates data preparation and workflow.\n");

    // Fetch data
    println!("Step 1: Fetching Data");
    println!("{:-<50}", "");

    let client = BybitClient::new();
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(60);

    println!("Fetching 60 days of BTCUSDT hourly data...");

    let candles = match client.get_klines("BTCUSDT", Interval::OneHour, start_time, end_time).await {
        Ok(c) => {
            println!("Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            println!("Could not fetch data: {}", e);
            println!("Continuing with synthetic data demonstration...\n");
            return demonstrate_with_synthetic_data();
        }
    };

    // Feature engineering
    println!("\nStep 2: Feature Engineering");
    println!("{:-<50}", "");

    let raw_features = prepare_features(&candles);
    println!("Raw features: {} x {}", raw_features.nrows(), raw_features.ncols());

    // Remove NaN rows
    let clean_features = remove_nan_rows(&raw_features);
    println!("After removing NaN: {} x {}", clean_features.nrows(), clean_features.ncols());

    // Normalize
    let mut processor = DataProcessor::new();
    let normalized = processor.fit_transform(&clean_features);
    println!("Normalized features ready");

    // Create dataset
    println!("\nStep 3: Dataset Creation");
    println!("{:-<50}", "");

    let lookback = 168;  // 1 week
    let forecast_horizon = 24;  // 1 day
    let stride = 1;

    let dataset = TimeSeriesDataset::new(
        normalized.clone(),
        lookback,
        forecast_horizon,
        stride,
    );

    println!("Lookback window: {} hours", lookback);
    println!("Forecast horizon: {} hours", forecast_horizon);
    println!("Total samples: {}", dataset.len());

    // Split dataset
    let (train_set, val_set, test_set) = dataset.train_val_test_split(0.7, 0.15);
    println!("Train samples: {}", train_set.len());
    println!("Val samples: {}", val_set.len());
    println!("Test samples: {}", test_set.len());

    // Create model
    println!("\nStep 4: Model Creation");
    println!("{:-<50}", "");

    let config = MambaTSConfig {
        n_variables: normalized.ncols(),
        d_model: 128,
        n_layers: 4,
        patch_size: 16,
        forecast_horizon,
        d_state: 16,
    };

    let model = MambaTS::new(config.clone());

    println!("Model configuration:");
    println!("  Input variables: {}", config.n_variables);
    println!("  Model dimension: {}", config.d_model);
    println!("  Layers: {}", config.n_layers);
    println!("  Patch size: {}", config.patch_size);

    // Demonstrate forward pass
    println!("\nStep 5: Forward Pass Test");
    println!("{:-<50}", "");

    if let Some((x, y)) = train_set.get(0) {
        println!("Input shape: {} x {}", x.nrows(), x.ncols());
        println!("Target shape: {} x {}", y.nrows(), y.ncols());

        match model.predict(&x) {
            Ok(pred) => {
                println!("Prediction shape: {} x {}", pred.nrows(), pred.ncols());
                println!("Forward pass successful!");
            }
            Err(e) => {
                println!("Forward pass error: {}", e);
            }
        }
    }

    // Simulated training loop
    println!("\nStep 6: Training Loop (Simulated)");
    println!("{:-<50}", "");

    let epochs = 5;
    let _batch_size = 32;

    let pb = ProgressBar::new(epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} epochs")
            .unwrap()
            .progress_chars("#>-"),
    );

    for epoch in 0..epochs {
        // In a real implementation, we would:
        // 1. Iterate over batches
        // 2. Compute forward pass
        // 3. Compute loss
        // 4. Backpropagate
        // 5. Update weights

        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Simulated losses
        let train_loss = 0.5 * (0.9_f64).powi(epoch as i32) + 0.01;
        let val_loss = 0.6 * (0.85_f64).powi(epoch as i32) + 0.02;

        pb.inc(1);
        pb.println(format!(
            "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}",
            epoch + 1, epochs, train_loss, val_loss
        ));
    }

    pb.finish_with_message("Training complete!");

    // Save model
    println!("\nStep 7: Model Saving");
    println!("{:-<50}", "");

    let model_path = "models/mambats_demo.json";
    std::fs::create_dir_all("models").ok();

    match model.save(model_path) {
        Ok(_) => println!("Model saved to {}", model_path),
        Err(e) => println!("Could not save model: {}", e),
    }

    // Final summary
    println!("\n");
    println!("Training Demonstration Complete!");
    println!("================================");
    println!();
    println!("For actual training with GPU acceleration:");
    println!("  cd python && python train.py --data bybit --symbol BTCUSDT");
    println!();
    println!("The trained PyTorch model can be exported and loaded in Rust");
    println!("for high-performance inference in production systems.");

    Ok(())
}

fn demonstrate_with_synthetic_data() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::Array2;
    use rand::Rng;

    println!("Generating synthetic data...\n");

    let mut rng = rand::thread_rng();
    let n_samples = 1000;
    let n_features = 17;

    let features = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        let trend = (i as f64 * 0.01).sin();
        let noise: f64 = rng.gen::<f64>() * 0.2;
        trend + noise + j as f64 * 0.05
    });

    println!("Synthetic features: {} x {}", features.nrows(), features.ncols());

    // Create dataset
    let lookback = 50;
    let forecast_horizon = 10;

    let dataset = TimeSeriesDataset::new(features, lookback, forecast_horizon, 1);
    println!("Dataset samples: {}", dataset.len());

    // Create and test model
    let config = MambaTSConfig {
        n_variables: n_features,
        d_model: 64,
        n_layers: 2,
        patch_size: 10,
        forecast_horizon,
        d_state: 8,
    };

    let model = MambaTS::new(config);

    if let Some((x, _y)) = dataset.get(0) {
        match model.predict(&x) {
            Ok(_) => println!("Model forward pass successful!"),
            Err(e) => println!("Error: {}", e),
        }
    }

    println!("\nUse Python implementation for actual training.");
    Ok(())
}
