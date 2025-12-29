//! Train a neural network on cryptocurrency data
//!
//! Usage: cargo run --bin train -- --data BTCUSDT_60.csv --epochs 100

use anyhow::Result;
use rust_nn_crypto::{
    data::OHLCVSeries,
    features::FeatureEngine,
    nn::{NeuralNetwork, activation::ActivationType, optimizer::Adam, NetworkConfig, LossFunction},
};
use std::env;

fn main() -> Result<()> {
    env_logger::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let mut data_path = "BTCUSDT_60.csv".to_string();
    let mut model_path = "model.json".to_string();
    let mut epochs = 100usize;
    let mut batch_size = 32usize;
    let mut learning_rate = 0.001f64;
    let mut hidden_layers = vec![64, 32, 16];
    let mut target_horizon = 1usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" | "-d" => {
                data_path = args.get(i + 1).cloned().unwrap_or(data_path);
                i += 2;
            }
            "--model" | "-m" => {
                model_path = args.get(i + 1).cloned().unwrap_or(model_path);
                i += 2;
            }
            "--epochs" | "-e" => {
                epochs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(epochs);
                i += 2;
            }
            "--batch" | "-b" => {
                batch_size = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(batch_size);
                i += 2;
            }
            "--lr" => {
                learning_rate = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(learning_rate);
                i += 2;
            }
            "--horizon" | "-h" => {
                target_horizon = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(target_horizon);
                i += 2;
            }
            "--help" => {
                print_help();
                return Ok(());
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("            Neural Network Training for Crypto Trading");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Load data
    println!("Loading data from {}...", data_path);
    let series = OHLCVSeries::load_csv(&data_path, "UNKNOWN".to_string(), "unknown".to_string())?;
    println!("Loaded {} candles", series.len());

    // Extract features
    println!("\nExtracting features...");
    let mut feature_engine = FeatureEngine::default_config();
    let (features, targets, valid_indices) = feature_engine.extract_features(&series, target_horizon);

    println!("Generated {} samples with {} features", features.nrows(), features.ncols());
    println!("Feature names: {:?}", feature_engine.get_feature_names());

    // Normalize features
    use rust_nn_crypto::data::{StandardNormalizer, Normalizer};
    let mut normalizer = StandardNormalizer::new();
    let features_normalized = normalizer.fit_transform(&features);

    // Train/test split
    let split_idx = (valid_indices.len() as f64 * 0.8) as usize;
    let train_features = features_normalized.slice(ndarray::s![..split_idx, ..]).to_owned();
    let test_features = features_normalized.slice(ndarray::s![split_idx.., ..]).to_owned();

    let train_targets = targets.slice(ndarray::s![..split_idx]).to_owned();
    let test_targets = targets.slice(ndarray::s![split_idx..]).to_owned();

    // Reshape targets
    let train_targets_2d = train_targets.into_shape((split_idx, 1))?;
    let test_targets_2d = test_targets.into_shape((valid_indices.len() - split_idx, 1))?;

    println!("\nDataset split:");
    println!("  Training samples: {}", train_features.nrows());
    println!("  Test samples: {}", test_features.nrows());

    // Create model
    println!("\nCreating neural network...");
    let input_size = features.ncols();

    let mut config = NetworkConfig::new(input_size);
    for &size in &hidden_layers {
        config = config.add_layer_with_dropout(size, ActivationType::LeakyReLU, 0.2);
    }
    config = config
        .output_layer(1, ActivationType::Linear)
        .with_loss(LossFunction::MSE);

    let mut model = NeuralNetwork::from_config(config);
    model.set_optimizer(Box::new(Adam::new(learning_rate)));

    model.summary();

    // Train
    println!("\nTraining for {} epochs with batch size {}...", epochs, batch_size);
    println!("─────────────────────────────────────────────────────────────────");

    let losses = model.train(&train_features, &train_targets_2d, epochs, batch_size, true);

    println!("─────────────────────────────────────────────────────────────────");

    // Evaluate
    println!("\nEvaluating on test set...");
    let train_loss = model.evaluate(&train_features, &train_targets_2d);
    let test_loss = model.evaluate(&test_features, &test_targets_2d);

    println!("  Training Loss (MSE): {:.6}", train_loss);
    println!("  Test Loss (MSE): {:.6}", test_loss);

    // Make predictions and calculate metrics
    let predictions = model.predict(&test_features);
    let pred_vec: Vec<f64> = predictions.column(0).to_vec();
    let actual_vec: Vec<f64> = test_targets_2d.column(0).to_vec();

    // Calculate direction accuracy
    let mut correct_direction = 0;
    for (pred, actual) in pred_vec.iter().zip(actual_vec.iter()) {
        if (pred > &0.0 && actual > &0.0) || (pred < &0.0 && actual < &0.0) {
            correct_direction += 1;
        }
    }
    let direction_accuracy = correct_direction as f64 / pred_vec.len() as f64 * 100.0;

    println!("  Direction Accuracy: {:.2}%", direction_accuracy);

    // Calculate correlation
    let mean_pred: f64 = pred_vec.iter().sum::<f64>() / pred_vec.len() as f64;
    let mean_actual: f64 = actual_vec.iter().sum::<f64>() / actual_vec.len() as f64;

    let mut cov = 0.0;
    let mut var_pred = 0.0;
    let mut var_actual = 0.0;

    for (p, a) in pred_vec.iter().zip(actual_vec.iter()) {
        cov += (p - mean_pred) * (a - mean_actual);
        var_pred += (p - mean_pred).powi(2);
        var_actual += (a - mean_actual).powi(2);
    }

    let correlation = if var_pred > 0.0 && var_actual > 0.0 {
        cov / (var_pred.sqrt() * var_actual.sqrt())
    } else {
        0.0
    };

    println!("  Correlation: {:.4}", correlation);

    // Save model
    println!("\nSaving model to {}...", model_path);
    model.save(&model_path)?;
    println!("Model saved successfully!");

    // Print training summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                      Training Complete!");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Initial Loss: {:.6}", losses.first().unwrap_or(&0.0));
    println!("  Final Loss: {:.6}", losses.last().unwrap_or(&0.0));
    println!("  Improvement: {:.2}%",
        (losses.first().unwrap_or(&1.0) - losses.last().unwrap_or(&1.0)) /
        losses.first().unwrap_or(&1.0) * 100.0
    );
    println!();

    Ok(())
}

fn print_help() {
    println!("Train a neural network for cryptocurrency trading");
    println!();
    println!("USAGE:");
    println!("    cargo run --bin train -- [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -d, --data <PATH>         Input CSV data file");
    println!("    -m, --model <PATH>        Output model file (default: model.json)");
    println!("    -e, --epochs <N>          Number of training epochs (default: 100)");
    println!("    -b, --batch <SIZE>        Batch size (default: 32)");
    println!("        --lr <RATE>           Learning rate (default: 0.001)");
    println!("    -h, --horizon <N>         Target prediction horizon (default: 1)");
    println!("        --help                Print help information");
    println!();
    println!("EXAMPLES:");
    println!("    cargo run --bin train -- --data BTCUSDT_60.csv --epochs 200");
    println!("    cargo run --bin train -- -d data.csv -m btc_model.json -e 500 -b 64");
}
