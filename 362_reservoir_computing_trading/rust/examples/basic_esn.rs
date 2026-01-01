//! Basic Echo State Network Example
//!
//! This example demonstrates how to create, train, and use an ESN
//! for time series prediction.
//!
//! Run with: cargo run --example basic_esn

use ndarray::{Array1, Array2};
use reservoir_trading::{EchoStateNetwork, EsnConfig};

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Reservoir Computing - Basic ESN Example");
    println!("═══════════════════════════════════════════════════════════\n");

    // Generate synthetic time series data
    println!("1. Generating synthetic time series data...");
    let n_samples = 1000;
    let n_inputs = 3;
    let n_outputs = 1;

    let mut inputs = Array2::zeros((n_samples, n_inputs));
    let mut targets = Array2::zeros((n_samples, n_outputs));

    for i in 0..n_samples {
        let t = i as f64 / 50.0;

        // Input features: sin, cos, and their product
        inputs[[i, 0]] = t.sin();
        inputs[[i, 1]] = t.cos();
        inputs[[i, 2]] = t.sin() * t.cos();

        // Target: predict future value (nonlinear combination)
        let future_t = t + 0.1;
        targets[[i, 0]] = future_t.sin() * 0.5 + (2.0 * future_t).cos() * 0.3;
    }

    println!("   Generated {} samples with {} features", n_samples, n_inputs);

    // Create ESN configuration
    println!("\n2. Creating ESN with configuration:");
    let config = EsnConfig {
        reservoir_size: 200,
        spectral_radius: 0.95,
        input_scaling: 0.5,
        leaking_rate: 0.3,
        sparsity: 0.1,
        regularization: 1e-6,
        seed: 42,
    };

    println!("   - Reservoir size: {}", config.reservoir_size);
    println!("   - Spectral radius: {}", config.spectral_radius);
    println!("   - Input scaling: {}", config.input_scaling);
    println!("   - Leaking rate: {}", config.leaking_rate);
    println!("   - Sparsity: {}", config.sparsity);
    println!("   - Regularization: {:e}", config.regularization);

    // Create and train ESN
    println!("\n3. Training ESN...");
    let mut esn = EchoStateNetwork::new(n_inputs, n_outputs, config);

    let train_size = 800;
    let washout = 100;

    let train_inputs = inputs.slice(ndarray::s![..train_size, ..]).to_owned();
    let train_targets = targets.slice(ndarray::s![..train_size, ..]).to_owned();

    let start = std::time::Instant::now();
    let mse = esn.fit(&train_inputs, &train_targets, washout).unwrap();
    let training_time = start.elapsed();

    println!("   Training completed in {:?}", training_time);
    println!("   Training MSE: {:.6}", mse);
    println!("   Number of trainable parameters: {}", esn.n_parameters());

    // Test predictions
    println!("\n4. Making predictions on test set...");
    esn.reset_state();

    // Warm up on training data
    let _ = esn.predict(&train_inputs);

    // Predict on test data
    let test_inputs = inputs.slice(ndarray::s![train_size.., ..]).to_owned();
    let test_targets = targets.slice(ndarray::s![train_size.., ..]).to_owned();

    let predictions = esn.predict(&test_inputs).unwrap();

    // Calculate test metrics
    let mut test_mse = 0.0;
    let mut test_mae = 0.0;

    for i in 0..predictions.nrows() {
        let error = predictions[[i, 0]] - test_targets[[i, 0]];
        test_mse += error * error;
        test_mae += error.abs();
    }

    test_mse /= predictions.nrows() as f64;
    test_mae /= predictions.nrows() as f64;

    println!("   Test MSE: {:.6}", test_mse);
    println!("   Test MAE: {:.6}", test_mae);
    println!("   Test R²: {:.4}", 1.0 - test_mse / variance(&test_targets.column(0).to_vec()));

    // Show sample predictions
    println!("\n5. Sample predictions (first 10 test samples):");
    println!("   {:>10} {:>10} {:>10}", "Actual", "Predicted", "Error");
    println!("   {:-<10} {:-<10} {:-<10}", "", "", "");

    for i in 0..10.min(predictions.nrows()) {
        let actual = test_targets[[i, 0]];
        let pred = predictions[[i, 0]];
        let error = (pred - actual).abs();
        println!("   {:>10.4} {:>10.4} {:>10.4}", actual, pred, error);
    }

    // Demonstrate online prediction
    println!("\n6. Online (one-step) prediction demo:");
    esn.reset_state();

    // Warm up
    for i in 0..train_size {
        let input = inputs.row(i).to_owned();
        let _ = esn.predict_one(&input);
    }

    // Online predictions
    println!("   Making 5 sequential predictions:");
    for i in 0..5 {
        let idx = train_size + i;
        let input = inputs.row(idx).to_owned();
        let pred = esn.predict_one(&input).unwrap();
        let actual = targets[[idx, 0]];

        println!(
            "   Step {}: Predicted = {:.4}, Actual = {:.4}, Error = {:.4}",
            i + 1,
            pred[0],
            actual,
            (pred[0] - actual).abs()
        );
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Example completed successfully!");
    println!("═══════════════════════════════════════════════════════════");
}

fn variance(data: &[f64]) -> f64 {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}
