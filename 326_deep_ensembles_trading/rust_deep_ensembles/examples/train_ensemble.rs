//! Example: Training a Deep Ensemble model
//!
//! This example demonstrates how to create, train, and evaluate
//! a Deep Ensemble model for uncertainty-aware predictions.

use deep_ensembles_trading::ensemble::config::EnsembleConfig;
use deep_ensembles_trading::ensemble::model::{compute_calibration_error, DeepEnsemble};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn main() {
    println!("=== Deep Ensemble Training Example ===\n");

    // Step 1: Generate synthetic data
    println!("1. Generating synthetic data...");
    let (x_train, y_train, x_val, y_val, x_test, y_test) = generate_data(2000, 8);

    println!("   Training samples:   {}", x_train.nrows());
    println!("   Validation samples: {}", x_val.nrows());
    println!("   Test samples:       {}", x_test.nrows());
    println!("   Features:           {}", x_train.ncols());

    // Step 2: Create ensemble configuration
    println!("\n2. Creating ensemble configuration...");
    let config = EnsembleConfig {
        num_models: 5,
        input_dim: 8,
        hidden_dims: vec![128, 64, 32],
        dropout_rate: 0.1,
        learning_rate: 0.005,
        batch_size: 64,
        max_epochs: 100,
        early_stopping_patience: 10,
        min_std: 1e-6,
    };

    println!("   Number of models: {}", config.num_models);
    println!("   Hidden dims: {:?}", config.hidden_dims);
    println!("   Learning rate: {}", config.learning_rate);

    // Step 3: Create and train ensemble
    println!("\n3. Training Deep Ensemble...");
    let mut ensemble = DeepEnsemble::new(config);

    let history = ensemble.train(&x_train, &y_train, Some(&x_val), Some(&y_val), true);

    println!("\n   Training complete!");
    println!("   Model losses: {:?}", history);

    // Step 4: Make predictions
    println!("\n4. Making predictions on test set...");
    let predictions = ensemble.predict(&x_test);

    println!("   Number of samples: {}", predictions.num_samples());
    println!("   Number of models: {}", predictions.num_models());

    // Step 5: Analyze uncertainty
    println!("\n5. Uncertainty Analysis:");

    let avg_total = predictions.total_std.iter().sum::<f64>() / predictions.num_samples() as f64;
    let avg_epistemic =
        predictions.epistemic_std.iter().sum::<f64>() / predictions.num_samples() as f64;
    let avg_aleatoric =
        predictions.aleatoric_std.iter().sum::<f64>() / predictions.num_samples() as f64;

    println!("   Average Total Uncertainty:     {:.4}", avg_total);
    println!("   Average Epistemic Uncertainty: {:.4}", avg_epistemic);
    println!("   Average Aleatoric Uncertainty: {:.4}", avg_aleatoric);
    println!(
        "   Epistemic/Total Ratio:         {:.2}%",
        (avg_epistemic / avg_total) * 100.0
    );

    // Step 6: Compute metrics
    println!("\n6. Model Metrics:");

    // MSE
    let mse: f64 = predictions
        .mean
        .iter()
        .zip(y_test.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / predictions.num_samples() as f64;

    // MAE
    let mae: f64 = predictions
        .mean
        .iter()
        .zip(y_test.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / predictions.num_samples() as f64;

    // Calibration error
    let ece = compute_calibration_error(&predictions, &y_test, 10);

    println!("   MSE: {:.4}", mse);
    println!("   MAE: {:.4}", mae);
    println!("   Expected Calibration Error: {:.4}", ece);

    // Step 7: Show sample predictions
    println!("\n7. Sample Predictions (first 5 test samples):");
    println!("   {:-^70}", "");
    println!(
        "   {:>10} | {:>10} | {:>8} | {:>8} | {:>8}",
        "True", "Predicted", "TotalStd", "Epist", "Aleat"
    );
    println!("   {:-^70}", "");

    for i in 0..5.min(predictions.num_samples()) {
        println!(
            "   {:>10.4} | {:>10.4} | {:>8.4} | {:>8.4} | {:>8.4}",
            y_test[i],
            predictions.mean[i],
            predictions.total_std[i],
            predictions.epistemic_std[i],
            predictions.aleatoric_std[i]
        );
    }

    // Step 8: Show individual model predictions for first sample
    println!("\n8. Individual Model Predictions (sample 0):");
    println!("   True value: {:.4}", y_test[0]);
    for i in 0..predictions.num_models() {
        println!(
            "   Model {}: {:.4} +/- {:.4}",
            i + 1,
            predictions.individual_means[i][0],
            predictions.individual_stds[i][0]
        );
    }
    println!(
        "   Ensemble: {:.4} +/- {:.4}",
        predictions.mean[0], predictions.total_std[0]
    );

    println!("\n=== Training example complete! ===");
}

/// Generate synthetic data for training
fn generate_data(
    n: usize,
    d: usize,
) -> (
    Array2<f64>,
    Array1<f64>,
    Array2<f64>,
    Array1<f64>,
    Array2<f64>,
    Array1<f64>,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate features
    let features = Array2::from_shape_fn((n, d), |_| normal.sample(&mut rng));

    // Generate targets: linear combination + non-linearity + noise
    let weights = Array1::from_shape_fn(d, |_| normal.sample(&mut rng));
    let linear = features.dot(&weights);

    let targets = Array1::from_shape_fn(n, |i| {
        let nonlinear = 0.3 * (features[[i, 0]] * 2.0).sin();
        let noise = 0.5 * normal.sample(&mut rng);
        linear[i] + nonlinear + noise
    });

    // Split data
    let train_end = (n as f64 * 0.6) as usize;
    let val_end = (n as f64 * 0.8) as usize;

    let x_train = features.slice(ndarray::s![..train_end, ..]).to_owned();
    let y_train = targets.slice(ndarray::s![..train_end]).to_owned();
    let x_val = features.slice(ndarray::s![train_end..val_end, ..]).to_owned();
    let y_val = targets.slice(ndarray::s![train_end..val_end]).to_owned();
    let x_test = features.slice(ndarray::s![val_end.., ..]).to_owned();
    let y_test = targets.slice(ndarray::s![val_end..]).to_owned();

    (x_train, y_train, x_val, y_val, x_test, y_test)
}
