//! Yield Curve Evolution Operator Example
//!
//! Demonstrates using DeepONet to learn the operator mapping
//! today's yield curve to a future yield curve.
//!
//! The yield curve is modeled using Nelson-Siegel parameterization,
//! and the operator learns how parameter perturbations affect the
//! full curve shape.
//!
//! # Usage
//! ```bash
//! cargo run --example yield_curve
//! ```

use colored::Colorize;
use ndarray::Array1;
use operator_learning_trading::{
    data,
    operator::{DeepONet, DeepONetConfig, nelson_siegel, relative_l2_error},
};

fn main() {
    println!("{}", "═══════════════════════════════════════════════════════".blue());
    println!("{}", "  Yield Curve Operator Learning Example".blue().bold());
    println!("{}", "═══════════════════════════════════════════════════════".blue());

    let n_maturities = 30;
    let n_samples = 1000;

    // Generate yield curve data
    println!("\n{} yield curve data...", "Generating".yellow());
    let (curves_today, curves_future) =
        data::generate_yield_curve_data(n_samples, n_maturities);

    println!("  Today curves:  ({}, {})", curves_today.nrows(), curves_today.ncols());
    println!("  Future curves: ({}, {})", curves_future.nrows(), curves_future.ncols());

    // Show a sample curve
    let maturities: Vec<f64> = (0..n_maturities)
        .map(|i| 0.25 + 29.75 * i as f64 / (n_maturities - 1) as f64)
        .collect();

    println!("\n  Sample yield curve (today):");
    println!("  {:>8}  {:>8}", "Mat(Y)", "Yield(%)");
    for j in (0..n_maturities).step_by(5) {
        println!("  {:>8.1}  {:>8.3}", maturities[j], curves_today[[0, j]] * 100.0);
    }

    // Prepare DeepONet training data
    // Branch input: today's curve at all maturities
    // Trunk input: single maturity (normalized)
    // Target: future yield at that maturity
    println!("\n{} DeepONet training data...", "Preparing".yellow());

    let total = n_samples * n_maturities;
    let mut branch_data = ndarray::Array2::zeros((total, n_maturities));
    let mut trunk_data = ndarray::Array2::zeros((total, 1));
    let mut targets = Array1::zeros(total);

    let mut idx = 0;
    for i in 0..n_samples {
        for j in 0..n_maturities {
            for k in 0..n_maturities {
                branch_data[[idx, k]] = curves_today[[i, k]];
            }
            trunk_data[[idx, 0]] = maturities[j] / 30.0;
            targets[idx] = curves_future[[i, j]];
            idx += 1;
        }
    }

    let train_end = (total as f64 * 0.8) as usize;
    let branch_train = branch_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let trunk_train = trunk_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let targets_train = targets.slice(ndarray::s![..train_end]).to_owned();

    let branch_test = branch_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let trunk_test = trunk_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let targets_test = targets.slice(ndarray::s![train_end..]).to_owned();

    // Train DeepONet
    let config = DeepONetConfig {
        branch_input_dim: n_maturities,
        trunk_input_dim: 1,
        latent_dim: 64,
        branch_hidden_dims: vec![128, 128],
        trunk_hidden_dims: vec![64, 64],
        learning_rate: 0.002,
        n_epochs: 50,
        batch_size: 128,
    };

    let mut model = DeepONet::new(config);
    println!("\n{} DeepONet (branch: {}D -> trunk: {}D -> latent: {}D)...",
        "Training".yellow(), n_maturities, 1, 64);

    let losses = model.train(&branch_train, &trunk_train, &targets_train);

    let test_error = model.evaluate(&branch_test, &trunk_test, &targets_test);
    println!("\n  Test Relative L2 Error: {}", format!("{:.6}", test_error).cyan());

    // Demonstrate predictions
    println!("\n{}", "Yield Curve Predictions:".yellow().bold());

    // Pick a test sample and predict the full future curve
    let test_sample_idx = 0;
    let test_base = train_end / n_maturities * n_maturities; // align to curve boundary

    println!("\n  {:>8}  {:>10}  {:>10}  {:>8}", "Mat(Y)", "Actual(%)", "Pred(%)", "Err(bp)");
    println!("  {}", "-".repeat(44));

    for j in 0..n_maturities {
        let data_idx = test_base + j;
        if data_idx >= total { break; }

        let branch_in = branch_data.row(data_idx).to_owned();
        let trunk_in = trunk_data.row(data_idx).to_owned();
        let pred = model.predict(&branch_in, &trunk_in);
        let actual = targets[data_idx];
        let error_bps = (pred - actual).abs() * 10000.0;

        if j % 3 == 0 || j == n_maturities - 1 {
            println!(
                "  {:>8.1}  {:>10.3}  {:>10.3}  {:>8.1}",
                maturities[j],
                actual * 100.0,
                pred * 100.0,
                error_bps
            );
        }
    }

    // Zero-shot generalization demo
    println!("\n{}", "Zero-Shot Generalization:".yellow().bold());
    println!("  Predicting at maturities NOT in training data:");

    let novel_maturities = vec![0.5, 2.5, 7.5, 12.0, 25.0];
    let curve_input = branch_data.row(test_base).to_owned();

    for &mat in &novel_maturities {
        let trunk_in = Array1::from_vec(vec![mat / 30.0]);
        let pred = model.predict(&curve_input, &trunk_in);

        // Compare with Nelson-Siegel interpolation
        let ns_curve = nelson_siegel(
            &Array1::from_vec(vec![mat]),
            0.04, -0.02, 0.01, 2.0, // approximate parameters
        );

        println!(
            "  Maturity {:.1}Y: Predicted = {:.3}%, NS reference = {:.3}%",
            mat,
            pred * 100.0,
            ns_curve[0] * 100.0,
        );
    }

    println!("\n{}", "Key takeaway:".green().bold());
    println!("  The operator can predict yields at ANY maturity, including ones");
    println!("  never seen during training. This is zero-shot generalization ---");
    println!("  impossible with standard neural networks.");

    println!("\n{}", "Done!".green().bold());
}
