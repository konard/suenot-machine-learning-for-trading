//! Basic Continuous Normalizing Flow Example
//!
//! This example demonstrates:
//! - Creating a CNF model
//! - Encoding and decoding data
//! - Computing log-probabilities
//! - Generating samples

use cnf_trading::{
    cnf::{ContinuousNormalizingFlow, ODESolver, ODEMethod},
    utils::{generate_synthetic_candles, compute_market_features, compute_features_batch, normalize_features},
};
use ndarray::Array1;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Continuous Normalizing Flows Basic Example ===\n");

    // Generate synthetic market data
    println!("Generating synthetic candle data...");
    let candles = generate_synthetic_candles(500, 100.0);
    println!("Generated {} candles\n", candles.len());

    // Compute features
    println!("Computing market features...");
    let lookback = 20;
    let features = compute_features_batch(&candles, lookback);
    println!("Feature matrix shape: {:?}\n", features.shape());

    // Normalize features
    println!("Normalizing features...");
    let (normalized, means, stds) = normalize_features(&features);
    println!("Means: {:?}", means);
    println!("Stds: {:?}\n", stds);

    // Create CNF model
    println!("Creating CNF model...");
    let dim = 9; // Number of features
    let hidden_dim = 64;
    let num_layers = 3;

    let cnf = ContinuousNormalizingFlow::new(dim, hidden_dim, num_layers)
        .with_solver(ODESolver::new(ODEMethod::RK4, 50));

    println!("Model created with dim={}, hidden_dim={}, num_layers={}\n",
             dim, hidden_dim, num_layers);

    // Test encoding and decoding
    println!("Testing encode/decode cycle...");
    let sample = normalized.row(0).to_owned();
    println!("Original: {:?}", sample);

    let (z, log_det_encode) = cnf.encode(&sample);
    println!("Encoded (z): {:?}", z);
    println!("Log-det (encode): {:.4}", log_det_encode);

    let (reconstructed, log_det_decode) = cnf.decode(&z);
    println!("Reconstructed: {:?}", reconstructed);
    println!("Log-det (decode): {:.4}", log_det_decode);

    let reconstruction_error: f64 = sample.iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("Reconstruction error: {:.6}\n", reconstruction_error);

    // Compute log-probabilities
    println!("Computing log-probabilities...");
    let log_probs = cnf.log_prob_batch(&normalized);
    let mean_log_prob = log_probs.mean().unwrap();
    let min_log_prob = log_probs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Mean log-prob: {:.4}", mean_log_prob);
    println!("Min log-prob: {:.4}", min_log_prob);
    println!("Max log-prob: {:.4}\n", max_log_prob);

    // Generate samples
    println!("Generating samples from the model...");
    let samples = cnf.sample(10);
    println!("Generated {} samples:", samples.nrows());
    for i in 0..samples.nrows().min(3) {
        println!("  Sample {}: {:?}", i, samples.row(i));
    }
    println!();

    // Demonstrate velocity field
    println!("Demonstrating velocity field...");
    let test_point = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    let t = 0.5;
    let velocity = cnf.velocity_field.forward(&test_point, t);
    println!("Point: {:?}", test_point);
    println!("Time: {}", t);
    println!("Velocity: {:?}\n", velocity);

    // Test with different ODE methods
    println!("Comparing ODE methods...");

    let euler_solver = ODESolver::new(ODEMethod::Euler, 100);
    let cnf_euler = ContinuousNormalizingFlow::new(dim, hidden_dim, num_layers)
        .with_solver(euler_solver);

    let rk4_solver = ODESolver::new(ODEMethod::RK4, 50);
    let cnf_rk4 = ContinuousNormalizingFlow::new(dim, hidden_dim, num_layers)
        .with_solver(rk4_solver);

    let sample = normalized.row(0).to_owned();

    let start = std::time::Instant::now();
    let (z_euler, _) = cnf_euler.encode(&sample);
    let euler_time = start.elapsed();

    let start = std::time::Instant::now();
    let (z_rk4, _) = cnf_rk4.encode(&sample);
    let rk4_time = start.elapsed();

    println!("Euler (100 steps): {:?}", euler_time);
    println!("RK4 (50 steps): {:?}", rk4_time);
    println!();

    println!("=== Example Complete ===");

    Ok(())
}
