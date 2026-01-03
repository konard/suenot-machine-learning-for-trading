//! Basic RealNVP Example
//!
//! This example demonstrates the core functionality of RealNVP:
//! - Creating a RealNVP flow
//! - Forward and inverse transformations
//! - Computing log probabilities
//! - Generating samples

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Normal, Distribution};

use realnvp_trading::flow::RealNVP;

fn main() {
    println!("=== RealNVP Basic Example ===\n");

    // Configuration
    let dim = 4;          // Input dimension
    let hidden_dim = 64;   // Hidden layer size
    let num_layers = 8;    // Number of coupling layers

    // Create RealNVP flow
    println!("Creating RealNVP flow with:");
    println!("  - Dimension: {}", dim);
    println!("  - Hidden dim: {}", hidden_dim);
    println!("  - Coupling layers: {}", num_layers);

    let flow = RealNVP::new(dim, hidden_dim, num_layers);

    // Example 1: Forward and Inverse Transformation
    println!("\n--- Example 1: Forward/Inverse Transformation ---");

    let x = Array1::from_vec(vec![1.0, -0.5, 0.3, -0.2]);
    println!("Original x: {:?}", x);

    let (z, log_det) = flow.forward(&x);
    println!("Latent z: {:?}", z);
    println!("Log determinant: {:.4}", log_det);

    let x_reconstructed = flow.inverse(&z);
    println!("Reconstructed x: {:?}", x_reconstructed);

    // Verify reconstruction
    let reconstruction_error: f64 = x.iter()
        .zip(x_reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("Reconstruction error: {:.2e}", reconstruction_error);

    // Example 2: Log Probability
    println!("\n--- Example 2: Log Probability ---");

    let test_points = vec![
        Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]),
        Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0]),
        Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0]),
    ];

    for point in &test_points {
        let log_prob = flow.log_prob(point);
        println!("Point {:?} -> log p(x) = {:.4}", point.as_slice().unwrap(), log_prob);
    }

    // Example 3: Sampling
    println!("\n--- Example 3: Sampling ---");

    let n_samples = 5;
    let samples = flow.sample(n_samples);

    println!("Generated {} samples:", n_samples);
    for (i, sample) in samples.rows().into_iter().enumerate() {
        println!("  Sample {}: {:?}", i + 1, sample.to_vec());
    }

    // Example 4: Batch Processing
    println!("\n--- Example 4: Batch Processing ---");

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Create batch of random data
    let batch_size = 100;
    let batch_data = Array2::from_shape_fn(
        (batch_size, dim),
        |_| normal.sample(&mut rng),
    );

    let mut flow_mut = RealNVP::new(dim, hidden_dim, num_layers);

    // Forward pass on batch
    let (z_batch, log_dets) = flow_mut.forward_batch(&batch_data);
    println!("Batch forward pass complete:");
    println!("  - Input shape: {:?}", batch_data.dim());
    println!("  - Output shape: {:?}", z_batch.dim());
    println!("  - Mean log det: {:.4}", log_dets.mean().unwrap_or(0.0));

    // Log probabilities for batch
    let log_probs = flow_mut.log_prob_batch(&batch_data);
    println!("Log probabilities:");
    println!("  - Mean: {:.4}", log_probs.mean().unwrap_or(0.0));
    println!("  - Std: {:.4}", log_probs.std(0.0));
    println!("  - Min: {:.4}", log_probs.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("  - Max: {:.4}", log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // Example 5: Density Estimation
    println!("\n--- Example 5: Density Estimation ---");

    let threshold = -15.0;
    println!("In-distribution threshold: log p(x) > {}", threshold);

    for point in &test_points {
        let log_prob = flow.log_prob(point);
        let in_dist = flow.is_in_distribution(point, threshold);
        println!(
            "Point {:?}: log p = {:.4}, in_dist = {}",
            point.as_slice().unwrap(),
            log_prob,
            in_dist
        );
    }

    // Example 6: Training Loss
    println!("\n--- Example 6: Negative Log Likelihood Loss ---");

    // Create synthetic data
    let train_data = Array2::from_shape_fn(
        (500, dim),
        |_| normal.sample(&mut rng),
    );

    let nll_loss = flow_mut.nll_loss(&train_data);
    println!("NLL Loss on random data: {:.4}", nll_loss);

    println!("\n=== Example Complete ===");
}
