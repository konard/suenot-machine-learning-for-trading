//! Basic Neural Spline Flow example
//!
//! This example demonstrates the basic usage of Neural Spline Flows:
//! - Creating an NSF model
//! - Forward and inverse transformations
//! - Log probability computation
//! - Sampling from the learned distribution

use ndarray::{Array1, Array2};
use neural_spline_flows::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== Neural Spline Flow Basic Example ===\n");

    // Create a simple synthetic dataset
    println!("1. Generating synthetic data...");
    let data = generate_synthetic_data(1000, 8);
    println!("   Generated {} samples with {} features", data.nrows(), data.ncols());

    // Create NSF configuration
    println!("\n2. Creating Neural Spline Flow model...");
    let config = NSFConfig::new(8)
        .with_num_layers(4)
        .with_hidden_dim(64)
        .with_num_bins(8);

    let mut model = NeuralSplineFlow::new(config);
    println!("   Model created with {} layers", model.config().num_layers);

    // Demonstrate forward transformation
    println!("\n3. Testing forward transformation...");
    let x = data.row(0).to_owned();
    let (z, log_det) = model.forward(&x);
    println!("   Input x: {:?}", x.as_slice().unwrap());
    println!("   Latent z: {:?}", z.as_slice().unwrap());
    println!("   Log determinant: {:.4}", log_det);

    // Demonstrate inverse transformation
    println!("\n4. Testing inverse transformation...");
    let (x_recovered, _) = model.inverse(&z);
    println!("   Recovered x: {:?}", x_recovered.as_slice().unwrap());

    // Check reconstruction error
    let error: f64 = x.iter().zip(x_recovered.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    println!("   Reconstruction MSE: {:.6}", error);

    // Compute log probability
    println!("\n5. Computing log probabilities...");
    let log_probs = model.log_prob_batch(&data.slice(ndarray::s![..10, ..]).to_owned());
    println!("   Log probs for first 10 samples:");
    for (i, lp) in log_probs.iter().enumerate() {
        println!("   Sample {}: {:.4}", i, lp);
    }

    // Generate samples from the model
    println!("\n6. Generating samples from the model...");
    let samples = model.sample(5);
    println!("   Generated 5 samples:");
    for i in 0..5 {
        println!("   Sample {}: {:?}", i, samples.row(i).as_slice().unwrap());
    }

    // Demonstrate density estimation
    println!("\n7. Density estimation...");
    let test_point = Array1::from_vec(vec![0.0; 8]);
    let density = model.density(&test_point);
    let is_in_dist = model.is_in_distribution(&test_point, -20.0);
    println!("   Density at origin: {:.6}", density);
    println!("   In distribution (threshold -20): {}", is_in_dist);

    // Test with outlier
    let outlier = Array1::from_vec(vec![10.0; 8]);
    let outlier_density = model.density(&outlier);
    let outlier_in_dist = model.is_in_distribution(&outlier, -20.0);
    println!("\n   Density at outlier (10, 10, ...): {:.6}", outlier_density);
    println!("   In distribution: {}", outlier_in_dist);

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Generate synthetic data for testing
fn generate_synthetic_data(n_samples: usize, n_features: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Create correlated features to simulate market data
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        // Base random values
        let base: f64 = normal.sample(&mut rng);

        for j in 0..n_features {
            // Add correlation with base and some independent noise
            let noise: f64 = normal.sample(&mut rng);
            let correlation = 0.5 + 0.3 * (j as f64 / n_features as f64);
            data[[i, j]] = correlation * base + (1.0 - correlation) * noise;
        }
    }

    data
}
