//! Example: Basic PCA
//!
//! This example demonstrates the basic concepts of PCA
//! using synthetic data.

use ndarray::{array, Array2};
use rand::Rng;

fn main() {
    println!("===========================================");
    println!("  Principal Component Analysis - Basics");
    println!("===========================================");
    println!();

    // Generate correlated 2D data
    let data = generate_correlated_data(200, 0.8);
    println!("Generated {} data points in 2D", data.nrows());

    // Calculate and display covariance
    let cov = covariance_matrix(&data);
    println!("\nCovariance Matrix:");
    println!("  [{:>8.4}, {:>8.4}]", cov[[0, 0]], cov[[0, 1]]);
    println!("  [{:>8.4}, {:>8.4}]", cov[[1, 0]], cov[[1, 1]]);

    // Perform PCA
    let (eigenvalues, eigenvectors) = eigen_decomposition(&cov);

    println!("\nEigenvalues (Explained Variance):");
    for (i, val) in eigenvalues.iter().enumerate() {
        let ratio = val / eigenvalues.iter().sum::<f64>();
        println!("  PC{}: {:.4} ({:.1}%)", i + 1, val, ratio * 100.0);
    }

    println!("\nPrincipal Components (Eigenvectors):");
    println!("  PC1: [{:>7.4}, {:>7.4}]", eigenvectors[0][0], eigenvectors[0][1]);
    println!("  PC2: [{:>7.4}, {:>7.4}]", eigenvectors[1][0], eigenvectors[1][1]);

    // Verify orthogonality
    let dot_product: f64 = eigenvectors[0]
        .iter()
        .zip(eigenvectors[1].iter())
        .map(|(a, b)| a * b)
        .sum();
    println!("\nPC1 · PC2 = {:.10} (should be ~0)", dot_product);

    // Project data onto first PC
    let projected = project_onto_pc(&data, &eigenvectors[0]);

    // Calculate reconstruction error
    let reconstructed = reconstruct_from_pc(&projected, &eigenvectors[0], &mean_vector(&data));
    let error = reconstruction_error(&data, &reconstructed);

    println!("\nDimensionality Reduction:");
    println!("  Original dimensions: 2");
    println!("  Reduced dimensions: 1");
    println!("  Reconstruction RMSE: {:.4}", error);

    // Compare with keeping both PCs
    println!(
        "\nFirst PC explains {:.1}% of variance",
        eigenvalues[0] / eigenvalues.iter().sum::<f64>() * 100.0
    );

    println!("\n===========================================");
    println!("  Demonstration Complete");
    println!("===========================================");
}

fn generate_correlated_data(n: usize, correlation: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Array2::zeros((n, 2));

    for i in 0..n {
        let x: f64 = rng.gen_range(-2.0..2.0);
        let noise: f64 = rng.gen_range(-1.0..1.0) * (1.0 - correlation.abs()).sqrt();
        let y = correlation * x + noise;

        data[[i, 0]] = x;
        data[[i, 1]] = y;
    }

    data
}

fn mean_vector(data: &Array2<f64>) -> Vec<f64> {
    let n = data.nrows() as f64;
    let mut mean = vec![0.0; data.ncols()];

    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            mean[j] += data[[i, j]];
        }
    }

    mean.iter_mut().for_each(|x| *x /= n);
    mean
}

fn covariance_matrix(data: &Array2<f64>) -> Array2<f64> {
    let n = data.nrows() as f64;
    let mean = mean_vector(data);
    let d = data.ncols();

    let mut cov = Array2::zeros((d, d));

    for i in 0..data.nrows() {
        for j in 0..d {
            for k in 0..d {
                cov[[j, k]] += (data[[i, j]] - mean[j]) * (data[[i, k]] - mean[k]);
            }
        }
    }

    cov /= n - 1.0;
    cov
}

fn eigen_decomposition(cov: &Array2<f64>) -> (Vec<f64>, Vec<Vec<f64>>) {
    // For 2x2 symmetric matrix, we can compute analytically
    let a = cov[[0, 0]];
    let b = cov[[0, 1]]; // = cov[[1, 0]] for symmetric
    let d = cov[[1, 1]];

    // Eigenvalues from quadratic formula
    let trace = a + d;
    let det = a * d - b * b;
    let discriminant = (trace * trace - 4.0 * det).sqrt();

    let lambda1 = (trace + discriminant) / 2.0;
    let lambda2 = (trace - discriminant) / 2.0;

    // Eigenvectors
    let v1 = if b.abs() > 1e-10 {
        let x = lambda1 - d;
        let len = (x * x + b * b).sqrt();
        vec![x / len, b / len]
    } else {
        vec![1.0, 0.0]
    };

    let v2 = if b.abs() > 1e-10 {
        let x = lambda2 - d;
        let len = (x * x + b * b).sqrt();
        vec![x / len, b / len]
    } else {
        vec![0.0, 1.0]
    };

    (vec![lambda1, lambda2], vec![v1, v2])
}

fn project_onto_pc(data: &Array2<f64>, pc: &[f64]) -> Vec<f64> {
    let mean = mean_vector(data);
    let n = data.nrows();
    let mut projected = vec![0.0; n];

    for i in 0..n {
        let mut dot = 0.0;
        for j in 0..data.ncols() {
            dot += (data[[i, j]] - mean[j]) * pc[j];
        }
        projected[i] = dot;
    }

    projected
}

fn reconstruct_from_pc(projected: &[f64], pc: &[f64], mean: &[f64]) -> Array2<f64> {
    let n = projected.len();
    let d = pc.len();
    let mut reconstructed = Array2::zeros((n, d));

    for i in 0..n {
        for j in 0..d {
            reconstructed[[i, j]] = projected[i] * pc[j] + mean[j];
        }
    }

    reconstructed
}

fn reconstruction_error(original: &Array2<f64>, reconstructed: &Array2<f64>) -> f64 {
    let n = original.nrows();
    let d = original.ncols();
    let mut sum_sq = 0.0;

    for i in 0..n {
        for j in 0..d {
            let diff = original[[i, j]] - reconstructed[[i, j]];
            sum_sq += diff * diff;
        }
    }

    (sum_sq / (n * d) as f64).sqrt()
}
