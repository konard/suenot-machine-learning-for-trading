//! Example: Bias-Variance Tradeoff Analysis
//!
//! Demonstrates the fundamental ML tradeoff:
//! - High bias (underfitting): Model too simple
//! - High variance (overfitting): Model too complex
//!
//! Uses polynomial regression to illustrate the concept.
//!
//! Run with: cargo run --example bias_variance

use ml_crypto::ml::BiasVarianceAnalyzer;
use ndarray::Array1;

fn main() {
    println!("=== Bias-Variance Tradeoff Example ===\n");

    // Define the true underlying function
    // We'll use a cosine function which can be approximated by polynomials
    let true_fn = |x: f64| (x * std::f64::consts::PI).cos();

    // Parameters
    let degrees = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let n_experiments = 100;
    let n_train = 30;
    let n_test = 50;
    let noise_std = 0.3;

    println!("Experimental Setup:");
    println!("  True function: cos(πx)");
    println!("  Noise std: {}", noise_std);
    println!("  Training samples per experiment: {}", n_train);
    println!("  Number of experiments: {}", n_experiments);
    println!("  Test samples: {}", n_test);
    println!();

    // Run bias-variance analysis
    println!("=== Bias-Variance Decomposition ===\n");

    let results = BiasVarianceAnalyzer::analyze_bias_variance(
        true_fn,
        &degrees,
        n_experiments,
        n_train,
        n_test,
        noise_std,
    );

    println!("{:>6} {:>12} {:>12} {:>12} {:>12}",
             "Degree", "Bias²", "Variance", "Noise²", "Total");
    println!("{:-<58}", "");

    let irreducible = noise_std.powi(2);

    for (degree, bias_sq, variance, total) in &results {
        println!("{:>6} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                 degree, bias_sq, variance, irreducible, total);
    }

    // Find optimal complexity
    let (best_degree, best_bias, best_var, min_error) = results
        .iter()
        .cloned()
        .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .unwrap();

    println!("\n=== Analysis ===\n");
    println!("Optimal polynomial degree: {}", best_degree);
    println!("  Bias²: {:.4}", best_bias);
    println!("  Variance: {:.4}", best_var);
    println!("  Total error: {:.4}", min_error);

    // Interpretation
    println!("\n=== Interpretation ===\n");

    if best_degree <= 2 {
        println!("A low-degree polynomial works best, suggesting:");
        println!("  - The true function is relatively simple");
        println!("  - Higher degrees would overfit the noise");
    } else if best_degree <= 5 {
        println!("A moderate-degree polynomial works best, suggesting:");
        println!("  - Good balance between bias and variance");
        println!("  - Model complexity matches true function complexity");
    } else {
        println!("A high-degree polynomial works best, suggesting:");
        println!("  - The true function has complex structure");
        println!("  - Or there's significant noise masking the optimal complexity");
    }

    // Learning curve example
    println!("\n=== Learning Curve (degree={}) ===\n", best_degree);

    // Generate full dataset
    let (x, _, y) = BiasVarianceAnalyzer::generate_data(true_fn, (-1.0, 1.0), 200, noise_std);

    let train_sizes: Vec<usize> = vec![10, 20, 30, 50, 80, 100, 120, 150];
    let learning_curve = BiasVarianceAnalyzer::learning_curve(&train_sizes, &x, &y, best_degree);

    println!("{:>12} {:>15} {:>15}",
             "Train Size", "Train MSE", "Test MSE");
    println!("{:-<45}", "");

    for (size, train_mse, test_mse) in &learning_curve {
        println!("{:>12} {:>15.4} {:>15.4}", size, train_mse, test_mse);
    }

    println!("\nLearning curve interpretation:");
    if let (Some(first), Some(last)) = (learning_curve.first(), learning_curve.last()) {
        let train_improvement = (first.1 - last.1) / first.1 * 100.0;
        let test_improvement = (first.2 - last.2) / first.2 * 100.0;

        println!("  Train error improved by {:.1}% with more data", train_improvement);
        println!("  Test error improved by {:.1}% with more data", test_improvement);

        let gap = last.2 - last.1;
        if gap > 0.1 {
            println!("  Large train-test gap suggests possible overfitting");
        } else {
            println!("  Small train-test gap suggests good generalization");
        }
    }

    // Validation curve
    println!("\n=== Validation Curve ===\n");

    let val_degrees: Vec<usize> = (1..=12).collect();
    let validation_curve = BiasVarianceAnalyzer::validation_curve(&val_degrees, &x, &y, 0.2);

    println!("{:>8} {:>15} {:>15} {:>12}",
             "Degree", "Train MSE", "Test MSE", "Gap");
    println!("{:-<55}", "");

    for (degree, train_mse, test_mse) in &validation_curve {
        let gap = test_mse - train_mse;
        println!("{:>8} {:>15.4} {:>15.4} {:>12.4}",
                 degree, train_mse, test_mse, gap);
    }

    // Find crossing point / optimal
    println!("\n=== Summary ===");
    println!();
    println!("The bias-variance tradeoff shows that:");
    println!("  - Simple models (low degree) have high BIAS but low VARIANCE");
    println!("  - Complex models (high degree) have low BIAS but high VARIANCE");
    println!("  - The optimal model balances both to minimize total error");
    println!();
    println!("For trading applications:");
    println!("  - Underfitting: Strategy misses real patterns");
    println!("  - Overfitting: Strategy learns noise, fails on new data");
    println!("  - Goal: Find the sweet spot that generalizes well");
}
