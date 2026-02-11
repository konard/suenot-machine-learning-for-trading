//! Physics Constraints Demo
//!
//! Demonstrates the evaluation of financial physics constraints
//! on both Heston-generated data and simple noise, showing why
//! physics constraints matter for realistic financial time series.
//!
//! Run with:
//!   cargo run --example physics_constraints_demo

use physics_constrained_gan::{
    heston::{self, HestonParams},
    physics::{self, PhysicsConfig},
    statistics,
};
use rand::Rng;
use rand_distr::{Distribution, Normal};

fn main() {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║   Physics Constraints Demo                           ║");
    println!("║   Comparing Heston vs Gaussian vs Uniform noise      ║");
    println!("╚═══════════════════════════════════════════════════════╝");

    let seq_len = 1000;
    let n_paths = 100;
    let config = PhysicsConfig::default();

    // ---- 1. Heston model (realistic) ----
    println!("\n--- 1. Heston Stochastic Volatility Model ---");
    let heston_params = HestonParams::default();
    let heston_paths = heston::generate_multiple_paths(&heston_params, n_paths, seq_len);
    let heston_flat: Vec<f64> = heston_paths.iter().flatten().cloned().collect();

    let heston_report = physics::evaluate_physics(&heston_flat, &config);
    let heston_stats = statistics::compute_statistics(&heston_flat);
    println!("  Total physics loss: {:.6}", heston_report.total_physics_loss);
    println!("  Mean: {:.8}, Std: {:.6}", heston_stats.mean, heston_stats.std);
    println!("  Kurtosis: {:.4}, Skewness: {:.4}", heston_stats.kurtosis, heston_stats.skewness);
    println!("  ACF(|r|, lag=1): {:.4}", heston_stats.acf_abs_lag1);
    println!("  Leverage corr: {:.4}", heston_stats.leverage_corr);

    // ---- 2. Gaussian noise (no stylized facts) ----
    println!("\n--- 2. Gaussian White Noise ---");
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 0.02).unwrap();
    let gaussian: Vec<f64> = (0..seq_len * n_paths)
        .map(|_| normal.sample(&mut rng))
        .collect();

    let gaussian_report = physics::evaluate_physics(&gaussian, &config);
    let gaussian_stats = statistics::compute_statistics(&gaussian);
    println!("  Total physics loss: {:.6}", gaussian_report.total_physics_loss);
    println!("  Mean: {:.8}, Std: {:.6}", gaussian_stats.mean, gaussian_stats.std);
    println!("  Kurtosis: {:.4} (expected ~0 for Gaussian)", gaussian_stats.kurtosis);
    println!("  ACF(|r|, lag=1): {:.4} (expected ~0)", gaussian_stats.acf_abs_lag1);
    println!("  Leverage corr: {:.4} (expected ~0)", gaussian_stats.leverage_corr);

    // ---- 3. Uniform noise ----
    println!("\n--- 3. Uniform Noise ---");
    let uniform: Vec<f64> = (0..seq_len * n_paths)
        .map(|_| rng.gen_range(-0.05..0.05))
        .collect();

    let uniform_report = physics::evaluate_physics(&uniform, &config);
    let uniform_stats = statistics::compute_statistics(&uniform);
    println!("  Total physics loss: {:.6}", uniform_report.total_physics_loss);
    println!("  Mean: {:.8}, Std: {:.6}", uniform_stats.mean, uniform_stats.std);
    println!("  Kurtosis: {:.4} (expected -1.2 for Uniform)", uniform_stats.kurtosis);
    println!("  ACF(|r|, lag=1): {:.4}", uniform_stats.acf_abs_lag1);
    println!("  Leverage corr: {:.4}", uniform_stats.leverage_corr);

    // ---- Summary comparison ----
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   Summary Comparison                                  ║");
    println!("╠════════════════════════════════════════════════════════╣");
    println!("║ {:20} {:>10} {:>10} {:>10} ║", "Metric", "Heston", "Gaussian", "Uniform");
    println!("╠════════════════════════════════════════════════════════╣");
    println!("║ {:20} {:>10.4} {:>10.4} {:>10.4} ║",
             "Physics Loss",
             heston_report.total_physics_loss,
             gaussian_report.total_physics_loss,
             uniform_report.total_physics_loss);
    println!("║ {:20} {:>10.4} {:>10.4} {:>10.4} ║",
             "Kurtosis",
             heston_stats.kurtosis,
             gaussian_stats.kurtosis,
             uniform_stats.kurtosis);
    println!("║ {:20} {:>10.4} {:>10.4} {:>10.4} ║",
             "ACF(|r|, lag=1)",
             heston_stats.acf_abs_lag1,
             gaussian_stats.acf_abs_lag1,
             uniform_stats.acf_abs_lag1);
    println!("║ {:20} {:>10.4} {:>10.4} {:>10.4} ║",
             "Leverage Corr",
             heston_stats.leverage_corr,
             gaussian_stats.leverage_corr,
             uniform_stats.leverage_corr);
    println!("║ {:20} {:>10.4} {:>10.4} {:>10.4} ║",
             "Martingale Loss",
             heston_report.martingale_loss,
             gaussian_report.martingale_loss,
             uniform_report.martingale_loss);
    println!("╚════════════════════════════════════════════════════════╝");

    println!("\nKey Takeaways:");
    println!("  1. Heston model naturally produces data with financial stylized facts");
    println!("  2. Gaussian noise has zero kurtosis, no vol clustering, no leverage effect");
    println!("  3. Uniform noise has NEGATIVE kurtosis (thin tails -- even worse)");
    println!("  4. Physics constraints help GANs learn to produce Heston-like properties");
    println!("  5. Without constraints, GANs tend to produce Gaussian-like outputs");
}
