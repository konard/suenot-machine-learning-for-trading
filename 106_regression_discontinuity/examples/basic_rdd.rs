//! Basic RDD usage example.
//!
//! Demonstrates Regression Discontinuity Design on synthetic data,
//! showing how to estimate treatment effects at thresholds.

use rdd_trading::model::rdd::{Kernel, RDDModel};
use rdd_trading::model::validator::RDDValidator;

fn main() {
    println!("=== Regression Discontinuity Design - Basic Example ===\n");

    // Generate synthetic data with treatment effect at cutoff
    let (running_var, outcome) = generate_synthetic_rdd_data(500, 0.0, 5.0, 42);

    println!("Generated {} observations", running_var.len());
    println!("True treatment effect: 5.0");
    println!();

    // --- Fit RDD Model ---
    println!("Fitting RDD Model:");
    let mut model = RDDModel::new(0.0, 20.0, Kernel::Triangular);

    match model.fit(&running_var, &outcome) {
        Ok(results) => {
            println!("  Treatment Effect (tau): {:.4}", results.tau);
            println!("  Standard Error: {:.4}", results.se);
            println!("  95% CI: [{:.4}, {:.4}]", results.ci_lower, results.ci_upper);
            println!("  t-statistic: {:.4}", results.t_stat);
            println!("  p-value: {:.6}", results.p_value);
            println!("  Bandwidth: {:.2}", results.bandwidth);
            println!("  N left: {}, N right: {}", results.n_left, results.n_right);
            println!();

            // Check significance
            if results.p_value < 0.05 {
                println!("  Result: SIGNIFICANT at 5% level");
            } else {
                println!("  Result: Not significant at 5% level");
            }
        }
        Err(e) => {
            println!("  Error: {}", e);
            return;
        }
    }

    // --- Try different kernels ---
    println!("\n--- Comparison across kernels ---");
    for kernel in [Kernel::Triangular, Kernel::Uniform, Kernel::Epanechnikov] {
        let mut m = RDDModel::new(0.0, 20.0, kernel);
        if let Ok(r) = m.fit(&running_var, &outcome) {
            println!("{:?}: tau = {:.4}, SE = {:.4}", kernel, r.tau, r.se);
        }
    }

    // --- Validation tests ---
    println!("\n--- Validation Tests ---");
    let validator = RDDValidator::new(0.0, 20.0, Kernel::Triangular);

    // Density test (McCrary)
    match validator.density_test(&running_var, 10) {
        Ok(density_result) => {
            println!("McCrary Density Test:");
            println!("  Log density difference: {:.4}", density_result.theta);
            println!("  p-value: {:.4}", density_result.p_value);
            if density_result.p_value > 0.05 {
                println!("  => No evidence of manipulation (p > 0.05)");
            } else {
                println!("  => Warning: Possible manipulation (p < 0.05)");
            }
        }
        Err(e) => println!("Density test error: {}", e),
    }

    // Placebo tests
    println!("\nPlacebo Tests at false cutoffs:");
    let placebos = validator.placebo_test(&[-30.0, 30.0], &running_var, &outcome);
    for p in placebos {
        if !p.tau.is_nan() {
            let sig = if p.p_value < 0.05 { "*" } else { "" };
            println!(
                "  Cutoff {:.0}: tau = {:.4}, p = {:.4} {}",
                p.cutoff, p.tau, p.p_value, sig
            );
        }
    }

    println!("\n=== Done ===");
}

/// Generate synthetic RDD data with known treatment effect.
fn generate_synthetic_rdd_data(
    n: usize,
    cutoff: f64,
    treatment_effect: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 2.0).unwrap();

    let mut running_var = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);

    for _ in 0..n {
        let x: f64 = rng.gen_range(-50.0..50.0);
        let treatment = if x >= cutoff { treatment_effect } else { 0.0 };
        let noise: f64 = normal.sample(&mut rng);
        let y = 10.0 + 0.1 * x + treatment + noise;

        running_var.push(x);
        outcome.push(y);
    }

    (running_var, outcome)
}
