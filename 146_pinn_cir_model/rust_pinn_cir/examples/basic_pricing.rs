//! Basic CIR Pricing Example
//!
//! Demonstrates:
//! - Analytical bond pricing
//! - Yield curve generation
//! - Path simulation
//! - Feller condition checking
//! - Credit risk (survival probability)

use pinn_cir_model::*;

fn main() {
    println!("=== CIR Model: Basic Pricing Example ===\n");

    // Define CIR parameters
    let params = CIRParams {
        kappa: 0.5,   // mean-reversion speed
        theta: 0.05,  // long-run mean (5%)
        sigma: 0.1,   // volatility
        r0: 0.03,     // initial rate (3%)
    };

    // Check Feller condition
    print_params_summary(&params);

    // 1. Bond pricing at different rates
    println!("\n--- Bond Prices (T = 5 years) ---");
    for r in [0.01, 0.03, 0.05, 0.07, 0.10, 0.15] {
        let price = cir_bond_price(r, 5.0, &params);
        let yield_val = cir_yield(r, 5.0, &params);
        println!(
            "  r = {:.1}%: P = {:.6}, y = {:.4}%",
            r * 100.0,
            price,
            yield_val * 100.0
        );
    }

    // 2. Yield curve
    let maturities = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
    print_yield_curve(params.r0, &maturities, &params);

    // 3. Simulate paths
    println!("\n--- Path Simulation (Euler) ---");
    let n_paths = 5;
    let paths = simulate_cir_euler(&params, 5.0, 1260, n_paths);

    for j in 0..n_paths {
        let final_rate = paths[[j, 1260]];
        let min_rate = (0..=1260).map(|i| paths[[j, i]]).fold(f64::INFINITY, f64::min);
        let max_rate = (0..=1260).map(|i| paths[[j, i]]).fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  Path {}: final={:.4}%, min={:.4}%, max={:.4}%",
            j + 1,
            final_rate * 100.0,
            min_rate * 100.0,
            max_rate * 100.0
        );
    }

    // 4. Conditional moments
    println!("\n--- Conditional Moments ---");
    for t in [1.0, 2.0, 5.0, 10.0] {
        let mean = cir_conditional_mean(params.r0, t, &params);
        let var = cir_conditional_variance(params.r0, t, &params);
        println!(
            "  t = {:.0}y: E[r] = {:.4}%, Std[r] = {:.4}%",
            t,
            mean * 100.0,
            var.sqrt() * 100.0
        );
    }

    // 5. Credit risk application
    println!("\n--- Credit Risk (CIR Intensity) ---");
    let intensity_params = CIRParams {
        kappa: 0.3,
        theta: 0.02,
        sigma: 0.08,
        r0: 0.015, // initial default intensity
    };

    for t in [1.0, 3.0, 5.0, 10.0] {
        let q = survival_probability(intensity_params.r0, t, &intensity_params);
        let spread = cds_spread(
            intensity_params.r0,
            t,
            &intensity_params,
            0.4,  // recovery
            0.03, // risk-free rate
            4,    // quarterly
        );
        println!(
            "  T = {:.0}y: Survival = {:.4}, CDS = {:.1} bps",
            t,
            q,
            spread * 10000.0
        );
    }

    // 6. Compare Feller-satisfied vs violated
    println!("\n--- Feller Condition Comparison ---");

    let satisfied = CIRParams {
        kappa: 0.5,
        theta: 0.05,
        sigma: 0.1,
        r0: 0.03,
    };
    let violated = CIRParams {
        kappa: 0.5,
        theta: 0.01,
        sigma: 0.5,
        r0: 0.03,
    };

    println!(
        "  Satisfied (kappa={}, theta={}, sigma={}): margin = {:.4}",
        satisfied.kappa,
        satisfied.theta,
        satisfied.sigma,
        satisfied.feller_margin()
    );
    println!(
        "  Violated  (kappa={}, theta={}, sigma={}): margin = {:.4}",
        violated.kappa,
        violated.theta,
        violated.sigma,
        violated.feller_margin()
    );

    println!("\n=== Done ===");
}
