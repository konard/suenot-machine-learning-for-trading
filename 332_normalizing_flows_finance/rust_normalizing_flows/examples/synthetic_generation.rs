//! Example: Generate synthetic financial data with normalizing flows
//!
//! This example demonstrates how to:
//! 1. Train a flow on historical returns
//! 2. Generate unlimited realistic synthetic scenarios
//! 3. Use synthetic data for Monte Carlo simulation
//! 4. Perform stress testing

use normalizing_flows_finance::prelude::*;
use normalizing_flows_finance::risk::metrics::{compute_risk_metrics, stress_test};
use normalizing_flows_finance::utils::returns::{generate_synthetic_returns, standardize};
use normalizing_flows_finance::utils::statistics::describe;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Synthetic Data Generation with Normalizing Flows");
    println!("═══════════════════════════════════════════════════════════");

    // Step 1: Prepare training data
    println!("\n[1] Preparing training data...");
    let original_returns = generate_synthetic_returns(500, 4.0); // Fat-tailed returns
    let orig_stats = describe(&original_returns);

    println!("    Original data:");
    println!("      Samples:  {}", orig_stats.count);
    println!("      Mean:     {:+.4}", orig_stats.mean);
    println!("      Std:      {:.4}", orig_stats.std);
    println!("      Kurtosis: {:.2}", orig_stats.kurtosis);

    let standardized = standardize(&original_returns);

    // Step 2: Train flow
    println!("\n[2] Training normalizing flow...");
    let config = FlowConfig::new(1).with_n_layers(6).with_hidden_dim(64);
    let mut flow = RealNVP::new(config);

    let _history = flow.train(&standardized, 80)?;
    println!("    Training complete!");

    // Step 3: Generate synthetic scenarios
    println!("\n[3] Generating synthetic scenarios...");

    for &n in &[1000, 10000, 100000] {
        let samples = flow.sample_flat(n);
        let stats = describe(&samples);
        println!(
            "    {:>7} samples: mean={:+.4}, std={:.4}, kurt={:.2}",
            n, stats.mean, stats.std, stats.kurtosis
        );
    }

    // Generate scaled back to original space
    let synthetic_100k_std = flow.sample_flat(100000);
    let synthetic_100k: Vec<f64> = synthetic_100k_std
        .iter()
        .map(|&x| x * orig_stats.std + orig_stats.mean)
        .collect();

    let synth_stats = describe(&synthetic_100k);
    println!("\n    Scaled to original space (100k samples):");
    println!("      Mean:     {:+.4}", synth_stats.mean);
    println!("      Std:      {:.4}", synth_stats.std);
    println!("      Kurtosis: {:.2}", synth_stats.kurtosis);

    // Step 4: Compare statistics
    println!("\n[4] Comparing distributions...");
    println!("\n    {:^15} {:>15} {:>15}", "Metric", "Original", "Synthetic");
    println!("    {}", "─".repeat(50));
    println!(
        "    {:^15} {:>15.4} {:>15.4}",
        "Mean", orig_stats.mean, synth_stats.mean
    );
    println!(
        "    {:^15} {:>15.4} {:>15.4}",
        "Std", orig_stats.std, synth_stats.std
    );
    println!(
        "    {:^15} {:>15.4} {:>15.4}",
        "Skewness", orig_stats.skewness, synth_stats.skewness
    );
    println!(
        "    {:^15} {:>15.4} {:>15.4}",
        "Kurtosis", orig_stats.kurtosis, synth_stats.kurtosis
    );
    println!(
        "    {:^15} {:>15.4} {:>15.4}",
        "Min", orig_stats.min, synth_stats.min
    );
    println!(
        "    {:^15} {:>15.4} {:>15.4}",
        "Max", orig_stats.max, synth_stats.max
    );

    // Step 5: Monte Carlo simulation
    println!("\n[5] Monte Carlo simulation for portfolio risk...");

    let initial_capital = 10000.0;
    let n_simulations = 1000;
    let n_days = 30;

    // Generate return paths
    let mut final_values = Vec::with_capacity(n_simulations);
    let mut rng_seed = 42u64;

    for _ in 0..n_simulations {
        // Generate 30 days of returns
        let path_returns: Vec<f64> = (0..n_days)
            .map(|_| {
                rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = (rng_seed as usize) % synthetic_100k.len();
                synthetic_100k[idx]
            })
            .collect();

        // Calculate cumulative return
        let cumulative: f64 = path_returns.iter().map(|r| (1.0 + r)).product();
        final_values.push(initial_capital * cumulative);
    }

    let final_stats = describe(&final_values);

    println!("\n    Monte Carlo Results ({} paths, {} days):", n_simulations, n_days);
    println!("      Initial capital:  ${:.0}", initial_capital);
    println!("      Mean final value: ${:.0}", final_stats.mean);
    println!("      Median:           ${:.0}", final_stats.median);
    println!("      5th percentile:   ${:.0}", final_stats.q1);
    println!("      95th percentile:  ${:.0}", final_stats.q3);
    println!("      Worst case:       ${:.0}", final_stats.min);
    println!("      Best case:        ${:.0}", final_stats.max);

    // Probability calculations
    let prob_loss = final_values.iter().filter(|&&v| v < initial_capital).count() as f64
        / final_values.len() as f64;
    let prob_loss_10 = final_values
        .iter()
        .filter(|&&v| v < initial_capital * 0.9)
        .count() as f64
        / final_values.len() as f64;
    let prob_gain_20 = final_values
        .iter()
        .filter(|&&v| v > initial_capital * 1.2)
        .count() as f64
        / final_values.len() as f64;

    println!("\n    Probability estimates:");
    println!("      P(any loss):     {:.1}%", prob_loss * 100.0);
    println!("      P(loss > 10%):   {:.1}%", prob_loss_10 * 100.0);
    println!("      P(gain > 20%):   {:.1}%", prob_gain_20 * 100.0);

    // Step 6: Extreme scenario analysis
    println!("\n[6] Extreme scenario analysis...");

    let threshold_1pct = synth_stats.q1; // Approximate 1st percentile
    let extreme_losses: Vec<f64> = synthetic_100k
        .iter()
        .filter(|&&x| x < threshold_1pct)
        .copied()
        .collect();

    println!(
        "    Extreme negative events (bottom 1%): {} occurrences",
        extreme_losses.len()
    );
    if !extreme_losses.is_empty() {
        let extreme_stats = describe(&extreme_losses);
        println!("      Average extreme loss: {:.2}%", extreme_stats.mean * 100.0);
        println!("      Worst loss:           {:.2}%", extreme_stats.min * 100.0);
    }

    // Step 7: Stress testing
    println!("\n[7] Stress testing scenarios...");

    let scenarios = ["2x_vol", "crash"];
    for scenario in scenarios {
        let result = stress_test(&flow, scenario, 50000);
        println!("\n    Scenario: {}", scenario);
        println!(
            "      95% VaR: {:.4} -> {:.4}",
            result.base_var_95, result.stressed_var_95
        );
        println!(
            "      Max loss: {:.4} -> {:.4}",
            result.max_loss_base, result.max_loss_stressed
        );
    }

    // Step 8: Risk metrics summary
    println!("\n[8] Flow-based risk metrics...");
    let metrics = compute_risk_metrics(&flow, 100000);

    println!("    VaR 95%:   {:.4}", metrics.var_95);
    println!("    VaR 99%:   {:.4}", metrics.var_99);
    println!("    CVaR 95%:  {:.4}", metrics.cvar_95);
    println!("    CVaR 99%:  {:.4}", metrics.cvar_99);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  Capabilities demonstrated:");
    println!("  1. Generate unlimited realistic return scenarios");
    println!("  2. Match historical distribution characteristics");
    println!("  3. Monte Carlo simulation for portfolio risk");
    println!("  4. Stress testing under extreme conditions");
    println!();
    println!("  Applications:");
    println!("  - Backtesting with more data than available");
    println!("  - Regulatory stress testing (Basel III)");
    println!("  - Risk budgeting and allocation");
    println!("  - Options pricing with realistic distributions");
    println!();

    Ok(())
}
