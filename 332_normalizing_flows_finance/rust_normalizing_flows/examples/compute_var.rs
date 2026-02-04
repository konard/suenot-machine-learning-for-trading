//! Example: Compute VaR and CVaR using normalizing flows
//!
//! This example demonstrates how to:
//! 1. Train a flow on return data
//! 2. Compute VaR at multiple confidence levels
//! 3. Compute CVaR (Expected Shortfall)
//! 4. Compare with Gaussian and historical VaR

use normalizing_flows_finance::prelude::*;
use normalizing_flows_finance::risk::metrics::{
    compare_distributions, compute_cvar, compute_risk_metrics, compute_var_multiple, stress_test,
};
use normalizing_flows_finance::utils::returns::{generate_synthetic_returns, standardize};
use normalizing_flows_finance::utils::statistics::{describe, percentile};

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("═══════════════════════════════════════════════════════════");
    println!("  VaR and CVaR Calculation with Normalizing Flows");
    println!("═══════════════════════════════════════════════════════════");

    // Step 1: Prepare data
    println!("\n[1] Preparing data...");
    let returns = generate_synthetic_returns(1000, 4.0); // Fat-tailed
    let stats = describe(&returns);

    println!("    Samples:  {}", stats.count);
    println!("    Mean:     {:+.4}", stats.mean);
    println!("    Std:      {:.4}", stats.std);
    println!("    Kurtosis: {:.2} (Gaussian = 0)", stats.kurtosis);

    // Standardize
    let standardized = standardize(&returns);

    // Step 2: Train flow
    println!("\n[2] Training normalizing flow...");
    let config = FlowConfig::new(1).with_n_layers(6).with_hidden_dim(64);
    let mut flow = RealNVP::new(config);

    let _history = flow.train(&standardized, 100)?;
    println!("    Training complete!");

    // Step 3: Compute VaR using different methods
    println!("\n[3] Computing VaR at different confidence levels...");
    println!("\n    {:^15} {:>12} {:>12} {:>12}", "Method", "VaR 99%", "VaR 95%", "VaR 90%");
    println!("    {}", "─".repeat(55));

    // Gaussian VaR
    let gauss_var_99 = stats.mean + stats.std * (-2.326); // z for 1%
    let gauss_var_95 = stats.mean + stats.std * (-1.645); // z for 5%
    let gauss_var_90 = stats.mean + stats.std * (-1.282); // z for 10%
    println!(
        "    {:^15} {:>12.4} {:>12.4} {:>12.4}",
        "Gaussian", gauss_var_99, gauss_var_95, gauss_var_90
    );

    // Historical VaR
    let hist_var_99 = percentile(&returns, 0.01);
    let hist_var_95 = percentile(&returns, 0.05);
    let hist_var_90 = percentile(&returns, 0.10);
    println!(
        "    {:^15} {:>12.4} {:>12.4} {:>12.4}",
        "Historical", hist_var_99, hist_var_95, hist_var_90
    );

    // Flow VaR (need to scale back from standardized)
    let std_stats = describe(&standardized);
    let flow_vars = compute_var_multiple(&flow, &[0.01, 0.05, 0.10], 100000);

    // Scale back to original space
    let flow_var_99 = flow_vars.get("VaR_99").unwrap() * stats.std + stats.mean;
    let flow_var_95 = flow_vars.get("VaR_95").unwrap() * stats.std + stats.mean;
    let flow_var_90 = flow_vars.get("VaR_90").unwrap() * stats.std + stats.mean;
    println!(
        "    {:^15} {:>12.4} {:>12.4} {:>12.4}",
        "Flow", flow_var_99, flow_var_95, flow_var_90
    );

    // Step 4: Compute CVaR
    println!("\n[4] Computing CVaR (Expected Shortfall)...");
    println!("\n    {:^15} {:>12} {:>12}", "Method", "CVaR 95%", "CVaR 99%");
    println!("    {}", "─".repeat(45));

    // Gaussian CVaR (closed form)
    let phi_95 = 0.1031; // pdf(z) at z = -1.645
    let phi_99 = 0.0267; // pdf(z) at z = -2.326
    let gauss_cvar_95 = stats.mean - stats.std * phi_95 / 0.05;
    let gauss_cvar_99 = stats.mean - stats.std * phi_99 / 0.01;
    println!(
        "    {:^15} {:>12.4} {:>12.4}",
        "Gaussian", gauss_cvar_95, gauss_cvar_99
    );

    // Historical CVaR
    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let hist_cvar_95: f64 = sorted[..(n as f64 * 0.05) as usize].iter().sum::<f64>()
        / ((n as f64 * 0.05) as usize) as f64;
    let hist_cvar_99: f64 = sorted[..(n as f64 * 0.01) as usize].iter().sum::<f64>()
        / ((n as f64 * 0.01) as usize).max(1) as f64;
    println!(
        "    {:^15} {:>12.4} {:>12.4}",
        "Historical", hist_cvar_95, hist_cvar_99
    );

    // Flow CVaR
    let (_, flow_cvar_95_std) = compute_cvar(&flow, 0.05, 100000);
    let (_, flow_cvar_99_std) = compute_cvar(&flow, 0.01, 100000);
    let flow_cvar_95 = flow_cvar_95_std * stats.std + stats.mean;
    let flow_cvar_99 = flow_cvar_99_std * stats.std + stats.mean;
    println!(
        "    {:^15} {:>12.4} {:>12.4}",
        "Flow", flow_cvar_95, flow_cvar_99
    );

    // Step 5: Comprehensive risk metrics
    println!("\n[5] Comprehensive risk metrics from flow...");
    let metrics = compute_risk_metrics(&flow, 100000);

    println!("\n    Statistic          Value");
    println!("    {}", "─".repeat(35));
    println!("    Mean:              {:+.4}", metrics.mean);
    println!("    Std:               {:.4}", metrics.std);
    println!("    Skewness:          {:+.4}", metrics.skewness);
    println!("    Kurtosis:          {:+.4}", metrics.kurtosis);
    println!("    VaR 90%:           {:+.4}", metrics.var_90);
    println!("    VaR 95%:           {:+.4}", metrics.var_95);
    println!("    VaR 99%:           {:+.4}", metrics.var_99);
    println!("    CVaR 90%:          {:+.4}", metrics.cvar_90);
    println!("    CVaR 95%:          {:+.4}", metrics.cvar_95);
    println!("    CVaR 99%:          {:+.4}", metrics.cvar_99);

    // Step 6: Distribution comparison
    println!("\n[6] Comparing flow samples to original data...");
    let flow_samples = flow.sample_flat(10000);
    let comparison = compare_distributions(&standardized, &flow_samples);

    println!("    Mean difference:   {:+.4}", comparison.mean_diff);
    println!("    Std difference:    {:+.4}", comparison.std_diff);
    println!("    KS statistic:      {:.4}", comparison.ks_statistic);

    // Step 7: Stress testing
    println!("\n[7] Stress testing...");

    for scenario in &["2x_vol", "crash"] {
        let result = stress_test(&flow, scenario, 100000);
        println!("\n    Scenario: {}", scenario);
        println!("      Base VaR 95%:     {:+.4}", result.base_var_95);
        println!("      Stressed VaR 95%: {:+.4}", result.stressed_var_95);
        println!("      Base max loss:    {:+.4}", result.max_loss_base);
        println!("      Stressed max:     {:+.4}", result.max_loss_stressed);
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Key Findings:");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  1. Gaussian VaR typically UNDERESTIMATES tail risk");
    println!("  2. Flow-based VaR captures true fat-tailed distribution");
    println!("  3. CVaR provides additional information about tail shape");
    println!("  4. Stress testing helps prepare for extreme scenarios");
    println!();

    Ok(())
}
