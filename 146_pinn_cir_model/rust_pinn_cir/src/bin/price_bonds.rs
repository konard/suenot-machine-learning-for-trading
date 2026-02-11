//! CIR Bond Pricing and Analysis
//!
//! Demonstrates analytical CIR bond pricing, yield curves, forward rates,
//! and credit risk calculations.

use clap::Parser;
use colored::*;
use pinn_cir_model::*;

#[derive(Parser, Debug)]
#[command(name = "CIR Bond Pricer")]
#[command(about = "Price bonds and compute yield curves using the CIR model")]
struct Args {
    /// Mean-reversion speed
    #[arg(long, default_value = "0.5")]
    kappa: f64,

    /// Long-run mean rate
    #[arg(long, default_value = "0.05")]
    theta: f64,

    /// Volatility coefficient
    #[arg(long, default_value = "0.1")]
    sigma: f64,

    /// Current short rate
    #[arg(long, default_value = "0.03")]
    rate: f64,

    /// Show credit risk analysis
    #[arg(long)]
    credit: bool,

    /// Number of simulation paths
    #[arg(long, default_value = "1000")]
    n_paths: usize,
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    println!("{}", "=".repeat(70).blue());
    println!("{}", "CIR Bond Pricing & Analysis".bold().blue());
    println!("{}", "=".repeat(70).blue());

    let params = CIRParams {
        kappa: args.kappa,
        theta: args.theta,
        sigma: args.sigma,
        r0: args.rate,
    };

    print_params_summary(&params);

    // =========================================================================
    // 1. Bond Prices at Various Rates and Maturities
    // =========================================================================
    println!("\n{}", "Bond Prices P(r, tau)".bold().cyan());
    println!("{}", "-".repeat(70));

    let rates = vec![0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15];
    let maturities = vec![0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0];

    // Header
    print!("{:>8}", "r \\ tau");
    for &tau in &maturities {
        print!("{:>10.1}y", tau);
    }
    println!();
    println!("{}", "-".repeat(8 + maturities.len() * 10));

    for &r in &rates {
        print!("{:>7.1}%", r * 100.0);
        for &tau in &maturities {
            let price = cir_bond_price(r, tau, &params);
            print!("{:>10.6}", price);
        }
        println!();
    }

    // =========================================================================
    // 2. Yield Curve
    // =========================================================================
    let yield_maturities = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
    print_yield_curve(args.rate, &yield_maturities, &params);

    // Long-run yield
    let gamma = params.gamma();
    let y_inf = 2.0 * params.kappa * params.theta / (params.kappa + gamma);
    println!("\nLong-run yield (tau -> inf): {:.4}%", y_inf * 100.0);

    // =========================================================================
    // 3. Yield Curves at Multiple Rates
    // =========================================================================
    println!("\n{}", "Yield Curves at Different Rates".bold().cyan());
    println!("{}", "-".repeat(70));

    let curve_rates = vec![0.01, 0.03, 0.05, 0.07, 0.10];
    let curve_maturities = vec![0.5, 1.0, 2.0, 5.0, 10.0, 30.0];

    print!("{:>8}", "r \\ tau");
    for &tau in &curve_maturities {
        print!("{:>10.1}y", tau);
    }
    println!();
    println!("{}", "-".repeat(8 + curve_maturities.len() * 10));

    for &r in &curve_rates {
        print!("{:>7.1}%", r * 100.0);
        let yields = generate_yield_curve(r, &curve_maturities, &params);
        for y in &yields {
            print!("{:>9.3}%", y * 100.0);
        }
        println!();
    }

    // =========================================================================
    // 4. Path Simulation Statistics
    // =========================================================================
    println!("\n{}", "Path Simulation".bold().cyan());
    println!("{}", "-".repeat(70));

    let n_steps = 2520; // ~10 years daily
    let t_final = 10.0;

    println!(
        "Simulating {} paths over {} years ({} steps)...",
        args.n_paths, t_final, n_steps
    );

    let paths = simulate_cir_euler(&params, t_final, n_steps, args.n_paths);

    // Statistics
    let mut min_rate = f64::INFINITY;
    let mut max_rate = f64::NEG_INFINITY;
    let mut sum_final = 0.0;
    let mut n_zero = 0usize;

    for j in 0..args.n_paths {
        for i in 0..=n_steps {
            let r = paths[[j, i]];
            min_rate = min_rate.min(r);
            max_rate = max_rate.max(r);
            if r < 1e-8 {
                n_zero += 1;
            }
        }
        sum_final += paths[[j, n_steps]];
    }

    let mean_final = sum_final / args.n_paths as f64;

    println!("  Min rate:         {:.6} ({:.3}%)", min_rate, min_rate * 100.0);
    println!("  Max rate:         {:.6} ({:.3}%)", max_rate, max_rate * 100.0);
    println!(
        "  Mean final rate:  {:.6} ({:.3}%) [theta = {:.3}%]",
        mean_final,
        mean_final * 100.0,
        params.theta * 100.0
    );
    println!(
        "  Near-zero events: {} ({:.2}%)",
        n_zero,
        n_zero as f64 / (args.n_paths * (n_steps + 1)) as f64 * 100.0
    );

    // Conditional moments check
    let mean_expected = cir_conditional_mean(params.r0, t_final, &params);
    let var_expected = cir_conditional_variance(params.r0, t_final, &params);
    println!("\n  Theoretical E[r(T)]  = {:.6}", mean_expected);
    println!("  Simulated  E[r(T)]  = {:.6}", mean_final);
    println!("  Theoretical Var[r(T)] = {:.8}", var_expected);

    // =========================================================================
    // 5. Credit Risk (optional)
    // =========================================================================
    if args.credit {
        println!("\n{}", "=".repeat(70).green());
        println!("{}", "Credit Risk Analysis".bold().green());
        println!("{}", "=".repeat(70).green());

        let intensity_params = CIRParams {
            kappa: 0.3,
            theta: 0.02,
            sigma: 0.08,
            r0: 0.015,
        };

        println!("\nDefault Intensity Parameters:");
        print_params_summary(&intensity_params);

        // Survival probabilities
        println!("\n{}", "Survival Probabilities:".bold());
        let cds_maturities = vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
        println!(
            "{:>10} {:>12} {:>12}",
            "Maturity", "Survival Q", "Default PD"
        );
        println!("{}", "-".repeat(36));

        for &t in &cds_maturities {
            let q = survival_probability(intensity_params.r0, t, &intensity_params);
            let pd = 1.0 - q;
            println!("{:>10.1}y {:>12.6} {:>11.4}%", t, q, pd * 100.0);
        }

        // CDS spreads
        println!("\n{}", "CDS Spreads:".bold());
        println!(
            "{:>10} {:>12}",
            "Maturity", "Spread (bps)"
        );
        println!("{}", "-".repeat(24));

        for &t in &cds_maturities {
            let spread = cds_spread(
                intensity_params.r0,
                t,
                &intensity_params,
                0.4,  // recovery
                0.03, // risk-free rate
                4,    // quarterly payments
            );
            println!("{:>10.1}y {:>12.1}", t, spread * 10000.0);
        }
    }

    println!("\n{}", "Done.".green());
}
