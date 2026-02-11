//! Synthetic Heston Model Data Generation Example
//!
//! Demonstrates generating synthetic financial time series using the
//! Heston stochastic volatility model with various parameter configurations
//! to simulate different market regimes.
//!
//! Run with:
//!   cargo run --example synthetic_heston

use physics_constrained_gan::{
    detect_regime,
    heston::{self, HestonParams},
    returns_to_prices,
    statistics,
    trading,
};

fn main() {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║   Heston Model - Synthetic Data Generation           ║");
    println!("╚═══════════════════════════════════════════════════════╝");

    let seq_len = 500;
    let n_paths = 200;

    // Define regime-specific parameters
    let regimes = vec![
        ("Bull Market", HestonParams {
            mu: 0.15, kappa: 2.0, theta: 0.03, xi: 0.2, rho: -0.5, v0: 0.03,
            dt: 1.0 / 252.0,
        }),
        ("Bear Market", HestonParams {
            mu: -0.10, kappa: 1.5, theta: 0.06, xi: 0.4, rho: -0.8, v0: 0.06,
            dt: 1.0 / 252.0,
        }),
        ("Sideways", HestonParams {
            mu: 0.02, kappa: 3.0, theta: 0.02, xi: 0.15, rho: -0.3, v0: 0.02,
            dt: 1.0 / 252.0,
        }),
        ("Crisis", HestonParams {
            mu: -0.30, kappa: 1.0, theta: 0.10, xi: 0.6, rho: -0.9, v0: 0.12,
            dt: 1.0 / 252.0,
        }),
    ];

    for (name, params) in &regimes {
        println!("\n--- {} ---", name);
        println!("  params: mu={:.3}, kappa={:.3}, theta={:.4}, xi={:.3}, rho={:.3}",
                 params.mu, params.kappa, params.theta, params.xi, params.rho);

        let paths = heston::generate_multiple_paths(params, n_paths, seq_len);
        let all_returns: Vec<f64> = paths.iter().flatten().cloned().collect();
        let stats = statistics::compute_statistics(&all_returns);

        println!("  Mean return (daily): {:.6}", stats.mean);
        println!("  Volatility (daily):  {:.6}", stats.std);
        println!("  Skewness:            {:.4}", stats.skewness);
        println!("  Excess kurtosis:     {:.4}", stats.kurtosis);
        println!("  Leverage corr:       {:.4}", stats.leverage_corr);
        println!("  ACF(|r|, lag=1):     {:.4}", stats.acf_abs_lag1);

        // Regime detection accuracy
        let mut correct = 0;
        let expected_regime = match *name {
            "Bull Market" => physics_constrained_gan::MarketRegime::Bull,
            "Bear Market" => physics_constrained_gan::MarketRegime::Bear,
            "Sideways" => physics_constrained_gan::MarketRegime::Sideways,
            "Crisis" => physics_constrained_gan::MarketRegime::Crisis,
            _ => physics_constrained_gan::MarketRegime::Sideways,
        };

        for path in &paths {
            let (detected, _) = detect_regime(path);
            if detected == expected_regime {
                correct += 1;
            }
        }
        println!("  Regime detection accuracy: {:.1}%",
                 100.0 * correct as f64 / n_paths as f64);

        // Price path summary
        let first_path = &paths[0];
        let prices = returns_to_prices(first_path, 50000.0);
        println!("  Sample path: {:.2} -> {:.2} ({:+.2}%)",
                 prices[0],
                 prices[prices.len() - 1],
                 (prices[prices.len() - 1] / prices[0] - 1.0) * 100.0);

        // Monte Carlo trading signal
        let signal = trading::generate_signal_from_paths(&paths, 0.3);
        println!("  MC Signal: {} (confidence: {:.2}, E[r]: {:.4})",
                 signal.direction, signal.confidence, signal.expected_return);
    }

    // ---- Compare all regimes side by side ----
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║   Regime Comparison                                           ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ {:10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} ║",
             "Regime", "Mean", "Std", "Kurt", "Skew", "Lev", "ACF|r|");
    println!("╠════════════════════════════════════════════════════════════════╣");

    for (name, params) in &regimes {
        let paths = heston::generate_multiple_paths(params, 100, seq_len);
        let flat: Vec<f64> = paths.iter().flatten().cloned().collect();
        let s = statistics::compute_statistics(&flat);
        println!("║ {:10} {:>8.5} {:>8.5} {:>8.2} {:>8.3} {:>8.3} {:>8.3} ║",
                 name, s.mean, s.std, s.kurtosis, s.skewness, s.leverage_corr, s.acf_abs_lag1);
    }
    println!("╚════════════════════════════════════════════════════════════════╝");
}
