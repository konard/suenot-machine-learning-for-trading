//! Funding Rate RDD Example.
//!
//! Demonstrates RDD analysis on cryptocurrency perpetual futures
//! funding rates using Bybit data.
//!
//! In perpetual futures, extreme funding rates often lead to
//! mean reversion as leveraged positions get squeezed.

use rdd_trading::backtest::engine::BacktestEngine;
use rdd_trading::data::bybit::{generate_synthetic_data, generate_synthetic_funding_rates};
use rdd_trading::model::rdd::{Kernel, RDDModel};
use rdd_trading::trading::strategy::RDDStrategy;

fn main() {
    println!("=== Funding Rate RDD Trading Example ===\n");

    // Generate synthetic data (in production, use BybitClient)
    println!("Generating synthetic funding rate data...");
    let funding_rates = generate_synthetic_funding_rates("BTCUSDT", 500, 42);
    let market_data = generate_synthetic_data("BTCUSDT", 500 * 3, 50000.0, 0.01, 42); // 3 candles per funding

    println!("Funding rate observations: {}", funding_rates.len());

    // --- Analyze funding rate distribution ---
    let rates: Vec<f64> = funding_rates.iter().map(|f| f.funding_rate_annualized).collect();
    let mean_rate = rates.iter().sum::<f64>() / rates.len() as f64;
    let extreme_positive = rates.iter().filter(|&&r| r > 50.0).count();
    let extreme_negative = rates.iter().filter(|&&r| r < -50.0).count();

    println!("\nFunding Rate Distribution (Annualized %):");
    println!("  Mean: {:.2}%", mean_rate);
    println!(
        "  Range: {:.2}% to {:.2}%",
        rates.iter().cloned().fold(f64::INFINITY, f64::min),
        rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "  Extreme positive (>50%): {} ({:.1}%)",
        extreme_positive,
        extreme_positive as f64 / rates.len() as f64 * 100.0
    );
    println!(
        "  Extreme negative (<-50%): {} ({:.1}%)",
        extreme_negative,
        extreme_negative as f64 / rates.len() as f64 * 100.0
    );

    // --- Calculate forward returns after funding ---
    // Each funding period is 8 hours, so we look at next-period returns
    let prices = market_data.close_prices();
    let mut forward_returns = Vec::with_capacity(funding_rates.len());

    for i in 0..funding_rates.len() {
        let price_idx = i * 3; // 3 hourly candles per 8-hour funding
        if price_idx + 3 < prices.len() && prices[price_idx] > 0.0 {
            let ret = (prices[price_idx + 3] - prices[price_idx]) / prices[price_idx];
            forward_returns.push(ret);
        } else {
            forward_returns.push(f64::NAN);
        }
    }

    // Filter valid observations
    let valid_obs: Vec<(f64, f64)> = rates
        .iter()
        .zip(forward_returns.iter())
        .filter(|(_, f)| !f.is_nan())
        .map(|(&r, &f)| (r, f))
        .collect();

    let running: Vec<f64> = valid_obs.iter().map(|(r, _)| *r).collect();
    let outcomes: Vec<f64> = valid_obs.iter().map(|(_, f)| *f).collect();

    println!("\nValid observations for RDD: {}", valid_obs.len());

    // --- RDD Analysis at +50% annualized funding ---
    println!("\n=== RDD Analysis: Funding Rate = +50% (Crowded Long) ===");

    let mut model_positive = RDDModel::new(50.0, 20.0, Kernel::Triangular);
    match model_positive.fit(&running, &outcomes) {
        Ok(results) => {
            println!("RDD Results:");
            println!("  Treatment Effect: {:.4}% ({:.4}%)", results.tau * 100.0, results.se * 100.0);
            println!("  95% CI: [{:.4}%, {:.4}%]", results.ci_lower * 100.0, results.ci_upper * 100.0);
            println!("  p-value: {:.4}", results.p_value);
            println!("  N left: {}, N right: {}", results.n_left, results.n_right);

            if results.tau < 0.0 && results.p_value < 0.1 {
                println!("  => Mean reversion detected: High funding predicts negative returns");
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    // --- RDD Analysis at -50% annualized funding ---
    println!("\n=== RDD Analysis: Funding Rate = -50% (Crowded Short) ===");

    let mut model_negative = RDDModel::new(-50.0, 20.0, Kernel::Triangular);
    match model_negative.fit(&running, &outcomes) {
        Ok(results) => {
            println!("RDD Results:");
            println!("  Treatment Effect: {:.4}% ({:.4}%)", results.tau * 100.0, results.se * 100.0);
            println!("  95% CI: [{:.4}%, {:.4}%]", results.ci_lower * 100.0, results.ci_upper * 100.0);
            println!("  p-value: {:.4}", results.p_value);
            println!("  N left: {}, N right: {}", results.n_left, results.n_right);

            if results.tau > 0.0 && results.p_value < 0.1 {
                println!("  => Mean reversion detected: Low funding predicts positive returns");
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    // --- Backtest Funding Rate Strategy ---
    println!("\n=== Backtesting Funding Rate Strategy ===");

    let mut strategy = RDDStrategy::funding_rate(50.0)
        .with_min_observations(30)
        .with_holding_period(3); // Hold for next funding period

    match strategy.fit(&running, &outcomes) {
        Ok(results) => {
            println!("Strategy fitted successfully");
            println!("  Effect size: {:.4}%", results.tau * 100.0);
            println!("  p-value: {:.4}", results.p_value);

            // Create signals based on funding rates
            let signals = strategy.generate_signals(&rates);
            let actionable = signals.iter().filter(|s| s.is_actionable()).count();
            println!("  Actionable signals: {} / {}", actionable, signals.len());

            // Simple backtest on 8-hourly prices
            let hourly_prices_subset: Vec<f64> = prices.iter().step_by(3).cloned().collect();
            let signals_subset: Vec<_> = signals.into_iter().take(hourly_prices_subset.len()).collect();

            let engine = BacktestEngine::new(100_000.0, 0.001)
                .with_slippage(0.0003)
                .with_max_position(0.2)
                .with_bars_per_year(365.0 * 3.0); // 3 funding periods per day

            match engine.run(&hourly_prices_subset, &signals_subset, 1) {
                Ok(results) => {
                    println!("\n{}", results.summary());
                }
                Err(e) => println!("Backtest error: {}", e),
            }
        }
        Err(e) => println!("Strategy fitting error: {}", e),
    }

    // --- Trading Implications ---
    println!("\n=== Trading Implications ===");
    println!("Funding Rate RDD Strategy:");
    println!("1. Monitor funding rate approaching +/-50% annualized");
    println!("2. When funding > 50%, anticipate short squeeze -> short bias");
    println!("3. When funding < -50%, anticipate long squeeze -> long bias");
    println!("4. Size positions inversely to distance from threshold");
    println!("5. Hold for 1 funding period (8 hours)");
    println!();
    println!("Risk Management:");
    println!("- Funding rate mean reversion is not guaranteed");
    println!("- Extreme funding can persist during strong trends");
    println!("- Use stop losses and position sizing");

    println!("\n=== Done ===");
}
