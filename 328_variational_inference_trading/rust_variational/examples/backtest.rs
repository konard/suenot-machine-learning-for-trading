//! Example: Backtest trading strategy
//!
//! This example demonstrates how to backtest the variational inference
//! trading strategy on historical data.

use variational_inference_trading::prelude::*;
use ndarray::Array1;
use rand::Rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Backtest Demo ===\n");

    // Create VAE model
    let config = VAEConfig::new(13, 8, vec![64, 32])
        .with_beta(2.0)
        .with_sequence_length(60);

    let vae = VAE::new(config.clone());

    // Create signal generator
    let signal_gen = SignalGenerator::new(0.6)
        .with_uncertainty_threshold(0.05)
        .with_position_limits(0.01, 0.1);

    // Create backtest engine
    let mut engine = BacktestEngine::new(10000.0)
        .with_commission(0.001)
        .with_slippage(0.0005)
        .with_stop_loss(0.02)
        .with_take_profit(0.04);

    println!("Backtest Configuration:");
    println!("  Initial capital: $10,000");
    println!("  Commission:      0.1%");
    println!("  Slippage:        0.05%");
    println!("  Stop loss:       2%");
    println!("  Take profit:     4%");

    // Generate synthetic market data
    println!("\nGenerating synthetic market data...");
    let n_periods = 500;
    let mut rng = rand::thread_rng();
    let input_size = config.input_dim * config.sequence_length;

    // Generate random walk prices
    let mut prices = Vec::with_capacity(n_periods);
    let mut price = 1000.0;
    for _ in 0..n_periods {
        price *= 1.0 + rng.gen_range(-0.01..0.01);  // 1% random walk
        prices.push(price);
    }

    // Generate timestamps
    let timestamps: Vec<i64> = (0..n_periods as i64)
        .map(|i| i * 60000)  // 1 minute intervals
        .collect();

    // Generate signals
    println!("Generating signals for {} periods...", n_periods);
    let signals: Vec<Signal> = (0..n_periods)
        .map(|_| {
            let features = Array1::from_vec(
                (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect()
            );
            let prediction = vae.predict(&features, 50);
            signal_gen.generate(&prediction)
        })
        .collect();

    // Run backtest
    println!("Running backtest...\n");
    let report = engine.run(&signals, &prices, &timestamps, "SYNTH/USDT");

    // Display results
    println!("{}", report.summary());

    // Additional analysis
    println!("\n--- Additional Metrics ---");
    println!("Volatility (annualized): {:.2}%", report.volatility * 100.0);
    println!("Max consecutive wins:    {}", report.max_consecutive_wins);
    println!("Max consecutive losses:  {}", report.max_consecutive_losses);

    // Equity curve summary
    if !report.equity_curve.is_empty() {
        let start_equity = report.equity_curve.first().unwrap().1;
        let end_equity = report.equity_curve.last().unwrap().1;
        let high_equity = report.equity_curve.iter()
            .map(|e| e.1)
            .fold(f64::NEG_INFINITY, f64::max);
        let low_equity = report.equity_curve.iter()
            .map(|e| e.1)
            .fold(f64::INFINITY, f64::min);

        println!("\n--- Equity Curve ---");
        println!("Start:   ${:.2}", start_equity);
        println!("End:     ${:.2}", end_equity);
        println!("High:    ${:.2}", high_equity);
        println!("Low:     ${:.2}", low_equity);
    }

    // Return distribution
    if !report.returns.is_empty() {
        let positive_returns = report.returns.iter().filter(|&&r| r > 0.0).count();
        let negative_returns = report.returns.iter().filter(|&&r| r < 0.0).count();
        let zero_returns = report.returns.len() - positive_returns - negative_returns;

        println!("\n--- Return Distribution ---");
        println!("Positive periods: {} ({:.1}%)",
                 positive_returns,
                 positive_returns as f64 / report.returns.len() as f64 * 100.0);
        println!("Negative periods: {} ({:.1}%)",
                 negative_returns,
                 negative_returns as f64 / report.returns.len() as f64 * 100.0);
        println!("Neutral periods:  {} ({:.1}%)",
                 zero_returns,
                 zero_returns as f64 / report.returns.len() as f64 * 100.0);
    }

    println!("\n=== Backtest Complete ===");

    Ok(())
}
