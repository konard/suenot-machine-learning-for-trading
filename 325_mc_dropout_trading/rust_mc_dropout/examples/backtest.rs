//! Example: Backtesting MC Dropout Trading Strategy
//!
//! This example demonstrates how to backtest a trading strategy
//! using MC Dropout for uncertainty estimation.

use mc_dropout_trading::prelude::*;
use ndarray::Array1;
use rand::Rng;

fn main() {
    println!("=== MC Dropout Backtest Example ===\n");

    // Generate synthetic price data
    let n_samples = 500;
    let mut rng = rand::thread_rng();

    println!("Generating synthetic market data ({} samples)...\n", n_samples);

    // Generate features and prices
    let mut prices: Vec<f64> = Vec::with_capacity(n_samples);
    let mut features: Vec<Array1<f64>> = Vec::with_capacity(n_samples);

    let mut price = 100.0;
    for i in 0..n_samples {
        // Random features
        let feature_vec: Vec<f64> = (0..20).map(|_| rng.gen_range(-1.0..1.0)).collect();
        features.push(Array1::from_vec(feature_vec));

        // Price with some signal from features and noise
        let signal = features[i][0] * 0.002 + features[i][1] * 0.001;
        let noise = rng.gen_range(-0.01..0.01);
        price *= 1.0 + signal + noise;
        prices.push(price);
    }

    println!("Price range: ${:.2} - ${:.2}",
             prices.iter().cloned().fold(f64::INFINITY, f64::min),
             prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // Create model
    let model_config = MCDropoutConfig::new(20)
        .with_hidden_dims(vec![64, 32])
        .with_dropout_rate(0.15)
        .with_mc_samples(30);

    let model = MCDropoutModel::new(model_config);

    // Create signal generator
    let signal_config = SignalConfig {
        confidence_threshold: 1.0,
        uncertainty_threshold: 0.05,
        max_position_size: 0.1,
        min_position_size: 0.01,
        use_kelly: true,
        min_return_threshold: 0.001,
        n_mc_samples: 30,
    };

    let signal_generator = SignalGenerator::new(signal_config);

    // Create backtest engine
    let backtest_config = BacktestConfig {
        initial_capital: 100000.0,
        commission: 0.001,
        slippage: 0.0005,
        holding_period: 1,
    };

    let engine = BacktestEngine::new(backtest_config);

    println!("\nRunning backtest...");
    println!("  Initial capital: $100,000");
    println!("  Commission: 0.1%");
    println!("  Slippage: 0.05%");
    println!("  Holding period: 1 bar\n");

    // Run backtest
    let report = engine.run(
        &model,
        &signal_generator,
        &features,
        &prices,
        None,
    );

    // Print results
    report.print_summary();

    // Additional analysis
    println!("\n=== Strategy Analysis ===");

    if report.is_profitable() {
        println!("Status: PROFITABLE");
    } else {
        println!("Status: NOT PROFITABLE");
    }

    if report.is_well_calibrated(0.1) {
        println!("Uncertainty Calibration: GOOD (within 10% of target 95% coverage)");
    } else {
        println!("Uncertainty Calibration: NEEDS IMPROVEMENT (>10% from target 95% coverage)");
    }

    // Risk analysis
    println!("\n=== Risk Analysis ===");
    if report.sharpe_ratio > 2.0 {
        println!("Sharpe Ratio: EXCELLENT (>{} is great)", 2.0);
    } else if report.sharpe_ratio > 1.0 {
        println!("Sharpe Ratio: GOOD (>1.0)");
    } else {
        println!("Sharpe Ratio: NEEDS IMPROVEMENT (<1.0)");
    }

    if report.max_drawdown < 0.1 {
        println!("Max Drawdown: LOW (<10%)");
    } else if report.max_drawdown < 0.2 {
        println!("Max Drawdown: MODERATE (10-20%)");
    } else {
        println!("Max Drawdown: HIGH (>20%)");
    }

    println!("\n=== Backtest complete! ===");
}
