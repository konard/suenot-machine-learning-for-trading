//! Example: Backtest Linformer trading strategy.
//!
//! Run with: cargo run --example backtest

use linformer::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("Linformer Backtesting Example");
    println!("=============================\n");

    // Generate synthetic price data with trend and noise
    let n_days = 500;
    let mut prices = Vec::with_capacity(n_days);
    let mut price = 100.0;

    for i in 0..n_days {
        // Add trend, seasonality, and noise
        let trend = 0.0005 * (i as f64);
        let seasonality = 2.0 * (i as f64 * 0.05).sin();
        let noise = (rand::random::<f64>() - 0.5) * 3.0;

        price = (price + trend + seasonality + noise).max(50.0);
        prices.push(price);
    }

    println!("Generated {} days of synthetic price data", n_days);
    println!(
        "Price range: ${:.2} - ${:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Calculate returns
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    // Create model and generate predictions
    println!("\nCreating Linformer model...");
    let seq_len = 64;
    let config = LinformerConfig::new(32, 2, seq_len, 16, 2)
        .with_n_features(6)
        .with_n_outputs(1);

    let model = Linformer::new(config)?;
    println!("{}", model.summary());

    // Calculate features
    let prices_arr = Array1::from_vec(prices.clone());
    let features = TechnicalFeatures::calculate_all(&prices_arr);
    let normalized = TechnicalFeatures::normalize_zscore(&features);

    // Generate predictions for backtest period
    println!("\nGenerating predictions...");
    let mut predictions = Vec::with_capacity(returns.len());

    for i in 0..returns.len() {
        if i + 1 >= seq_len {
            // Get sequence window
            let start = i + 1 - seq_len;
            let end = i + 1;
            let seq = normalized
                .slice(ndarray::s![start..end, ..])
                .to_owned();

            let pred = model.forward(&seq);
            predictions.push(pred[0]);
        } else {
            predictions.push(0.0);
        }
    }

    println!("Generated {} predictions", predictions.len());

    // Configure backtest
    println!("\nConfiguring backtest...");
    let backtest_config = BacktestConfig {
        initial_capital: 10000.0,
        transaction_cost: 0.001,
        slippage: 0.0005,
        long_threshold: 0.001,
        short_threshold: -0.001,
        position_size: 1.0,
        risk_free_rate: 0.02,
        periods_per_year: 252.0,
    };

    println!("  Initial capital: ${:.2}", backtest_config.initial_capital);
    println!(
        "  Transaction cost: {:.2}%",
        backtest_config.transaction_cost * 100.0
    );
    println!("  Slippage: {:.2}%", backtest_config.slippage * 100.0);

    // Run backtest
    let backtester = Backtester::with_config(backtest_config);
    let result = backtester.run(&predictions, &returns);

    // Print results
    println!("\n{}", result.metrics.summary());

    // Compare to benchmark
    println!("\n{}", backtester.compare_to_benchmark(&result, &returns));

    // Position analysis
    let long_count = result.positions.iter().filter(|&&p| p == 1).count();
    let short_count = result.positions.iter().filter(|&&p| p == -1).count();
    let flat_count = result.positions.iter().filter(|&&p| p == 0).count();

    println!("\nPosition Distribution:");
    println!("  Long:  {} ({:.1}%)", long_count, 100.0 * long_count as f64 / result.positions.len() as f64);
    println!("  Short: {} ({:.1}%)", short_count, 100.0 * short_count as f64 / result.positions.len() as f64);
    println!("  Flat:  {} ({:.1}%)", flat_count, 100.0 * flat_count as f64 / result.positions.len() as f64);

    // Equity curve summary
    if let (Some(min), Some(max)) = (
        result.equity_curve.iter().cloned().reduce(f64::min),
        result.equity_curve.iter().cloned().reduce(f64::max),
    ) {
        println!("\nEquity Curve:");
        println!("  Start: ${:.2}", result.equity_curve.first().unwrap_or(&0.0));
        println!("  End:   ${:.2}", result.equity_curve.last().unwrap_or(&0.0));
        println!("  Min:   ${:.2}", min);
        println!("  Max:   ${:.2}", max);
    }

    println!("\nBacktest completed successfully!");

    Ok(())
}
