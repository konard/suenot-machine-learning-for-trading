//! Example: Full backtest with conformal prediction

use conformal_trading::conformal::predictor::SplitConformalPredictor;
use conformal_trading::strategy::signal::{SignalGenerator, PositionScaling};
use conformal_trading::backtest::engine::{BacktestEngine, BacktestConfig, BacktestResult};
use conformal_trading::backtest::metrics::CoverageMetrics;
use rand::Rng;
use rand_distr::{Distribution, Normal};

fn main() {
    println!("Conformal Prediction Trading - Backtest Demo");
    println!("{}", "=".repeat(50));

    // ==========================================================
    // Generate Synthetic Market Data
    // ==========================================================
    println!("\nGenerating synthetic market data...");

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0001, 0.015).unwrap(); // Daily: ~0.01% drift, 1.5% vol

    // Generate price path
    let n_periods = 2000;
    let mut prices = vec![100.0];

    for _ in 1..n_periods {
        let return_: f64 = normal.sample(&mut rng);
        let new_price = prices.last().unwrap() * (1.0 + return_);
        prices.push(new_price);
    }

    // Calculate returns
    let returns: Vec<f64> = prices.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    println!("  Generated {} periods", n_periods);
    println!("  Price range: ${:.2} - ${:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // ==========================================================
    // Create Features (Simplified)
    // ==========================================================
    println!("\nCreating features...");

    let lookback = 20;
    let prediction_horizon = 5;

    // Features: simple momentum and volatility
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for i in lookback..(returns.len() - prediction_horizon) {
        let momentum: f64 = returns[i-lookback..i].iter().sum();
        let volatility: f64 = {
            let mean = returns[i-lookback..i].iter().sum::<f64>() / lookback as f64;
            let var: f64 = returns[i-lookback..i].iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / lookback as f64;
            var.sqrt()
        };

        features.push(vec![momentum, volatility, returns[i-1]]);
        targets.push(returns[i + prediction_horizon - 1]);
    }

    println!("  Created {} samples with {} features", features.len(), 3);

    // ==========================================================
    // Split Data
    // ==========================================================
    let train_end = (features.len() as f64 * 0.6) as usize;
    let cal_end = (features.len() as f64 * 0.8) as usize;

    let y_train = &targets[..train_end];
    let y_cal = &targets[train_end..cal_end];
    let y_test = &targets[cal_end..];

    println!("\nData split:");
    println!("  Training: {} samples", train_end);
    println!("  Calibration: {} samples", cal_end - train_end);
    println!("  Test: {} samples", targets.len() - cal_end);

    // ==========================================================
    // Train Simple Predictor (using mean for simplicity)
    // ==========================================================
    println!("\nTraining predictor...");

    // In practice, you'd use a real ML model. Here we use simple momentum-based prediction.
    let train_mean: f64 = y_train.iter().sum::<f64>() / y_train.len() as f64;

    // Calibration predictions (add noise to simulate real model)
    let y_pred_cal: Vec<f64> = y_cal.iter()
        .map(|&y| y + rng.gen_range(-0.01..0.01))
        .collect();

    // Test predictions
    let y_pred_test: Vec<f64> = y_test.iter()
        .map(|&y| y + rng.gen_range(-0.01..0.01))
        .collect();

    // ==========================================================
    // Calibrate Conformal Predictor
    // ==========================================================
    println!("\nCalibrating conformal predictor...");

    let mut cp = SplitConformalPredictor::new(0.1);

    // Compute calibration scores
    let cal_scores: Vec<f64> = y_cal.iter()
        .zip(y_pred_cal.iter())
        .map(|(&y, &yp)| (y - yp).abs())
        .collect();

    cp.calibrate_from_scores(cal_scores);

    println!("  Quantile: {:.6}", cp.quantile().unwrap());

    // ==========================================================
    // Generate Prediction Intervals
    // ==========================================================
    println!("\nGenerating prediction intervals...");

    let intervals: Vec<_> = y_pred_test.iter()
        .map(|&pred| cp.predict_interval(pred).unwrap())
        .collect();

    let lower_bounds: Vec<f64> = intervals.iter().map(|i| i.lower).collect();
    let upper_bounds: Vec<f64> = intervals.iter().map(|i| i.upper).collect();

    // Coverage metrics
    let coverage_metrics = CoverageMetrics::calculate(
        y_test,
        &lower_bounds,
        &upper_bounds,
        0.1
    );

    println!("\n{}", coverage_metrics);

    // ==========================================================
    // Generate Trading Signals
    // ==========================================================
    println!("\nGenerating trading signals...");

    let train_vol: f64 = {
        let mean = y_train.iter().sum::<f64>() / y_train.len() as f64;
        let var: f64 = y_train.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / y_train.len() as f64;
        var.sqrt()
    };

    let signal_gen = SignalGenerator::new(0.4, 1.0)
        .with_historical_volatility(train_vol)
        .with_position_scaling(PositionScaling::InverseWidth);

    let timestamps: Vec<i64> = (0..intervals.len())
        .map(|i| (cal_end + i) as i64 * 3600000)
        .collect();

    let signals = signal_gen.generate_batch(&timestamps, &intervals);

    // Signal statistics
    let long_count = signals.iter().filter(|s| s.signal_type == conformal_trading::strategy::signal::SignalType::Long).count();
    let short_count = signals.iter().filter(|s| s.signal_type == conformal_trading::strategy::signal::SignalType::Short).count();
    let hold_count = signals.iter().filter(|s| s.signal_type == conformal_trading::strategy::signal::SignalType::Hold).count();

    println!("\nSignal distribution:");
    println!("  LONG:  {} ({:.1}%)", long_count, 100.0 * long_count as f64 / signals.len() as f64);
    println!("  SHORT: {} ({:.1}%)", short_count, 100.0 * short_count as f64 / signals.len() as f64);
    println!("  HOLD:  {} ({:.1}%)", hold_count, 100.0 * hold_count as f64 / signals.len() as f64);

    // ==========================================================
    // Run Backtest
    // ==========================================================
    println!("\n{}", "=".repeat(50));
    println!("RUNNING BACKTEST");
    println!("{}", "=".repeat(50));

    let config = BacktestConfig {
        initial_capital: 10000.0,
        max_leverage: 1.0,
        transaction_cost: 0.001,
        slippage: 0.0005,
    };

    let mut engine = BacktestEngine::new(config.clone());

    // Get test prices
    let test_prices = &prices[cal_end + lookback..cal_end + lookback + signals.len()];

    let performance = engine.run(&signals, test_prices);

    // Create result
    let result = BacktestResult::new(
        performance,
        coverage_metrics,
        engine.equity_curve().to_vec(),
        engine.trades().to_vec(),
        signals,
    );

    result.print_summary();

    // ==========================================================
    // Sensitivity Analysis
    // ==========================================================
    println!("\n{}", "=".repeat(50));
    println!("SENSITIVITY ANALYSIS");
    println!("{}", "=".repeat(50));

    println!("\nVarying alpha (miscoverage rate):");
    println!("{:<10} {:>12} {:>12} {:>12} {:>12}",
        "Alpha", "Coverage", "Avg Width", "Sharpe", "Return");
    println!("{}", "-".repeat(60));

    for alpha in [0.05, 0.10, 0.15, 0.20, 0.30] {
        let mut cp_test = SplitConformalPredictor::new(alpha);
        cp_test.calibrate_from_scores(cal_scores.clone());

        let intervals: Vec<_> = y_pred_test.iter()
            .map(|&pred| cp_test.predict_interval(pred).unwrap())
            .collect();

        let lower: Vec<f64> = intervals.iter().map(|i| i.lower).collect();
        let upper: Vec<f64> = intervals.iter().map(|i| i.upper).collect();

        let cov = CoverageMetrics::calculate(y_test, &lower, &upper, alpha);

        let signals = signal_gen.generate_batch(&timestamps, &intervals);

        let mut engine = BacktestEngine::new(config.clone());
        let perf = engine.run(&signals, test_prices);

        println!("{:<10.2} {:>12.1}% {:>12.6} {:>12.2} {:>12.2}%",
            alpha,
            cov.coverage * 100.0,
            cov.mean_width,
            perf.sharpe_ratio,
            perf.total_return * 100.0
        );
    }

    println!("\nBacktest complete!");
}
