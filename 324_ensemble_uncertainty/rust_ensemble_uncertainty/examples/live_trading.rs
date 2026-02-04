//! Example: Live trading simulation with uncertainty-aware strategy

use ensemble_uncertainty_trading::prelude::*;
use ensemble_uncertainty_trading::ensemble::traits::EnsembleModel;
use ensemble_uncertainty_trading::backtest::engine::{BacktestEngine, BacktestConfig};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() -> anyhow::Result<()> {
    println!("=".repeat(60));
    println!("Ensemble Uncertainty Trading Strategy");
    println!("=".repeat(60));

    // Generate synthetic market data
    println!("\n--- Generating Market Data ---");
    let n_periods = 1000;
    let n_features = 20;

    // Simulate price series
    let mut prices = vec![50000.0_f64];
    let returns: Vec<f64> = (0..n_periods - 1)
        .map(|i| {
            let trend = 0.0001 * (i as f64 / n_periods as f64 * 2.0 * std::f64::consts::PI).sin();
            let noise = rand::random::<f64>() * 0.02 - 0.01;
            trend + noise
        })
        .collect();

    for ret in &returns {
        let last_price = *prices.last().unwrap();
        prices.push(last_price * (1.0 + ret));
    }

    let timestamps: Vec<i64> = (0..n_periods)
        .map(|i| 1704067200000 + i as i64 * 3600000) // Hourly timestamps
        .collect();

    // Generate features (technical indicators simulation)
    let x = Array2::random((n_periods, n_features), Uniform::new(-1.0, 1.0));

    // Generate target (future returns)
    let y: Array1<f64> = returns.iter().copied().collect();

    // Split data
    let train_size = (n_periods as f64 * 0.7) as usize;
    let x_train = x.slice(ndarray::s![..train_size, ..]).to_owned();
    let y_train = y.slice(ndarray::s![..train_size]).to_owned();
    let x_test = x.slice(ndarray::s![train_size.., ..]).to_owned();

    println!("Training periods: {}", train_size);
    println!("Test periods: {}", n_periods - train_size);

    // Train ensemble
    println!("\n--- Training Ensemble ---");
    let mut rf = RandomForestEnsemble::new(50, Some(10));
    rf.fit(&x_train, &y_train)?;
    println!("Random Forest trained");

    let mut gb = GradientBoostingEnsemble::new(50, true);
    gb.fit(&x_train, &y_train)?;
    println!("Gradient Boosting trained");

    // Get predictions with uncertainty
    println!("\n--- Generating Predictions ---");
    let (rf_preds, rf_unc) = rf.predict_with_uncertainty(&x_test)?;
    let (gb_preds, gb_unc) = gb.predict_with_uncertainty(&x_test)?;

    // Combine predictions
    let predictions: Vec<f64> = rf_preds.iter()
        .zip(gb_preds.iter())
        .map(|(&r, &g)| (r + g) / 2.0)
        .collect();

    let uncertainties: Vec<f64> = rf_unc.iter()
        .zip(gb_unc.iter())
        .map(|(&r, &g)| {
            let disagreement = (rf_preds[0] - gb_preds[0]).abs();
            ((r * r + g * g) / 2.0 + disagreement * disagreement).sqrt()
        })
        .collect();

    println!("Mean prediction: {:.6}", predictions.iter().sum::<f64>() / predictions.len() as f64);
    println!("Mean uncertainty: {:.6}", uncertainties.iter().sum::<f64>() / uncertainties.len() as f64);

    // Generate trading signals
    println!("\n--- Generating Trading Signals ---");
    let strategy = UncertaintyStrategy::new(0.001, 0.03, 0.6);

    let test_prices = &prices[train_size..];
    let signals: Vec<Signal> = predictions.iter()
        .zip(uncertainties.iter())
        .enumerate()
        .map(|(i, (&pred, &unc))| {
            strategy.generate_signal(
                "BTCUSDT",
                pred,
                unc,
                Some(test_prices[i]),
                Some(0.55), // Historical win rate
            )
        })
        .collect();

    // Count signal types
    let long_signals = signals.iter().filter(|s| s.signal_type == SignalType::Long).count();
    let short_signals = signals.iter().filter(|s| s.signal_type == SignalType::Short).count();
    let hold_signals = signals.iter().filter(|s| s.signal_type == SignalType::Hold).count();

    println!("Signal distribution:");
    println!("  LONG:  {} ({:.1}%)", long_signals, long_signals as f64 / signals.len() as f64 * 100.0);
    println!("  SHORT: {} ({:.1}%)", short_signals, short_signals as f64 / signals.len() as f64 * 100.0);
    println!("  HOLD:  {} ({:.1}%)", hold_signals, hold_signals as f64 / signals.len() as f64 * 100.0);

    // Run backtest
    println!("\n--- Running Backtest ---");
    let config = BacktestConfig {
        initial_capital: 100000.0,
        transaction_cost: 0.001,
        slippage: 0.0005,
        risk_free_rate: 0.02,
    };

    let backtester = BacktestEngine::new(config);

    let test_timestamps = &timestamps[train_size..];
    let result = backtester.run(
        test_timestamps,
        test_prices,
        &predictions,
        &uncertainties,
        &strategy,
        5, // hold periods
    );

    result.print();
    result.print_confidence_analysis();

    // Compare with no-uncertainty strategy
    println!("\n--- Comparison: With vs Without Uncertainty ---");

    let strategy_no_unc = UncertaintyStrategy::new(0.001, 1.0, 0.0); // No filtering
    let result_no_unc = backtester.run(
        test_timestamps,
        test_prices,
        &predictions,
        &uncertainties,
        &strategy_no_unc,
        5,
    );

    println!("\n{:<25} {:>15} {:>15}", "Metric", "With Unc.", "Without Unc.");
    println!("{}", "-".repeat(55));
    println!("{:<25} {:>14.2}% {:>14.2}%", "Total Return",
        result.total_return * 100.0, result_no_unc.total_return * 100.0);
    println!("{:<25} {:>15.2} {:>15.2}", "Sharpe Ratio",
        result.sharpe_ratio, result_no_unc.sharpe_ratio);
    println!("{:<25} {:>15.2} {:>15.2}", "Sortino Ratio",
        result.sortino_ratio, result_no_unc.sortino_ratio);
    println!("{:<25} {:>14.2}% {:>14.2}%", "Max Drawdown",
        result.max_drawdown * 100.0, result_no_unc.max_drawdown * 100.0);
    println!("{:<25} {:>14.2}% {:>14.2}%", "Win Rate",
        result.win_rate * 100.0, result_no_unc.win_rate * 100.0);
    println!("{:<25} {:>15} {:>15}", "Total Trades",
        result.total_trades, result_no_unc.total_trades);

    // Improvement summary
    println!("\n--- Improvement from Uncertainty-Aware Trading ---");
    let sharpe_improvement = result.sharpe_ratio - result_no_unc.sharpe_ratio;
    let dd_improvement = result_no_unc.max_drawdown - result.max_drawdown; // Less negative is better

    if sharpe_improvement > 0.0 {
        println!("  Sharpe improved by: {:.2}", sharpe_improvement);
    } else {
        println!("  Sharpe changed by: {:.2}", sharpe_improvement);
    }

    if dd_improvement > 0.0 {
        println!("  Max drawdown improved by: {:.2}%", dd_improvement * 100.0);
    }

    println!("\n{}", "=".repeat(60));
    println!("Trading simulation complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}
