//! Backtest Example
//!
//! This example demonstrates backtesting a RealNVP trading strategy
//! on historical cryptocurrency data from Bybit.

use anyhow::Result;
use ndarray::Array2;

use realnvp_trading::{
    api::BybitClient,
    backtest::Backtester,
    flow::RealNVP,
    trading::RealNVPTrader,
    utils::{compute_market_state, normalize_features, train_val_test_split},
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== RealNVP Backtest Example ===\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch historical data
    println!("Fetching BTCUSDT hourly data from Bybit...");
    let candles = client.get_klines("BTCUSDT", "60", 1000).await?;
    println!("Fetched {} candles", candles.len());

    if candles.len() < 200 {
        println!("Not enough data for backtesting. Need at least 200 candles.");
        return Ok(());
    }

    // Prepare features
    println!("\nPreparing features...");
    let lookback = 20;
    let mut features_list = Vec::new();

    for i in lookback..candles.len() {
        let window = &candles[i - lookback..i];
        let state = compute_market_state(window);
        features_list.push(state.to_vec());
    }

    let feature_dim = features_list[0].len();
    let n_samples = features_list.len();

    let features_flat: Vec<f64> = features_list.into_iter().flatten().collect();
    let features = Array2::from_shape_vec((n_samples, feature_dim), features_flat)?;

    println!("Feature matrix: {} samples x {} features", n_samples, feature_dim);

    // Split data
    let (train, val, test) = train_val_test_split(&features, 0.6, 0.2);
    println!("\nData split:");
    println!("  Train: {} samples", train.nrows());
    println!("  Validation: {} samples", val.nrows());
    println!("  Test: {} samples", test.nrows());

    // Normalize features
    let (train_norm, mean, std) = normalize_features(&train);
    println!("\nFeatures normalized");

    // Create RealNVP model
    println!("\nCreating RealNVP model...");
    let flow = RealNVP::new(feature_dim, 64, 8);
    let mut trader = RealNVPTrader::new(flow);
    trader.fit_normalizer(&train);

    // Compute threshold from training data
    let log_probs: Vec<f64> = train_norm.rows()
        .into_iter()
        .map(|row| trader.model().log_prob(&row.to_owned()))
        .collect();

    let mean_log_prob = log_probs.iter().sum::<f64>() / log_probs.len() as f64;
    let std_log_prob = {
        let variance: f64 = log_probs.iter()
            .map(|p| (p - mean_log_prob).powi(2))
            .sum::<f64>() / log_probs.len() as f64;
        variance.sqrt()
    };

    let threshold = mean_log_prob - 2.0 * std_log_prob;
    trader.set_threshold(threshold);
    println!("Log probability threshold: {:.4}", threshold);

    // Run backtest on training data
    println!("\n=== Training Period Backtest ===");
    let train_start = lookback;
    let train_end = train_start + train.nrows();
    let train_candles = &candles[..train_end];

    let backtester = Backtester::with_params(lookback, 50, 252.0 * 24.0); // Hourly data
    let train_results = backtester.run(&trader, train_candles);
    train_results.print_summary();

    // Run backtest on validation data
    println!("\n=== Validation Period Backtest ===");
    let val_end = train_end + val.nrows();
    let val_candles = &candles[..val_end];

    let val_results = backtester.run(&trader, val_candles);
    val_results.print_summary();

    // Run backtest on test data
    println!("\n=== Test Period Backtest ===");
    let test_results = backtester.run(&trader, &candles);
    test_results.print_summary();

    // Compare periods
    println!("\n=== Performance Comparison ===");
    println!("{:<20} {:>12} {:>12} {:>12}",
        "Metric", "Train", "Validation", "Test");
    println!("{}", "-".repeat(56));
    println!("{:<20} {:>11.4}% {:>11.4}% {:>11.4}%",
        "Total Return",
        train_results.metrics.total_return * 100.0,
        val_results.metrics.total_return * 100.0,
        test_results.metrics.total_return * 100.0);
    println!("{:<20} {:>12.4} {:>12.4} {:>12.4}",
        "Sharpe Ratio",
        train_results.metrics.sharpe_ratio,
        val_results.metrics.sharpe_ratio,
        test_results.metrics.sharpe_ratio);
    println!("{:<20} {:>12.4} {:>12.4} {:>12.4}",
        "Sortino Ratio",
        train_results.metrics.sortino_ratio,
        val_results.metrics.sortino_ratio,
        test_results.metrics.sortino_ratio);
    println!("{:<20} {:>11.4}% {:>11.4}% {:>11.4}%",
        "Max Drawdown",
        train_results.metrics.max_drawdown * 100.0,
        val_results.metrics.max_drawdown * 100.0,
        test_results.metrics.max_drawdown * 100.0);
    println!("{:<20} {:>11.2}% {:>11.2}% {:>11.2}%",
        "Win Rate",
        train_results.metrics.win_rate * 100.0,
        val_results.metrics.win_rate * 100.0,
        test_results.metrics.win_rate * 100.0);
    println!("{:<20} {:>12} {:>12} {:>12}",
        "Num Trades",
        train_results.metrics.num_trades,
        val_results.metrics.num_trades,
        test_results.metrics.num_trades);
    println!("{:<20} {:>12.4} {:>12.4} {:>12.4}",
        "Profit Factor",
        train_results.metrics.profit_factor,
        val_results.metrics.profit_factor,
        test_results.metrics.profit_factor);
    println!("{:<20} {:>11.2}% {:>11.2}% {:>11.2}%",
        "In-Dist Ratio",
        train_results.metrics.in_distribution_ratio * 100.0,
        val_results.metrics.in_distribution_ratio * 100.0,
        test_results.metrics.in_distribution_ratio * 100.0);

    // Equity curve statistics
    println!("\n=== Equity Curve Analysis ===");
    let equity = test_results.equity_curve();
    let returns = test_results.returns();

    if !equity.is_empty() {
        let final_equity = equity.last().unwrap_or(&0.0);
        let max_equity = equity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_equity = equity.iter().cloned().fold(f64::INFINITY, f64::min);

        println!("Final Equity: {:.4}%", final_equity * 100.0);
        println!("Peak Equity: {:.4}%", max_equity * 100.0);
        println!("Trough Equity: {:.4}%", min_equity * 100.0);

        // Return statistics
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let return_variance: f64 = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let return_std = return_variance.sqrt();

        println!("\nReturn Statistics:");
        println!("  Mean: {:.6}%", mean_return * 100.0);
        println!("  Std: {:.6}%", return_std * 100.0);

        let positive_returns: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).cloned().collect();
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if !positive_returns.is_empty() {
            let avg_win = positive_returns.iter().sum::<f64>() / positive_returns.len() as f64;
            println!("  Avg Win: {:.6}%", avg_win * 100.0);
        }
        if !negative_returns.is_empty() {
            let avg_loss = negative_returns.iter().sum::<f64>() / negative_returns.len() as f64;
            println!("  Avg Loss: {:.6}%", avg_loss * 100.0);
        }
    }

    // Risk metrics on test period
    println!("\n=== Risk Metrics (Test Period) ===");
    let risk = trader.compute_risk_metrics(10000);
    println!("VaR (5%): {:.4}%", risk.var_5 * 100.0);
    println!("VaR (1%): {:.4}%", risk.var_1 * 100.0);
    println!("CVaR (5%): {:.4}%", risk.cvar_5 * 100.0);
    println!("Max Expected Loss: {:.4}%", risk.max_loss * 100.0);
    println!("Prob of Loss: {:.2}%", risk.prob_loss * 100.0);

    println!("\n=== Backtest Complete ===");

    Ok(())
}
