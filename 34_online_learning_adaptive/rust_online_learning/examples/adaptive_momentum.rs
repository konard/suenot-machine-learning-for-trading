//! Adaptive Momentum Strategy Example
//!
//! Demonstrates a complete adaptive momentum trading strategy using online learning.
//!
//! Run with: cargo run --example adaptive_momentum

use online_learning::api::BybitClient;
use online_learning::drift::{DriftDetector, ADWIN};
use online_learning::features::MomentumFeatures;
use online_learning::models::AdaptiveMomentumWeights;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Adaptive Momentum Trading Strategy ===\n");

    // Fetch data
    let client = BybitClient::new();
    let symbol = "BTCUSDT";

    println!("Fetching {} data from Bybit...", symbol);
    let candles = client.get_klines(symbol, "1h", 800).await?;
    println!("Fetched {} candles\n", candles.len());

    // Configuration
    let periods = vec![12, 24, 48, 96];
    let learning_rate = 0.02;
    let signal_threshold = 0.0005;
    let transaction_cost = 0.001; // 0.1%

    let factor_names: Vec<String> = periods.iter().map(|p| format!("mom_{}h", p)).collect();

    println!("Strategy Configuration:");
    println!("  Momentum periods: {:?}", periods);
    println!("  Learning rate: {}", learning_rate);
    println!("  Signal threshold: {}", signal_threshold);
    println!("  Transaction cost: {:.2}%", transaction_cost * 100.0);
    println!();

    // Initialize components
    let feature_gen = MomentumFeatures::new(periods.clone());
    let mut weights = AdaptiveMomentumWeights::new(periods.len(), learning_rate, factor_names.clone());
    let mut drift_detector = ADWIN::new(0.01);

    // Warmup period
    let warmup = *periods.iter().max().unwrap() + 1;

    // Trading metrics
    let mut total_pnl = 0.0;
    let mut trades = 0;
    let mut wins = 0;
    let mut prev_position = 0.0;
    let mut equity_curve = vec![1.0];
    let mut weight_history: Vec<Vec<f64>> = Vec::new();
    let mut drift_points: Vec<usize> = Vec::new();

    println!("Starting trading simulation...\n");

    // Main trading loop
    for i in warmup..candles.len() - 1 {
        // Compute features
        let features = match feature_gen.compute(&candles[..=i]) {
            Some(f) => f,
            None => continue,
        };

        // Generate signal using current weights
        let weighted_signal = weights.predict(&features);

        // Determine position
        let position = if weighted_signal > signal_threshold {
            1.0
        } else if weighted_signal < -signal_threshold {
            -1.0
        } else {
            0.0
        };

        // Calculate actual return (next period)
        let actual_return = (candles[i + 1].close - candles[i].close) / candles[i].close;

        // Trade PnL
        let mut pnl = position * actual_return;

        // Apply transaction costs on position change
        if (position - prev_position).abs() > 0.5 {
            pnl -= transaction_cost;
        }

        total_pnl += pnl;

        // Track trades
        if position != 0.0 && position != prev_position {
            trades += 1;
            if pnl > 0.0 {
                wins += 1;
            }
        }

        // Update equity curve
        let last_equity = *equity_curve.last().unwrap();
        equity_curve.push(last_equity * (1.0 + pnl));

        // Drift detection on prediction error
        let prediction_error = (weighted_signal - actual_return).abs();
        if drift_detector.update(prediction_error) {
            drift_points.push(i);

            // On drift: could reset model or adjust learning rate
            // Here we just continue with online learning
        }

        // Update model weights with actual return
        weights.update(&features, actual_return);

        // Store weight history periodically
        if i % 24 == 0 {
            weight_history.push(weights.get_weights().to_vec());
        }

        prev_position = position;

        // Progress update
        if (i - warmup) % 100 == 0 && i > warmup {
            let current_equity = *equity_curve.last().unwrap();
            println!(
                "  Progress: {}/{}, Equity: {:.4}, Trades: {}",
                i - warmup,
                candles.len() - warmup - 1,
                current_equity,
                trades
            );
        }
    }

    // Calculate results
    println!("\n=== Trading Results ===\n");

    let final_equity = *equity_curve.last().unwrap();
    let total_return = (final_equity - 1.0) * 100.0;
    let win_rate = if trades > 0 { wins as f64 / trades as f64 * 100.0 } else { 0.0 };

    // Calculate Sharpe ratio
    let returns: Vec<f64> = equity_curve.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_return = (returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64).sqrt();
    let sharpe = if std_return > 0.0 { mean_return / std_return * (252.0_f64 * 24.0).sqrt() } else { 0.0 };

    // Calculate max drawdown
    let mut peak = 1.0;
    let mut max_dd = 0.0;
    for &eq in &equity_curve {
        peak = peak.max(eq);
        max_dd = max_dd.max((peak - eq) / peak);
    }

    println!("Performance Metrics:");
    println!("  Total Return: {:.2}%", total_return);
    println!("  Final Equity: {:.4}", final_equity);
    println!("  Sharpe Ratio: {:.2}", sharpe);
    println!("  Max Drawdown: {:.2}%", max_dd * 100.0);
    println!("  Total Trades: {}", trades);
    println!("  Win Rate: {:.2}%", win_rate);
    println!("  Drift Events: {}", drift_points.len());
    println!();

    // Display final weights
    println!("Final Adaptive Weights:");
    let final_weights = weights.get_weights();
    for (name, weight) in factor_names.iter().zip(final_weights.iter()) {
        let bar_len = (weight.abs() * 40.0) as usize;
        let bar: String = if *weight >= 0.0 {
            format!("+{}", "█".repeat(bar_len))
        } else {
            format!("-{}", "█".repeat(bar_len))
        };
        println!("  {:>10}: {:>6.3} {}", name, weight, bar);
    }

    // Weight evolution summary
    if !weight_history.is_empty() {
        println!("\nWeight Evolution (every 24 hours):");

        // First and last weights
        let first_weights = &weight_history[0];
        let last_weights = weight_history.last().unwrap();

        println!("  {:>12} {:>10} {:>10} {:>10}", "Factor", "Start", "End", "Change");
        for (j, name) in factor_names.iter().enumerate() {
            let change = last_weights[j] - first_weights[j];
            let change_str = if change >= 0.0 {
                format!("+{:.3}", change)
            } else {
                format!("{:.3}", change)
            };
            println!(
                "  {:>12} {:>10.3} {:>10.3} {:>10}",
                name, first_weights[j], last_weights[j], change_str
            );
        }
    }

    // Best performing factor
    if let Some((name, weight)) = weights.best_factor() {
        println!("\nBest Performing Factor: {} (weight: {:.3})", name, weight);
    }

    println!("\n=== Strategy Insights ===\n");

    // Determine which momentum is most important
    let weights_vec = weights.get_weights();
    let max_idx = weights_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!("Key Observations:");
    println!(
        "  - The {} momentum factor has the highest weight",
        factor_names[max_idx]
    );
    println!(
        "  - {} drift events detected during the period",
        drift_points.len()
    );

    if drift_points.len() > 5 {
        println!("  - High market volatility suggested by frequent drift detection");
        println!("  - Consider using faster learning rate or shorter lookback periods");
    } else if drift_points.is_empty() {
        println!("  - Market conditions appear stable");
        println!("  - Could potentially use longer lookback periods for more signal");
    }

    if total_return > 0.0 {
        println!("  - Strategy showed positive returns during the test period");
    } else {
        println!("  - Strategy had negative returns - consider parameter tuning");
    }

    Ok(())
}
