//! Example: Trading Strategy with Conformal Prediction
//!
//! This example demonstrates how to use conformal prediction intervals
//! to build a trading strategy that only trades when confident.
//!
//! Run with: cargo run --example trading_strategy

use conformal_prediction_trading::{
    api::bybit::{BybitClient, Interval},
    conformal::{model::LinearModel, split::SplitConformalPredictor},
    data::{features::FeatureEngineering, processor::DataProcessor},
    strategy::{sizing::PositionSizer, trading::ConformalTradingStrategy},
};

fn main() -> anyhow::Result<()> {
    println!("=== Trading Strategy with Conformal Prediction ===\n");

    // Fetch data
    println!("Fetching data from Bybit...");
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", Interval::Hour4, Some(500), None, None)?;
    println!("Received {} candles\n", klines.len());

    // Generate features and targets
    let (features, _) = FeatureEngineering::generate_features(&klines);
    let targets = FeatureEngineering::create_returns(&klines, 1);

    // Clean data
    let valid_start = 30;
    let features = features.slice(ndarray::s![valid_start.., ..]).to_owned();
    let targets: Vec<f64> = targets[valid_start..].to_vec();

    // Split data
    let ((x_train, y_train), (x_calib, y_calib), (x_test, y_test)) =
        DataProcessor::train_calib_test_split(&features, &targets, 0.6, 0.2);

    println!(
        "Data split: {} train, {} calib, {} test\n",
        x_train.nrows(),
        x_calib.nrows(),
        x_test.nrows()
    );

    // Train conformal predictor
    println!("Training conformal predictor...");
    let model = LinearModel::new(true);
    let mut cp = SplitConformalPredictor::new(model, 0.1);
    cp.fit(&x_train, &y_train, &x_calib, &y_calib);
    println!(
        "Calibration done. Interval width: {:.4}%\n",
        cp.interval_width() * 100.0
    );

    // Create trading strategies
    let conservative_strategy =
        ConformalTradingStrategy::new(0.01, 0.002); // Narrow threshold
    let moderate_strategy =
        ConformalTradingStrategy::new(0.02, 0.003); // Moderate threshold
    let aggressive_strategy =
        ConformalTradingStrategy::new(0.05, 0.001); // Wide threshold

    // Get predictions
    let intervals = cp.predict(&x_test);

    // Backtest each strategy
    println!("Backtesting strategies...\n");

    for (name, strategy) in [
        ("Conservative", &conservative_strategy),
        ("Moderate", &moderate_strategy),
        ("Aggressive", &aggressive_strategy),
    ] {
        let signals = strategy.generate_signals(&intervals);

        let n_trades = signals.iter().filter(|s| s.trade).count();
        let trade_freq = n_trades as f64 / signals.len() as f64;

        // Calculate PnL
        let mut total_pnl = 0.0;
        let mut wins = 0;
        let mut losses = 0;

        for (signal, &actual) in signals.iter().zip(y_test.iter()) {
            if signal.trade {
                let pnl = signal.direction as f64 * signal.size * actual;
                total_pnl += pnl;
                if pnl > 0.0 {
                    wins += 1;
                } else if pnl < 0.0 {
                    losses += 1;
                }
            }
        }

        let win_rate = if n_trades > 0 {
            wins as f64 / n_trades as f64
        } else {
            0.0
        };

        println!("--- {} Strategy ---", name);
        println!("Width threshold: {:.2}%", strategy.width_threshold() * 100.0);
        println!("Min edge: {:.2}%", strategy.min_edge() * 100.0);
        println!("Trades: {} ({:.1}% of periods)", n_trades, trade_freq * 100.0);
        println!("Win rate: {:.1}%", win_rate * 100.0);
        println!("Total PnL: {:.4}%", total_pnl * 100.0);
        println!();
    }

    // Detailed analysis of moderate strategy
    println!("\n=== Detailed Analysis: Moderate Strategy ===\n");

    let signals = moderate_strategy.generate_signals(&intervals);

    // Analyze skip reasons
    let mut skip_reasons: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for signal in &signals {
        if !signal.trade {
            let reason = signal.skip_reason.as_ref().unwrap_or(&"unknown".to_string());
            *skip_reasons.entry(reason.clone()).or_insert(0) += 1;
        }
    }

    println!("Skip reasons:");
    for (reason, count) in &skip_reasons {
        println!("  {}: {} ({:.1}%)", reason, count, *count as f64 / signals.len() as f64 * 100.0);
    }

    // Show some example trades
    println!("\n--- Example Trades (first 10) ---");
    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10}",
        "#", "Actual%", "Lower%", "Upper%", "Width%", "Dir", "Size", "PnL%"
    );

    let mut trade_count = 0;
    for (i, (signal, &actual)) in signals.iter().zip(y_test.iter()).enumerate() {
        if signal.trade && trade_count < 10 {
            let pnl = signal.direction as f64 * signal.size * actual;
            let dir = if signal.direction > 0 { "LONG" } else { "SHORT" };
            println!(
                "{:>5} {:>10.4} {:>10.4} {:>10.4} {:>8.4} {:>8} {:>10.4} {:>10.4}",
                i,
                actual * 100.0,
                signal.lower * 100.0,
                signal.upper * 100.0,
                signal.interval_width * 100.0,
                dir,
                signal.size,
                pnl * 100.0
            );
            trade_count += 1;
        }
    }

    // Compare with always-trade baseline
    println!("\n=== Comparison with Always-Trade Baseline ===\n");

    let baseline_pnl: f64 = y_test.iter().sum();
    let conformal_signals = moderate_strategy.generate_signals(&intervals);
    let conformal_pnl: f64 = conformal_signals
        .iter()
        .zip(y_test.iter())
        .map(|(s, &actual)| s.direction as f64 * s.size * actual)
        .sum();

    println!("Always trade (long only): {:.4}%", baseline_pnl * 100.0);
    println!("Conformal strategy: {:.4}%", conformal_pnl * 100.0);
    println!(
        "Improvement: {:.4}%",
        (conformal_pnl - baseline_pnl) * 100.0
    );

    println!("\n=== Done! ===");

    Ok(())
}
