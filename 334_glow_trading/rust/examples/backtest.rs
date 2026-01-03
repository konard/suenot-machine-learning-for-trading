//! Example: Backtest GLOW trading strategy
//!
//! This example demonstrates how to run a backtest
//! using a trained GLOW model on historical data.
//!
//! Run with: cargo run --example backtest

use anyhow::Result;
use glow_trading::{
    Candle, Checkpoint, GLOWTrader, TraderConfig,
    Backtest, BacktestConfig,
};

fn main() -> Result<()> {
    println!("=== GLOW Trading: Backtester ===\n");

    // Load model
    let model_file = "glow_model.bin";
    println!("Loading model from {}...", model_file);

    let checkpoint = match Checkpoint::load(model_file) {
        Ok(c) => c,
        Err(_) => {
            println!("Model file not found. Please train model first:");
            println!("  cargo run --example train_model");
            return Ok(());
        }
    };

    println!("Model loaded successfully!");
    println!("  Features: {}", checkpoint.model.config.num_features);
    println!("  Levels: {}", checkpoint.model.config.num_levels);

    // Load data
    let data_file = "btc_data.csv";
    println!("\nLoading data from {}...", data_file);

    let mut rdr = match csv::Reader::from_path(data_file) {
        Ok(r) => r,
        Err(_) => {
            println!("Data file not found. Please fetch data first:");
            println!("  cargo run --example fetch_data");
            return Ok(());
        }
    };

    let mut candles = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let candle = Candle {
            timestamp: record[0].parse()?,
            open: record[1].parse()?,
            high: record[2].parse()?,
            low: record[3].parse()?,
            close: record[4].parse()?,
            volume: record[5].parse()?,
            turnover: record[6].parse()?,
        };
        candles.push(candle);
    }

    println!("Loaded {} candles", candles.len());

    // Create trader
    let trader_config = TraderConfig {
        likelihood_threshold: -20.0,
        confidence_threshold: 0.3,
        num_regimes: 4,
        lookback: 20,
        max_position: 1.0,
    };

    let mut trader = GLOWTrader::new(checkpoint.model, trader_config.clone());

    if let Some(normalizer) = checkpoint.normalizer {
        trader.set_normalizer(normalizer);
    }

    println!("\nTrader configuration:");
    println!("  Likelihood threshold: {}", trader_config.likelihood_threshold);
    println!("  Confidence threshold: {}", trader_config.confidence_threshold);
    println!("  Max position: {:.0}%", trader_config.max_position * 100.0);

    // Run backtest
    let initial_capital = 10000.0;
    let backtest_config = BacktestConfig {
        initial_capital,
        transaction_cost: 0.001,
        slippage: 0.0005,
        warmup: 100,
        lookback: 20,
    };

    println!("\nBacktest configuration:");
    println!("  Initial capital: ${:.2}", backtest_config.initial_capital);
    println!("  Transaction cost: {:.2}%", backtest_config.transaction_cost * 100.0);
    println!("  Slippage: {:.2}%", backtest_config.slippage * 100.0);
    println!("  Warmup period: {} candles", backtest_config.warmup);

    println!("\nRunning backtest...");
    let backtest = Backtest::new(backtest_config);
    let result = backtest.run(&mut trader, &candles);

    // Print results
    println!("\n{:=<60}", "");
    println!("=== BACKTEST RESULTS ===");
    println!("{:=<60}", "");

    println!("\n--- Performance Metrics ---");
    println!("Total Return:        {:>10.2}%", result.metrics.total_return * 100.0);
    println!("Annualized Return:   {:>10.2}%", result.metrics.annualized_return * 100.0);
    println!("Sharpe Ratio:        {:>10.2}", result.metrics.sharpe_ratio);
    println!("Sortino Ratio:       {:>10.2}", result.metrics.sortino_ratio);
    println!("Calmar Ratio:        {:>10.2}", result.metrics.calmar_ratio);

    println!("\n--- Risk Metrics ---");
    println!("Max Drawdown:        {:>10.2}%", result.metrics.max_drawdown * 100.0);

    println!("\n--- Trade Statistics ---");
    println!("Number of Trades:    {:>10}", result.metrics.num_trades);
    println!("Win Rate:            {:>10.2}%", result.metrics.win_rate * 100.0);
    println!("Average Win:         ${:>9.2}", result.metrics.avg_win);
    println!("Average Loss:        ${:>9.2}", result.metrics.avg_loss);
    println!("Profit Factor:       {:>10.2}", result.metrics.profit_factor);

    println!("\n--- Equity ---");
    println!("Initial Capital:     ${:>9.2}", initial_capital);
    println!("Final Equity:        ${:>9.2}", result.final_equity());
    println!("Absolute P&L:        ${:>9.2}", result.total_return());

    // Print equity curve summary
    if !result.steps.is_empty() {
        let equity_curve = result.equity_curve();
        let max_equity = equity_curve.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_equity = equity_curve.iter().cloned().fold(f64::INFINITY, f64::min);

        println!("\n--- Equity Curve ---");
        println!("Peak Equity:         ${:>9.2}", max_equity);
        println!("Trough Equity:       ${:>9.2}", min_equity);
    }

    // Signal statistics
    let signals = result.signal_series();
    let long_signals = signals.iter().filter(|&&s| s > 0.0).count();
    let short_signals = signals.iter().filter(|&&s| s < 0.0).count();
    let neutral_signals = signals.iter().filter(|&&s| s == 0.0).count();

    println!("\n--- Signal Distribution ---");
    println!("Long Signals:        {:>10}", long_signals);
    println!("Short Signals:       {:>10}", short_signals);
    println!("Neutral Signals:     {:>10}", neutral_signals);

    // In-distribution statistics
    let in_dist_count = result.steps.iter().filter(|s| s.in_distribution).count();
    let in_dist_ratio = in_dist_count as f64 / result.steps.len() as f64;
    println!("\n--- Distribution Statistics ---");
    println!("In-Distribution:     {:>10.2}%", in_dist_ratio * 100.0);

    // Regime distribution
    let mut regime_counts = [0usize; 4];
    for step in &result.steps {
        if step.regime < 4 {
            regime_counts[step.regime] += 1;
        }
    }
    println!("\n--- Regime Distribution ---");
    for (i, count) in regime_counts.iter().enumerate() {
        let pct = *count as f64 / result.steps.len() as f64 * 100.0;
        println!("Regime {}:            {:>10.2}%", i, pct);
    }

    println!("\n{:=<60}", "");

    // Save results to CSV
    let output_file = "backtest_results.csv";
    let mut wtr = csv::Writer::from_path(output_file)?;
    wtr.write_record(&[
        "timestamp", "price", "signal", "log_likelihood",
        "in_distribution", "regime", "position", "pnl", "cumulative_pnl", "equity"
    ])?;

    for step in &result.steps {
        wtr.write_record(&[
            step.timestamp.to_string(),
            step.price.to_string(),
            step.signal.to_string(),
            step.log_likelihood.to_string(),
            step.in_distribution.to_string(),
            step.regime.to_string(),
            step.position.to_string(),
            step.pnl.to_string(),
            step.cumulative_pnl.to_string(),
            step.equity.to_string(),
        ])?;
    }
    wtr.flush()?;

    println!("\nResults saved to {}", output_file);

    Ok(())
}
