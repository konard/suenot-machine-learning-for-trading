//! Multi-Task Trading Strategy Example
//!
//! Demonstrates end-to-end workflow: feature extraction, multi-task prediction,
//! signal generation, and backtesting with the task-agnostic model.
//!
//! Run with: cargo run --example multi_task_strategy

use task_agnostic_trading::model::{TaskAgnosticConfig, TaskAgnosticModel, TaskType};
use task_agnostic_trading::training::{MultiTaskTrainer, TrainerConfig};
use task_agnostic_trading::strategy::{SignalConfig, SignalGenerator};
use task_agnostic_trading::backtest::{BacktestEngine, BacktestConfig};
use ndarray::Array2;

fn main() {
    println!("=== Multi-Task Trading Strategy ===\n");

    // Step 1: Create model
    println!("Step 1: Creating task-agnostic model...");
    let config = TaskAgnosticConfig::default()
        .with_input_dim(20)
        .with_repr_dim(16);
    let mut model = TaskAgnosticModel::new(config);
    println!("  Model: {} task heads, repr_dim=16\n", model.num_tasks());

    // Step 2: Generate synthetic training data
    println!("Step 2: Generating training data...");
    let n_train = 200;
    let features = generate_training_features(n_train, 20);
    let labels = generate_training_labels(n_train);
    println!("  Training samples: {}\n", n_train);

    // Step 3: Train with gradient harmonization
    println!("Step 3: Training with gradient harmonization...");
    let trainer_config = TrainerConfig {
        epochs: 20,
        batch_size: 32,
        patience: 5,
        log_interval: 5,
        ..Default::default()
    };
    let trainer = MultiTaskTrainer::new(trainer_config);
    let train_result = trainer.train(&mut model, &features, &labels);

    println!("  Epochs completed: {}", train_result.epochs.len());
    println!("  Final loss: {:.4}", train_result.final_loss());
    println!("  Early stopped: {}", train_result.early_stopped);
    println!("  Final task weights:");
    let task_names = ["Trend", "Volatility", "Regime", "Risk"];
    for (i, w) in train_result.final_weights.iter().enumerate() {
        if i < task_names.len() {
            println!("    {}: {:.3}", task_names[i], w);
        }
    }
    println!();

    // Step 4: Generate predictions on test data
    println!("Step 4: Generating predictions on test data...");
    let n_test = 100;
    let test_features = generate_training_features(n_test, 20);
    let fused = model.predict(&test_features);

    let signal_gen = SignalGenerator::new(SignalConfig::default());
    let signals = signal_gen.generate(&fused);

    // Signal distribution
    let mut buy_count = 0;
    let mut sell_count = 0;
    let mut hold_count = 0;
    for s in &signals {
        match s.signal_type {
            task_agnostic_trading::strategy::SignalType::StrongBuy
            | task_agnostic_trading::strategy::SignalType::Buy => buy_count += 1,
            task_agnostic_trading::strategy::SignalType::StrongSell
            | task_agnostic_trading::strategy::SignalType::Sell => sell_count += 1,
            task_agnostic_trading::strategy::SignalType::Hold => hold_count += 1,
        }
    }
    println!("  Signal distribution:");
    println!("    Buy:  {} ({:.0}%)", buy_count, buy_count as f64 / n_test as f64 * 100.0);
    println!("    Sell: {} ({:.0}%)", sell_count, sell_count as f64 / n_test as f64 * 100.0);
    println!("    Hold: {} ({:.0}%)", hold_count, hold_count as f64 / n_test as f64 * 100.0);
    println!();

    // Step 5: Backtest
    println!("Step 5: Backtesting strategy...");
    let prices = generate_price_series(n_test, 100.0);
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        transaction_cost: 0.001,
        slippage: 0.0005,
        periods_per_year: 365.0,
        ..Default::default()
    };
    let engine = BacktestEngine::new(backtest_config);
    let result = engine.run(&signals, &prices);

    println!("  --- Backtest Results ---");
    println!("  Total Return:      {:.2}%", result.metrics.total_return_pct);
    println!("  Sharpe Ratio:      {:.2}", result.metrics.sharpe_ratio);
    println!("  Sortino Ratio:     {:.2}", result.metrics.sortino_ratio);
    println!("  Max Drawdown:      {:.2}%", result.metrics.max_drawdown_pct);
    println!("  Win Rate:          {:.1}%", result.metrics.win_rate);
    println!("  Profit Factor:     {:.2}", result.metrics.profit_factor);
    println!("  Total Trades:      {}", result.metrics.total_trades);
    println!("  Avg Trade Return:  {:.2}%", result.metrics.avg_trade_return);
    println!("  Final Equity:      ${:.2}", result.metrics.final_equity);

    // Step 6: Compare with buy-and-hold
    println!("\n  --- Benchmark: Buy & Hold ---");
    let bh_return = (prices.last().unwrap() - prices.first().unwrap()) / prices.first().unwrap() * 100.0;
    println!("  Buy & Hold Return: {:.2}%", bh_return);
    println!(
        "  Strategy vs B&H:   {:+.2}%",
        result.metrics.total_return_pct - bh_return
    );

    println!("\n=== Strategy Example Complete ===");
}

fn generate_training_features(n: usize, dim: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, dim), |(i, j)| {
        let seed = (i * 31 + j * 17 + 42) as f64;
        ((seed * 0.1).sin() + 1.0) / 2.0 * match j % 4 {
            0 => 0.1,
            1 => 0.5,
            2 => 0.3,
            _ => 1.0,
        }
    })
}

fn generate_training_labels(n: usize) -> Vec<(TaskType, Array2<f64>)> {
    vec![
        (
            TaskType::TrendPrediction,
            Array2::from_shape_fn((n, 3), |(i, j)| {
                if j == i % 3 { 1.0 } else { 0.0 }
            }),
        ),
        (
            TaskType::VolatilityForecast,
            Array2::from_shape_fn((n, 1), |(i, _)| {
                (i as f64 * 0.1).sin().abs() * 0.5
            }),
        ),
        (
            TaskType::RegimeDetection,
            Array2::from_shape_fn((n, 4), |(i, j)| {
                if j == i % 4 { 1.0 } else { 0.0 }
            }),
        ),
        (
            TaskType::RiskAssessment,
            Array2::from_shape_fn((n, 3), |(i, j)| {
                if j == i % 3 { 1.0 } else { 0.0 }
            }),
        ),
    ]
}

fn generate_price_series(n: usize, start_price: f64) -> Vec<f64> {
    let mut prices = vec![start_price];
    for i in 1..n {
        let seed = (i * 73 + 11) as f64;
        let return_val = (seed * 0.1).sin() * 0.02;
        let prev = prices[i - 1];
        prices.push(prev * (1.0 + return_val));
    }
    prices
}
