//! Backtest a Random Forest trading strategy
//!
//! Usage: cargo run --bin backtest -- --symbol BTCUSDT --days 365

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use crypto_ml::api::{BybitClient, Interval, Symbol};
use crypto_ml::backtest::{Backtest, BacktestConfig};
use crypto_ml::features::FeatureEngine;
use crypto_ml::models::random_forest::ForestConfig;
use crypto_ml::models::{RandomForest, TaskType};
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "Backtest Random Forest trading strategy")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of days of data
    #[arg(short, long, default_value = "365")]
    days: i64,

    /// Number of trees
    #[arg(short, long, default_value = "50")]
    trees: usize,

    /// Initial capital
    #[arg(long, default_value = "10000")]
    capital: f64,

    /// Transaction cost (%)
    #[arg(long, default_value = "0.1")]
    fee: f64,

    /// Allow short selling
    #[arg(long)]
    allow_short: bool,

    /// Train/test split ratio
    #[arg(long, default_value = "0.7")]
    train_ratio: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("crypto_ml=info")
        .init();

    let args = Args::parse();

    println!("╔═══════════════════════════════════════════╗");
    println!("║  Random Forest Trading Strategy Backtest  ║");
    println!("╚═══════════════════════════════════════════╝\n");

    // Fetch data
    info!("Fetching {} data for {} days...", args.symbol, args.days);

    let client = BybitClient::new();
    let symbol = Symbol::new(&args.symbol);

    let end = Utc::now();
    let start = end - Duration::days(args.days);

    let candles = client
        .get_historical_klines(&symbol, Interval::Hour4, start, end)
        .await?;

    println!("Fetched {} candles (4h timeframe)\n", candles.len());

    // Generate features
    info!("Generating features...");
    let engine = FeatureEngine::new().with_horizon(1);
    let mut dataset = engine.generate(&candles);
    dataset.to_binary_classification();

    println!(
        "Dataset: {} samples, {} features\n",
        dataset.n_samples(),
        dataset.n_features()
    );

    // Split into train/test
    let train_size = (args.train_ratio * dataset.n_samples() as f64) as usize;
    let split = dataset.train_test_split(1.0 - args.train_ratio);

    println!("Training period: {} samples", split.train.n_samples());
    println!("Testing period:  {} samples\n", split.test.n_samples());

    // Train Random Forest
    println!("Training Random Forest ({} trees)...", args.trees);

    let config = ForestConfig {
        n_trees: args.trees,
        max_depth: 6,
        min_samples_split: 20,
        min_samples_leaf: 10,
        max_features: None,
        bootstrap: true,
        seed: 42,
        task: TaskType::Classification,
        oob_score: true,
    };

    let start_time = std::time::Instant::now();
    let mut forest = RandomForest::new(config);
    forest.fit(&split.train);
    let training_time = start_time.elapsed();

    println!("Training completed in {:.2}s\n", training_time.as_secs_f64());

    // Model performance
    println!("=== Model Performance ===\n");

    let train_acc = forest.accuracy(&split.train);
    let test_acc = forest.accuracy(&split.test);

    println!("Training Accuracy: {:.2}%", train_acc * 100.0);
    println!("Test Accuracy:     {:.2}%", test_acc * 100.0);

    if let Some(oob) = forest.oob_score() {
        println!("OOB Score:         {:.2}%", oob * 100.0);
    }

    // Generate predictions for backtest
    let test_predictions = forest.predict(&split.test);

    // Map predictions to probabilities for signal generation
    let test_proba: Vec<Vec<f64>> = forest.predict_proba(&split.test);

    // Use probability of class 1 (up) as prediction strength
    let predictions: Vec<f64> = test_proba.iter().map(|p| p[1] - 0.5).collect();

    // Get test candles (offset by lookback)
    let lookback = 50; // Approximate lookback from feature generation
    let test_start_idx = train_size + lookback;
    let test_candles = &candles[test_start_idx..test_start_idx + split.test.n_samples()];

    // Run backtest
    println!("\n=== Running Backtest ===\n");

    let backtest_config = BacktestConfig {
        initial_capital: args.capital,
        position_size: 1.0,
        transaction_cost: args.fee / 100.0,
        slippage: 0.0005,
        long_threshold: 0.1,  // Go long if prob > 0.6
        short_threshold: -0.1, // Go short if prob < 0.4
        allow_short: args.allow_short,
    };

    let backtest = Backtest::new(backtest_config);
    let result = backtest.run(test_candles, &predictions);

    // Print results
    result.print_summary();

    // Trade statistics
    let trade_stats = result.trade_stats();
    trade_stats.print();

    // Buy & Hold comparison
    println!("\n=== Buy & Hold Comparison ===\n");

    let first_price = test_candles.first().map(|c| c.close).unwrap_or(1.0);
    let last_price = test_candles.last().map(|c| c.close).unwrap_or(1.0);
    let buy_hold_return = (last_price - first_price) / first_price;

    println!(
        "Strategy Return: {:>10.2}%",
        result.metrics.total_return * 100.0
    );
    println!("Buy & Hold Return: {:>8.2}%", buy_hold_return * 100.0);
    println!(
        "Outperformance:    {:>8.2}%",
        (result.metrics.total_return - buy_hold_return) * 100.0
    );

    // Risk comparison
    println!("\n=== Risk Metrics ===\n");

    println!(
        "Strategy Sharpe:      {:>8.2}",
        result.metrics.sharpe_ratio
    );
    println!(
        "Strategy Max Drawdown: {:>7.2}%",
        result.metrics.max_drawdown * 100.0
    );

    // Equity curve summary
    println!("\n=== Equity Curve ===\n");

    let equity = &result.equity_curve;
    let n_points = 10;
    let step = equity.len() / n_points;

    println!("Period    │ Capital");
    println!("──────────┼───────────");

    for i in (0..equity.len()).step_by(step.max(1)) {
        let pct = i as f64 / equity.len() as f64 * 100.0;
        println!("{:>6.1}%   │ ${:.2}", pct, equity[i]);
    }
    println!("{:>6.1}%   │ ${:.2}", 100.0, equity.last().unwrap_or(&0.0));

    // Feature importance for trading
    println!("\n=== Most Important Features for Trading ===\n");

    let ranking = forest.feature_importance_ranking();
    for (i, (name, imp)) in ranking.iter().take(10).enumerate() {
        let bar = "█".repeat((imp * 30.0) as usize);
        println!("{:2}. {:25} {:.4} {}", i + 1, name, imp, bar);
    }

    // Summary
    println!("\n╔═══════════════════════════════════════════╗");
    println!("║              Backtest Summary             ║");
    println!("╠═══════════════════════════════════════════╣");
    println!(
        "║  Initial Capital:    ${:>15.2}   ║",
        args.capital
    );
    println!(
        "║  Final Capital:      ${:>15.2}   ║",
        result.final_capital
    );
    println!(
        "║  Total Return:       {:>15.2}%   ║",
        result.metrics.total_return * 100.0
    );
    println!(
        "║  Sharpe Ratio:       {:>16.2}   ║",
        result.metrics.sharpe_ratio
    );
    println!(
        "║  Max Drawdown:       {:>15.2}%   ║",
        result.metrics.max_drawdown * 100.0
    );
    println!(
        "║  Win Rate:           {:>15.2}%   ║",
        result.metrics.win_rate * 100.0
    );
    println!(
        "║  Total Trades:       {:>17}   ║",
        result.trades.len()
    );
    println!("╚═══════════════════════════════════════════╝");

    Ok(())
}
