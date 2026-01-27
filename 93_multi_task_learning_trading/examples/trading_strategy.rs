//! Trading strategy example with backtesting.
//!
//! Demonstrates end-to-end workflow: data generation, feature
//! engineering, model training, strategy execution, and backtesting.

use mtl_trading::prelude::*;
use mtl_trading::data::bybit::SimulatedDataGenerator;
use mtl_trading::data::features::FeatureGenerator;
use mtl_trading::mtl::trainer::{MTLTrainer, TrainingBatch};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== Multi-Task Learning: Trading Strategy & Backtest ===\n");

    // --- Training Phase ---
    println!("--- Training Phase ---");
    let train_klines = SimulatedDataGenerator::generate_regime_klines(500, 50000.0);
    let feature_gen = FeatureGenerator::new(20);

    let features = feature_gen.generate_features(&train_klines);
    let targets = feature_gen.generate_targets(&train_klines, 5);
    let min_len = features.len().min(targets.len());

    let batch = TrainingBatch {
        features: features[..min_len].iter().map(|f| f.features.clone()).collect(),
        targets: targets[..min_len].iter().map(|t| t.targets.clone()).collect(),
    };

    let model = MultiTaskModel::new(10, &[64, 32], 16);
    let mut trainer = MTLTrainer::new(model, 0.005, true);

    println!("Training on {} samples with regime changes...", min_len);
    let _ = trainer.train(&batch, 40, 32, 20).unwrap();
    println!("Training complete.\n");

    // --- Backtesting Phase ---
    println!("--- Backtesting Phase ---");
    let test_klines = SimulatedDataGenerator::generate_regime_klines(300, 50000.0);
    let test_features = feature_gen.generate_features(&test_klines);
    let n = test_features.len();

    // Map features to prices
    let feature_timestamps: Vec<i64> = test_features.iter().map(|f| f.timestamp).collect();
    let test_prices: Vec<f64> = test_klines.iter()
        .filter(|k| feature_timestamps.contains(&k.timestamp))
        .map(|k| k.close)
        .collect();
    let test_len = n.min(test_prices.len());

    let trained_model = trainer.into_model();
    let engine = BacktestEngine::new(
        trained_model,
        0.001,  // return threshold
        0.55,   // direction threshold
        0.05,   // volatility cap
        0.001,  // transaction cost
    );

    let results = engine.run(
        &test_features[..test_len],
        &test_prices[..test_len],
        10000.0,
    );

    println!("Backtest completed: {} steps", results.len());

    // Compute and display metrics
    let metrics = engine.compute_metrics(&results);
    println!("\n--- Performance Metrics ---");
    println!("  Total Return:          {:.2}%", metrics.total_return * 100.0);
    println!("  Annualized Return:     {:.2}%", metrics.annualized_return * 100.0);
    println!("  Annualized Volatility: {:.2}%", metrics.annualized_volatility * 100.0);
    println!("  Sharpe Ratio:          {:.4}", metrics.sharpe_ratio);
    println!("  Sortino Ratio:         {:.4}", metrics.sortino_ratio);
    println!("  Max Drawdown:          {:.2}%", metrics.max_drawdown * 100.0);
    println!("  Win Rate:              {:.2}%", metrics.win_rate * 100.0);
    println!("  Profit Factor:         {:.4}", metrics.profit_factor);
    println!("  Number of Trades:      {}", metrics.num_trades);

    // Capital curve summary
    if let (Some(first), Some(last)) = (results.first(), results.last()) {
        println!("\n--- Capital Curve ---");
        println!("  Start: ${:.2}", first.capital);
        println!("  End:   ${:.2}", last.capital);
    }

    println!("\nDone!");
}
