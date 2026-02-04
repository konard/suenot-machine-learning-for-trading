//! Example: Backtesting a Deep Ensemble trading strategy
//!
//! This example demonstrates how to backtest a trading strategy
//! using the Deep Ensemble model with uncertainty-aware position sizing.

use deep_ensembles_trading::backtest::engine::{BacktestConfig, BacktestEngine};
use deep_ensembles_trading::ensemble::config::EnsembleConfig;
use deep_ensembles_trading::ensemble::model::DeepEnsemble;
use deep_ensembles_trading::strategy::uncertainty::UncertaintyStrategy;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

fn main() {
    println!("=== Deep Ensemble Backtest Example ===\n");

    // Step 1: Generate synthetic data
    println!("1. Generating synthetic market data...");
    let (features, prices, targets) = generate_synthetic_data(1000, 8);
    println!("   Generated {} time steps with {} features", features.nrows(), features.ncols());

    // Split data
    let train_end = (features.nrows() as f64 * 0.6) as usize;
    let val_end = (features.nrows() as f64 * 0.8) as usize;

    let x_train = features.slice(ndarray::s![..train_end, ..]).to_owned();
    let y_train = targets.slice(ndarray::s![..train_end]).to_owned();
    let x_val = features.slice(ndarray::s![train_end..val_end, ..]).to_owned();
    let y_val = targets.slice(ndarray::s![train_end..val_end]).to_owned();
    let x_test = features.slice(ndarray::s![val_end.., ..]).to_owned();
    let prices_test = &prices[val_end..];

    println!("   Train: {}, Val: {}, Test: {}", x_train.nrows(), x_val.nrows(), x_test.nrows());

    // Step 2: Create and train ensemble
    println!("\n2. Training Deep Ensemble...");
    let config = EnsembleConfig {
        num_models: 5,
        input_dim: 8,
        hidden_dims: vec![64, 32],
        learning_rate: 0.005,
        batch_size: 32,
        max_epochs: 50,
        early_stopping_patience: 10,
        ..EnsembleConfig::default()
    };

    let mut ensemble = DeepEnsemble::new(config);
    let history = ensemble.train(&x_train, &y_train, Some(&x_val), Some(&y_val), true);
    println!("   Training complete. Final losses: {:?}", history);

    // Step 3: Evaluate model on test set
    println!("\n3. Evaluating model...");
    let predictions = ensemble.predict(&x_test);

    println!("   Prediction statistics:");
    println!("   Mean prediction range: [{:.4}, {:.4}]",
             predictions.mean.iter().cloned().fold(f64::INFINITY, f64::min),
             predictions.mean.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!("   Avg total uncertainty: {:.4}",
             predictions.total_std.iter().sum::<f64>() / predictions.num_samples() as f64);
    println!("   Avg epistemic uncertainty: {:.4}",
             predictions.epistemic_std.iter().sum::<f64>() / predictions.num_samples() as f64);

    // Step 4: Create strategy
    println!("\n4. Creating trading strategy...");
    let strategy = UncertaintyStrategy::new(
        0.6,  // confidence threshold
        0.5,  // epistemic threshold
        1.2,  // signal strength threshold
        0.1,  // max position size
    );

    // Step 5: Run backtest
    println!("\n5. Running backtest...");
    let backtest_config = BacktestConfig {
        initial_capital: 100000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
    };

    let engine = BacktestEngine::new(backtest_config);

    // Recreate ensemble for fresh predictions during backtest
    let mut ensemble_for_backtest = DeepEnsemble::new(EnsembleConfig {
        num_models: 5,
        input_dim: 8,
        hidden_dims: vec![64, 32],
        ..EnsembleConfig::default()
    });
    ensemble_for_backtest.train(&x_train, &y_train, Some(&x_val), Some(&y_val), false);

    let report = engine.run(
        &mut ensemble_for_backtest,
        &strategy,
        &x_test,
        prices_test,
        "BTCUSDT",
    );

    // Step 6: Print results
    println!("{}", report.summary());

    // Step 7: Compare with buy-and-hold
    println!("6. Comparison with Buy-and-Hold:");
    let bh_return = (prices_test.last().unwrap() - prices_test.first().unwrap())
        / prices_test.first().unwrap();
    let strategy_return = report.metrics.total_return;

    println!("   Strategy Return:     {:.2}%", strategy_return * 100.0);
    println!("   Buy-and-Hold Return: {:.2}%", bh_return * 100.0);
    println!("   Outperformance:      {:.2}%", (strategy_return - bh_return) * 100.0);

    println!("\n=== Backtest complete! ===");
}

/// Generate synthetic market data for backtesting
fn generate_synthetic_data(n: usize, n_features: usize) -> (Array2<f64>, Vec<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate features
    let features = Array2::from_shape_fn((n, n_features), |_| normal.sample(&mut rng));

    // Generate price series with trend and mean reversion
    let mut prices = Vec::with_capacity(n);
    let mut price = 100.0;

    for i in 0..n {
        let trend = 0.0001;
        let volatility = 0.02;
        let mean_reversion = 0.01 * (100.0 - price) / price;

        let return_pct = trend + mean_reversion + volatility * rng.gen::<f64>() * 2.0 - volatility;
        price *= 1.0 + return_pct;
        prices.push(price);
    }

    // Generate targets (future returns)
    let horizon = 5;
    let mut targets = Array1::zeros(n);
    for i in 0..n.saturating_sub(horizon) {
        targets[i] = (prices[i + horizon] - prices[i]) / prices[i];
    }

    (features, prices, targets)
}
