//! Example: Backtest flow-based trading strategy
//!
//! This example demonstrates how to backtest a flow-based trading
//! strategy using synthetic market data.

use flow_models_trading::prelude::*;
use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};
use chrono::{DateTime, Utc, Duration};

fn main() {
    // Initialize logging
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("        Flow Models Trading - Strategy Backtest");
    println!("═══════════════════════════════════════════════════════════════");

    // Generate synthetic market data
    println!("\n1. Generating synthetic market data...");
    let (features, prices, timestamps, returns) = generate_market_data(1000, 8);

    println!("   Total samples: {}", features.nrows());
    println!("   Feature dimension: {}", features.ncols());
    println!("   Date range: {} to {}",
             timestamps.first().map(|t| t.format("%Y-%m-%d").to_string()).unwrap_or_default(),
             timestamps.last().map(|t| t.format("%Y-%m-%d").to_string()).unwrap_or_default());

    // Split into train and test
    let train_size = (features.nrows() as f64 * 0.7) as usize;

    let train_features = features.slice(ndarray::s![0..train_size, ..]).to_owned();
    let train_returns = returns.slice(ndarray::s![0..train_size]).to_owned();

    let test_features = features.slice(ndarray::s![train_size.., ..]).to_owned();
    let test_prices = prices.slice(ndarray::s![train_size..]).to_owned();
    let test_timestamps: Vec<_> = timestamps[train_size..].to_vec();

    println!("\n   Train samples: {}", train_features.nrows());
    println!("   Test samples: {}", test_features.nrows());

    // Create and configure flow model
    println!("\n2. Creating flow model...");
    let config = FlowConfig::default()
        .with_input_dim(8)
        .with_num_layers(4)
        .with_hidden_dim(64);

    let mut model = NormalizingFlow::new(config);

    // Initialize by training briefly
    println!("\n3. Training flow model...");
    let _ = model.train(&train_features, 20);

    // Create trading strategy
    println!("\n4. Creating trading strategy...");
    let mut strategy = FlowTradingStrategy::new(model)
        .with_confidence_threshold(0.5)
        .with_n_regimes(4)
        .with_anomaly_percentile(5.0);

    // Fit strategy
    println!("\n5. Fitting strategy...");
    strategy.fit(&train_features, Some(&train_returns));

    // Configure backtest
    println!("\n6. Configuring backtest...");
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.1,
        transaction_cost_bps: 10.0,
        slippage_bps: 5.0,
    };

    println!("   Initial capital: ${:,.0}", backtest_config.initial_capital);
    println!("   Position size: {}%", backtest_config.position_size * 100.0);
    println!("   Transaction cost: {} bps", backtest_config.transaction_cost_bps);
    println!("   Slippage: {} bps", backtest_config.slippage_bps);

    // Run backtest
    println!("\n7. Running backtest...");
    let mut engine = BacktestEngine::new(backtest_config);
    let report = engine.run(&mut strategy, &test_features, &test_prices, &test_timestamps);

    // Display results
    println!("\n{}", report);

    // Show sample signals
    println!("\n8. Sample signals from the strategy:");
    println!("   {:>5} {:>12} {:>12} {:>15} {:>15}",
             "Index", "Price", "Signal", "Confidence", "Regime");
    println!("   {:-<65}", "");

    for i in 0..test_features.nrows().min(15) {
        let signal = strategy.generate_signal(&test_features.row(i).to_owned());
        let signal_str = format!("{:?}", signal.signal_type);
        let regime = signal.regime.as_deref().unwrap_or("Unknown");

        println!("   {:>5} ${:>11.2} {:>12} {:>15.2} {:>15}",
                 i,
                 test_prices[i],
                 signal_str,
                 signal.confidence,
                 regime);
    }

    // Performance summary
    println!("\n9. Performance Summary:");
    println!("   ─────────────────────────────────────────────────────");

    if report.is_profitable() {
        println!("   [OK] Strategy is profitable");
    } else {
        println!("   [!!] Strategy lost money");
    }

    if report.has_acceptable_sharpe() {
        println!("   [OK] Sharpe ratio > 1.0 (good risk-adjusted returns)");
    } else {
        println!("   [--] Sharpe ratio < 1.0 (needs improvement)");
    }

    if report.has_manageable_drawdown() {
        println!("   [OK] Max drawdown < 20% (manageable)");
    } else {
        println!("   [!!] Max drawdown >= 20% (high risk)");
    }

    println!("   ─────────────────────────────────────────────────────");
    println!("   Quality Score: {:.1}/100", report.quality_score());

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    Backtest complete!");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Generate synthetic market data
fn generate_market_data(
    n_samples: usize,
    dim: usize,
) -> (Array2<f64>, Array1<f64>, Vec<DateTime<Utc>>, Array1<f64>) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    // Generate features
    let mut features = Array2::zeros((n_samples, dim));
    for i in 0..n_samples {
        // Simulate different regimes
        let regime = (i / (n_samples / 4)) % 4;
        let (mean, vol) = match regime {
            0 => (0.001, 0.02),  // Calm bull
            1 => (-0.001, 0.03), // Volatile bear
            2 => (0.002, 0.025), // Strong bull
            _ => (-0.002, 0.035), // Crash
        };

        for j in 0..dim {
            features[[i, j]] = normal.sample(&mut rng) * vol + mean;
        }
    }

    // Generate prices (random walk)
    let mut prices = Array1::zeros(n_samples);
    let mut returns = Array1::zeros(n_samples);

    prices[0] = 100.0;
    for i in 1..n_samples {
        let ret = normal.sample(&mut rng) * 0.02 + 0.0001; // Small positive drift
        returns[i] = ret;
        prices[i] = prices[i - 1] * (1.0 + ret);
    }

    // Generate timestamps (hourly)
    let start_time = Utc::now() - Duration::hours(n_samples as i64);
    let timestamps: Vec<DateTime<Utc>> = (0..n_samples)
        .map(|i| start_time + Duration::hours(i as i64))
        .collect();

    (features, prices, timestamps, returns)
}
