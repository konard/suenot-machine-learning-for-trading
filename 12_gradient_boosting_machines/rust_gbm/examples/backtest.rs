//! Example: Backtesting a Long-Short Strategy
//!
//! Run with: cargo run --example backtest

use anyhow::Result;
use rust_gbm::{
    data::{BybitClient, Dataset, Interval},
    features::FeatureEngineer,
    models::{GbmParams, GbmRegressor},
    strategies::{print_backtest_summary, LongShortStrategy, StrategyConfig},
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("Backtesting Example");
    println!("{}", "=".repeat(40));

    // Fetch data
    println!("\n📥 Fetching historical data...");
    let client = BybitClient::new();
    let candles = client
        .get_klines("BTCUSDT", Interval::Hour4, Some(1000), None, None)
        .await?;
    println!("   Fetched {} candles (4-hour timeframe)", candles.len());

    // Engineer features
    let engineer = FeatureEngineer::new();
    let dataset = engineer.build_clean_features(&candles);

    // Split: 60% train, 40% backtest
    let split_idx = (dataset.len() as f64 * 0.6) as usize;

    let train = Dataset {
        feature_names: dataset.feature_names.clone(),
        features: dataset.features[..split_idx].to_vec(),
        targets: dataset.targets[..split_idx].to_vec(),
        timestamps: dataset.timestamps[..split_idx].to_vec(),
        symbol: dataset.symbol.clone(),
    };

    let test = Dataset {
        feature_names: dataset.feature_names.clone(),
        features: dataset.features[split_idx..].to_vec(),
        targets: dataset.targets[split_idx..].to_vec(),
        timestamps: dataset.timestamps[split_idx..].to_vec(),
        symbol: dataset.symbol.clone(),
    };

    println!("\n📊 Data Split:");
    println!("   Training period: {} samples", train.len());
    println!("   Backtest period: {} samples", test.len());

    // Train model
    println!("\n🤖 Training model...");
    let params = GbmParams {
        n_estimators: 100,
        max_depth: 4,
        learning_rate: 0.1,
        min_samples_split: 10,
        min_samples_leaf: 5,
        subsample: 0.8,
    };

    let mut model = GbmRegressor::with_params(params);
    model.fit(&train)?;

    // Get prices for backtest
    // Account for the feature lookback period
    let feature_lookback = 50; // Approximate lookback for feature calculation
    let close_prices: Vec<f64> = candles[split_idx + feature_lookback..]
        .iter()
        .take(test.len())
        .map(|c| c.close)
        .collect();

    // Test different strategy configurations
    let strategy_configs = vec![
        ("Conservative", StrategyConfig {
            long_threshold: 0.3,
            short_threshold: 0.3,
            initial_capital: 10000.0,
            position_size: 0.3,
            fee_rate: 0.001,
            max_positions: 1,
        }),
        ("Moderate", StrategyConfig {
            long_threshold: 0.1,
            short_threshold: 0.1,
            initial_capital: 10000.0,
            position_size: 0.5,
            fee_rate: 0.001,
            max_positions: 1,
        }),
        ("Aggressive", StrategyConfig {
            long_threshold: 0.05,
            short_threshold: 0.05,
            initial_capital: 10000.0,
            position_size: 0.8,
            fee_rate: 0.001,
            max_positions: 1,
        }),
    ];

    // Ensure matching lengths
    let test_len = test.len().min(close_prices.len());
    let test_subset = Dataset {
        feature_names: test.feature_names.clone(),
        features: test.features[..test_len].to_vec(),
        targets: test.targets[..test_len].to_vec(),
        timestamps: test.timestamps[..test_len].to_vec(),
        symbol: test.symbol.clone(),
    };
    let prices_subset = &close_prices[..test_len];

    println!("\n💰 Running backtests with different configurations...\n");
    println!("{:<15} {:>12} {:>12} {:>10} {:>10}",
        "Strategy", "Return %", "Max DD %", "Sharpe", "# Trades");
    println!("{}", "-".repeat(65));

    for (name, config) in strategy_configs {
        let mut strategy = LongShortStrategy::with_config(config);

        // Clone model for each strategy test
        let mut model_clone = GbmRegressor::with_params(GbmParams {
            n_estimators: 100,
            max_depth: 4,
            learning_rate: 0.1,
            min_samples_split: 10,
            min_samples_leaf: 5,
            subsample: 0.8,
        });
        model_clone.fit(&train)?;
        strategy.set_model(model_clone);

        let metrics = strategy.backtest(&test_subset, prices_subset)?;

        println!("{:<15} {:>11.2}% {:>11.2}% {:>10.2} {:>10}",
            name,
            metrics.total_return,
            metrics.max_drawdown,
            metrics.sharpe_ratio,
            metrics.num_trades);
    }

    // Detailed results for moderate strategy
    println!("\n📈 Detailed Results for Moderate Strategy:");
    let mut strategy = LongShortStrategy::with_config(StrategyConfig {
        long_threshold: 0.1,
        short_threshold: 0.1,
        initial_capital: 10000.0,
        position_size: 0.5,
        fee_rate: 0.001,
        max_positions: 1,
    });

    let mut model_final = GbmRegressor::with_params(GbmParams {
        n_estimators: 100,
        max_depth: 4,
        learning_rate: 0.1,
        min_samples_split: 10,
        min_samples_leaf: 5,
        subsample: 0.8,
    });
    model_final.fit(&train)?;
    strategy.set_model(model_final);

    let metrics = strategy.backtest(&test_subset, prices_subset)?;
    print_backtest_summary(&metrics);

    // Compare with buy & hold
    let buy_hold = if let (Some(first), Some(last)) = (prices_subset.first(), prices_subset.last()) {
        (last - first) / first * 100.0
    } else {
        0.0
    };

    println!("\n📊 vs Buy & Hold:");
    println!("   Strategy: {:.2}%", metrics.total_return);
    println!("   Buy & Hold: {:.2}%", buy_hold);
    println!("   Alpha: {:.2}%", metrics.total_return - buy_hold);

    Ok(())
}
