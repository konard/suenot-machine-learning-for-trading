//! Main entry point for the Rust GBM trading demo
//!
//! This program demonstrates how to:
//! 1. Fetch cryptocurrency data from Bybit
//! 2. Engineer features from OHLCV data
//! 3. Train a Gradient Boosting Machine model
//! 4. Backtest a long-short trading strategy

use anyhow::Result;
use chrono::{Duration, Utc};
use rust_gbm::{
    data::{BybitClient, Interval},
    features::{FeatureConfig, FeatureEngineer},
    models::{time_series_cv, GbmParams, GbmRegressor},
    strategies::{print_backtest_summary, LongShortStrategy, StrategyConfig},
};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// Available demo modes
#[derive(Debug, Clone, Copy)]
enum DemoMode {
    /// Fetch data and display statistics
    FetchData,
    /// Train a model and show evaluation metrics
    TrainModel,
    /// Run a full backtest
    Backtest,
    /// Run all demos
    All,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    println!("\n{}", "=".repeat(60));
    println!("  Rust Gradient Boosting Machine for Crypto Trading");
    println!("  Data Source: Bybit Exchange");
    println!("{}\n", "=".repeat(60));

    // Run all demos
    run_demo(DemoMode::All).await
}

async fn run_demo(mode: DemoMode) -> Result<()> {
    match mode {
        DemoMode::FetchData => demo_fetch_data().await,
        DemoMode::TrainModel => demo_train_model().await,
        DemoMode::Backtest => demo_backtest().await,
        DemoMode::All => {
            demo_fetch_data().await?;
            println!("\n");
            demo_train_model().await?;
            println!("\n");
            demo_backtest().await?;
            Ok(())
        }
    }
}

/// Demo: Fetch data from Bybit and display statistics
async fn demo_fetch_data() -> Result<()> {
    println!("📊 Demo 1: Fetching Data from Bybit");
    println!("{}", "-".repeat(40));

    let client = BybitClient::new();

    // Fetch recent 1-hour candles for BTC/USDT
    let symbol = "BTCUSDT";
    let interval = Interval::Hour1;
    let limit = 500;

    info!("Fetching {} {} candles for {}", limit, "1H", symbol);

    let candles = client
        .get_klines(symbol, interval, Some(limit), None, None)
        .await?;

    println!("\n✅ Fetched {} candles", candles.len());

    if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
        println!("   Period: {} to {}",
            first.timestamp.format("%Y-%m-%d %H:%M"),
            last.timestamp.format("%Y-%m-%d %H:%M"));

        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_price = prices.iter().sum::<f64>() / prices.len() as f64;

        println!("\n   Price Statistics:");
        println!("   - Min:  ${:.2}", min_price);
        println!("   - Max:  ${:.2}", max_price);
        println!("   - Avg:  ${:.2}", avg_price);
        println!("   - Last: ${:.2}", last.close);

        let total_volume: f64 = candles.iter().map(|c| c.volume).sum();
        println!("\n   Volume Statistics:");
        println!("   - Total: {:.2} BTC", total_volume);
        println!("   - Avg per candle: {:.4} BTC", total_volume / candles.len() as f64);
    }

    // Also fetch order book
    println!("\n   Fetching order book...");
    let orderbook = client.get_orderbook(symbol, Some(10)).await?;
    if let (Some(bid), Some(ask)) = (orderbook.best_bid(), orderbook.best_ask()) {
        println!("   - Best Bid: ${:.2}", bid);
        println!("   - Best Ask: ${:.2}", ask);
        println!("   - Spread: ${:.2} ({:.4}%)",
            ask - bid,
            (ask - bid) / bid * 100.0);
    }

    Ok(())
}

/// Demo: Train a GBM model and evaluate performance
async fn demo_train_model() -> Result<()> {
    println!("🤖 Demo 2: Training Gradient Boosting Model");
    println!("{}", "-".repeat(40));

    // Fetch data
    let client = BybitClient::new();
    let candles = client
        .get_klines("BTCUSDT", Interval::Hour1, Some(1000), None, None)
        .await?;

    println!("   Fetched {} candles for training", candles.len());

    // Engineer features
    let engineer = FeatureEngineer::new();
    let dataset = engineer.build_clean_features(&candles);

    println!("   Created {} samples with {} features",
        dataset.len(),
        dataset.num_features());

    // Split data
    let (train, test) = dataset.train_test_split(0.8);
    println!("   Train set: {} samples", train.len());
    println!("   Test set: {} samples", test.len());

    // Train model
    let params = GbmParams {
        n_estimators: 100,
        max_depth: 4,
        learning_rate: 0.1,
        min_samples_split: 10,
        min_samples_leaf: 5,
        subsample: 0.8,
    };

    println!("\n   Training GBM with parameters:");
    println!("   - n_estimators: {}", params.n_estimators);
    println!("   - max_depth: {}", params.max_depth);
    println!("   - learning_rate: {}", params.learning_rate);

    let mut model = GbmRegressor::with_params(params.clone());
    model.fit(&train)?;

    // Evaluate
    let train_metrics = model.evaluate(&train)?;
    let test_metrics = model.evaluate(&test)?;

    println!("\n   📈 Training Set Metrics:");
    println!("   - RMSE: {:.4}", train_metrics.rmse.unwrap_or(0.0));
    println!("   - R²: {:.4}", train_metrics.r2.unwrap_or(0.0));
    println!("   - Directional Accuracy: {:.2}%",
        train_metrics.directional_accuracy.unwrap_or(0.0));

    println!("\n   📉 Test Set Metrics:");
    println!("   - RMSE: {:.4}", test_metrics.rmse.unwrap_or(0.0));
    println!("   - R²: {:.4}", test_metrics.r2.unwrap_or(0.0));
    println!("   - Directional Accuracy: {:.2}%",
        test_metrics.directional_accuracy.unwrap_or(0.0));

    // Cross-validation
    println!("\n   Running 5-fold time-series cross-validation...");
    let cv_result = time_series_cv(&dataset, &params, 5)?;

    println!("\n   📊 Cross-Validation Results:");
    println!("   - Mean RMSE: {:.4} (±{:.4})",
        cv_result.mean_metrics.rmse.unwrap_or(0.0),
        cv_result.std_metrics.get("rmse").unwrap_or(&0.0));
    println!("   - Mean R²: {:.4}", cv_result.mean_metrics.r2.unwrap_or(0.0));
    println!("   - Mean Dir. Accuracy: {:.2}%",
        cv_result.mean_metrics.directional_accuracy.unwrap_or(0.0));

    Ok(())
}

/// Demo: Run a backtest with the long-short strategy
async fn demo_backtest() -> Result<()> {
    println!("💰 Demo 3: Backtesting Long-Short Strategy");
    println!("{}", "-".repeat(40));

    // Fetch data
    let client = BybitClient::new();
    let candles = client
        .get_klines("BTCUSDT", Interval::Hour1, Some(1000), None, None)
        .await?;

    println!("   Fetched {} candles", candles.len());

    // Engineer features
    let engineer = FeatureEngineer::new();
    let dataset = engineer.build_clean_features(&candles);

    // Split: use first 70% for training, rest for backtesting
    let split_idx = (dataset.len() as f64 * 0.7) as usize;

    let train = rust_gbm::Dataset {
        feature_names: dataset.feature_names.clone(),
        features: dataset.features[..split_idx].to_vec(),
        targets: dataset.targets[..split_idx].to_vec(),
        timestamps: dataset.timestamps[..split_idx].to_vec(),
        symbol: dataset.symbol.clone(),
    };

    let test = rust_gbm::Dataset {
        feature_names: dataset.feature_names.clone(),
        features: dataset.features[split_idx..].to_vec(),
        targets: dataset.targets[split_idx..].to_vec(),
        timestamps: dataset.timestamps[split_idx..].to_vec(),
        symbol: dataset.symbol.clone(),
    };

    println!("   Training period: {} samples", train.len());
    println!("   Backtest period: {} samples", test.len());

    // Train model
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

    // Get close prices for backtest period
    let close_prices: Vec<f64> = candles[split_idx + engineer.feature_names().len()..]
        .iter()
        .take(test.len())
        .map(|c| c.close)
        .collect();

    // Configure strategy
    let strategy_config = StrategyConfig {
        long_threshold: 0.1,    // Go long if predicted return > 0.1%
        short_threshold: 0.1,   // Go short if predicted return < -0.1%
        initial_capital: 10000.0,
        position_size: 0.5,     // Use 50% of capital per trade
        fee_rate: 0.001,        // 0.1% fee (Bybit taker fee)
        max_positions: 1,
    };

    println!("\n   Strategy Configuration:");
    println!("   - Initial Capital: ${:.2}", strategy_config.initial_capital);
    println!("   - Position Size: {:.0}%", strategy_config.position_size * 100.0);
    println!("   - Long Threshold: {:.2}%", strategy_config.long_threshold);
    println!("   - Short Threshold: {:.2}%", strategy_config.short_threshold);
    println!("   - Fee Rate: {:.2}%", strategy_config.fee_rate * 100.0);

    // Run backtest
    let mut strategy = LongShortStrategy::with_config(strategy_config);
    strategy.set_model(model);

    // Ensure we have matching lengths
    let test_len = test.len().min(close_prices.len());
    let test_subset = rust_gbm::Dataset {
        feature_names: test.feature_names.clone(),
        features: test.features[..test_len].to_vec(),
        targets: test.targets[..test_len].to_vec(),
        timestamps: test.timestamps[..test_len].to_vec(),
        symbol: test.symbol.clone(),
    };
    let prices_subset = &close_prices[..test_len];

    let metrics = strategy.backtest(&test_subset, prices_subset)?;

    // Print results
    print_backtest_summary(&metrics);

    // Show sample trades
    let trades = strategy.trades();
    if !trades.is_empty() {
        println!("\n   Sample Trades (last 5):");
        for trade in trades.iter().rev().take(5) {
            let direction = if trade.signal == rust_gbm::Signal::Long { "LONG" } else { "SHORT" };
            println!("   {:5} | Entry: ${:.2} | Exit: ${:.2} | PnL: ${:.2} | Return: {:.2}%",
                direction,
                trade.entry_price,
                trade.exit_price,
                trade.pnl,
                trade.return_pct);
        }
    }

    // Buy and hold comparison
    let buy_hold_return = if let (Some(first), Some(last)) = (close_prices.first(), close_prices.last()) {
        (last - first) / first * 100.0
    } else {
        0.0
    };

    println!("\n   📊 Strategy vs Buy & Hold:");
    println!("   - Strategy Return: {:.2}%", metrics.total_return);
    println!("   - Buy & Hold Return: {:.2}%", buy_hold_return);
    println!("   - Outperformance: {:.2}%", metrics.total_return - buy_hold_return);

    Ok(())
}
