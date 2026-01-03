//! Backtest Example
//!
//! This example demonstrates how to backtest a Neural Spline Flow
//! trading strategy using historical cryptocurrency data.

use ndarray::Array2;
use neural_spline_flows::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== Neural Spline Flow Backtesting Example ===\n");

    // Fetch data from Bybit
    println!("1. Fetching historical data...");
    let client = BybitClient::new();
    let symbol = "BTCUSDT";
    let interval = "60"; // 1 hour

    let candles = match client.get_klines(symbol, interval, 1000).await {
        Ok(c) => {
            println!("   Fetched {} candles from Bybit", c.len());
            c
        }
        Err(e) => {
            println!("   API error: {}, using synthetic data", e);
            generate_synthetic_candles(1000)
        }
    };

    // Split data into train and test
    let split_idx = candles.len() * 7 / 10;
    let train_candles = &candles[..split_idx];
    let test_candles = &candles[split_idx..];

    println!("   Train set: {} candles", train_candles.len());
    println!("   Test set: {} candles", test_candles.len());

    // Extract and normalize features for training
    println!("\n2. Preparing training data...");
    let lookback = 20;
    let train_features: Vec<FeatureVector> = candles_to_features(train_candles, lookback);
    let (normalized_train, mean, std) = normalize_features(&train_features);

    let feature_dim = normalized_train[0].len();
    let n_train = normalized_train.len();

    println!("   Feature dimension: {}", feature_dim);
    println!("   Training samples: {}", n_train);

    // Convert to matrix
    let mut train_matrix = Array2::zeros((n_train, feature_dim));
    for (i, f) in normalized_train.iter().enumerate() {
        for (j, v) in f.values.iter().enumerate() {
            train_matrix[[i, j]] = *v;
        }
    }

    // Create and train model
    println!("\n3. Training Neural Spline Flow model...");
    let nsf_config = NSFConfig::new(feature_dim)
        .with_num_layers(4)
        .with_hidden_dim(64)
        .with_num_bins(8);

    let mut model = NeuralSplineFlow::new(nsf_config);

    match model.fit(&train_matrix) {
        Ok(stats) => {
            println!("   Training complete!");
            println!("   Final NLL: {:.4}", stats.final_loss);
        }
        Err(e) => {
            println!("   Training failed: {}", e);
            return Ok(());
        }
    }

    // Configure backtesting
    println!("\n4. Configuring backtest...");
    let backtest_config = BacktestConfig {
        initial_capital: 10000.0,
        transaction_cost: 0.001, // 0.1%
        slippage: 0.0005, // 0.05%
        lookback,
        warmup: lookback + 10,
        max_position: 1.0,
        signal_config: SignalGeneratorConfig {
            return_feature_idx: 0,
            density_threshold: -15.0,
            confidence_threshold: 0.25,
            z_threshold: 0.3,
            num_samples: 500,
        },
    };

    println!("   Initial capital: ${:.2}", backtest_config.initial_capital);
    println!("   Transaction cost: {:.2}%", backtest_config.transaction_cost * 100.0);
    println!("   Slippage: {:.2}%", backtest_config.slippage * 100.0);

    // Run backtest
    println!("\n5. Running backtest on test data...");
    let engine = BacktestEngine::new(model.clone(), backtest_config);
    let result = engine.run(test_candles);

    // Print results
    print_summary(&result);

    // Additional analysis
    println!("\n6. Detailed Analysis...\n");

    // Equity curve statistics
    if !result.bars.is_empty() {
        let equities: Vec<f64> = result.bars.iter().map(|b| b.equity).collect();
        let final_equity = equities.last().unwrap_or(&10000.0);
        let max_equity = equities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_equity = equities.iter().cloned().fold(f64::INFINITY, f64::min);

        println!("Equity Curve:");
        println!("   Starting: ${:.2}", equities.first().unwrap_or(&10000.0));
        println!("   Final:    ${:.2}", final_equity);
        println!("   Maximum:  ${:.2}", max_equity);
        println!("   Minimum:  ${:.2}", min_equity);
    }

    // Trade analysis
    if !result.trades.is_empty() {
        println!("\nTrade Analysis:");
        println!("   Total trades: {}", result.trades.len());

        let long_trades: Vec<&Trade> = result.trades.iter().filter(|t| t.position_size > 0.0).collect();
        let short_trades: Vec<&Trade> = result.trades.iter().filter(|t| t.position_size < 0.0).collect();

        println!("   Long trades: {}", long_trades.len());
        println!("   Short trades: {}", short_trades.len());

        // Best and worst trades
        if let Some(best) = result.trades.iter().max_by(|a, b| a.pnl.partial_cmp(&b.pnl).unwrap()) {
            println!("   Best trade: {:.2}%", best.return_pct);
        }
        if let Some(worst) = result.trades.iter().min_by(|a, b| a.pnl.partial_cmp(&b.pnl).unwrap()) {
            println!("   Worst trade: {:.2}%", worst.return_pct);
        }
    }

    // Signal distribution
    println!("\nSignal Distribution:");
    let signals: Vec<f64> = result.bars.iter().map(|b| b.signal).collect();
    let long_signals = signals.iter().filter(|&&s| s > 0.0).count();
    let short_signals = signals.iter().filter(|&&s| s < 0.0).count();
    let no_signals = signals.iter().filter(|&&s| s == 0.0).count();

    println!("   Long signals:  {} ({:.1}%)", long_signals, long_signals as f64 / signals.len() as f64 * 100.0);
    println!("   Short signals: {} ({:.1}%)", short_signals, short_signals as f64 / signals.len() as f64 * 100.0);
    println!("   No trade:      {} ({:.1}%)", no_signals, no_signals as f64 / signals.len() as f64 * 100.0);

    // Monthly breakdown (approximate)
    println!("\nMonthly Performance (approximate):");
    let days_per_month = 30 * 24; // hours
    let mut current_month_pnl = 0.0;
    let mut month_count = 1;

    for (i, bar) in result.bars.iter().enumerate() {
        current_month_pnl += bar.pnl;

        if (i + 1) % days_per_month == 0 || i == result.bars.len() - 1 {
            let month_return = current_month_pnl / backtest_config.initial_capital * 100.0;
            println!("   Month {}: {:+.2}%", month_count, month_return);
            current_month_pnl = 0.0;
            month_count += 1;
        }
    }

    // Risk metrics from model
    println!("\n7. Model Risk Metrics...");
    let risk_manager = RiskManager::with_defaults(model);
    let risk_metrics = risk_manager.compute_risk_metrics();

    println!("   Expected Return: {:.4}", risk_metrics.expected_return);
    println!("   Return Std: {:.4}", risk_metrics.return_std);
    println!("   VaR (95%): {:.4}", risk_metrics.var);
    println!("   CVaR (95%): {:.4}", risk_metrics.cvar);
    println!("   Skewness: {:.4}", risk_metrics.skewness);
    println!("   Kurtosis: {:.4}", risk_metrics.kurtosis);

    println!("\n=== Backtest Complete ===");
    Ok(())
}

/// Generate synthetic candles for testing
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use chrono::Utc;
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let now = Utc::now();
    let mut price = 50000.0;

    // Add some trending and mean-reverting behavior
    let mut trend = 0.0;

    (0..n)
        .map(|i| {
            // Trend component (slowly changing)
            if i % 100 == 0 {
                trend = rng.gen_range(-0.001..0.001);
            }

            // Random walk with trend
            let noise = rng.gen_range(-0.015..0.015);
            price *= 1.0 + trend + noise;

            // Keep price reasonable
            price = price.clamp(30000.0, 80000.0);

            let volatility = rng.gen_range(0.005..0.015);
            let high = price * (1.0 + volatility);
            let low = price * (1.0 - volatility);
            let open = price * (1.0 + rng.gen_range(-0.005..0.005));

            Candle {
                timestamp: now - chrono::Duration::hours((n - i) as i64),
                open,
                high,
                low,
                close: price,
                volume: rng.gen_range(100.0..500.0) * (1.0 + noise.abs() * 10.0),
            }
        })
        .collect()
}

fn candles_to_features(candles: &[Candle], lookback: usize) -> Vec<FeatureVector> {
    let mut features = Vec::new();

    for i in lookback..candles.len() {
        let window = &candles[(i - lookback)..i];
        features.push(extract_features(window));
    }

    features
}
