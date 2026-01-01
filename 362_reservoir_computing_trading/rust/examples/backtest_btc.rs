//! BTC Backtesting Example
//!
//! This example demonstrates how to backtest a reservoir computing
//! trading strategy on historical Bitcoin data from Bybit.
//!
//! Run with: cargo run --example backtest_btc

use reservoir_trading::{
    BacktestConfig, Backtester, BybitClient, BybitConfig,
    EsnConfig, TradingConfig, SignalThresholds,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Reservoir Computing - BTC Backtesting Example");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Fetch historical data from Bybit
    println!("1. Fetching historical BTC data from Bybit...");
    let config = BybitConfig::mainnet();
    let client = BybitClient::new(config);

    // Fetch 1000 1-hour candles
    let symbol = "BTCUSDT";
    let interval = "60"; // 1 hour
    let klines = client.get_klines(symbol, interval, 1000).await?;

    println!("   Symbol: {}", symbol);
    println!("   Interval: {} minutes", interval);
    println!("   Fetched {} candles", klines.len());

    if klines.is_empty() {
        println!("   ERROR: No data received. Please check your connection.");
        return Ok(());
    }

    // Show data range
    let first = klines.first().unwrap();
    let last = klines.last().unwrap();
    let start_time = chrono::DateTime::from_timestamp_millis(first.start_time as i64)
        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_default();
    let end_time = chrono::DateTime::from_timestamp_millis(last.start_time as i64)
        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_default();

    println!("   Date range: {} to {}", start_time, end_time);
    println!("   Price range: ${:.2} - ${:.2}", first.close, last.close);

    // Calculate basic statistics
    let returns: Vec<f64> = klines
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let volatility = {
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
            / (returns.len() - 1) as f64;
        variance.sqrt() * (24.0 * 365.0_f64).sqrt() // Annualized (hourly data)
    };

    println!("\n   Market Statistics:");
    println!("   - Annualized Volatility: {:.1}%", volatility * 100.0);
    println!("   - Mean Hourly Return: {:.4}%", mean_return * 100.0);

    // Configure backtesting
    println!("\n2. Configuring backtest...");

    let esn_config = EsnConfig {
        reservoir_size: 500,
        spectral_radius: 0.95,
        input_scaling: 0.5,
        leaking_rate: 0.3,
        sparsity: 0.1,
        regularization: 1e-6,
        seed: 42,
    };

    let trading_config = TradingConfig {
        max_position: 1.0,
        position_scale: 0.5,
        transaction_cost: 0.001, // 0.1% (Bybit taker fee)
        stop_loss: Some(0.03),   // 3% stop loss
        take_profit: Some(0.06), // 6% take profit
        max_drawdown: 0.15,      // 15% max drawdown
        thresholds: SignalThresholds {
            strong_buy: 0.4,
            buy: 0.15,
            sell: -0.15,
            strong_sell: -0.4,
        },
    };

    let backtest_config = BacktestConfig {
        train_ratio: 0.6,          // 60% training, 40% testing
        washout: 100,              // 100 samples washout
        initial_capital: 10000.0,  // $10,000 starting capital
        rolling_retrain: false,    // No rolling retraining
        retrain_interval: 200,     // Retrain every 200 bars (if enabled)
        esn_config,
        trading_config,
    };

    println!("   ESN Configuration:");
    println!("   - Reservoir Size: {}", backtest_config.esn_config.reservoir_size);
    println!("   - Spectral Radius: {}", backtest_config.esn_config.spectral_radius);
    println!("   - Leaking Rate: {}", backtest_config.esn_config.leaking_rate);

    println!("\n   Trading Configuration:");
    println!("   - Transaction Cost: {:.2}%", backtest_config.trading_config.transaction_cost * 100.0);
    println!("   - Stop Loss: {:.1}%", backtest_config.trading_config.stop_loss.unwrap() * 100.0);
    println!("   - Take Profit: {:.1}%", backtest_config.trading_config.take_profit.unwrap() * 100.0);
    println!("   - Max Drawdown: {:.1}%", backtest_config.trading_config.max_drawdown * 100.0);

    // Run backtest
    println!("\n3. Running backtest...");
    let start = std::time::Instant::now();

    let backtester = Backtester::new(backtest_config);
    let result = backtester.run(&klines);

    let elapsed = start.elapsed();
    println!("   Backtest completed in {:?}", elapsed);

    // Display results
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("{}", result.metrics.summary());
    println!("═══════════════════════════════════════════════════════════════════");

    // Additional analysis
    println!("\n4. Additional Analysis:");

    // Prediction accuracy
    if !result.predictions.is_empty() && !result.actual_returns.is_empty() {
        let correct_direction: usize = result
            .predictions
            .iter()
            .zip(result.actual_returns.iter())
            .filter(|(pred, actual)| pred.signum() == actual.signum())
            .count();

        let direction_accuracy = correct_direction as f64 / result.predictions.len() as f64;
        println!("   Direction Accuracy: {:.1}%", direction_accuracy * 100.0);

        // Correlation
        let pred_mean = result.predictions.iter().sum::<f64>() / result.predictions.len() as f64;
        let actual_mean = result.actual_returns.iter().sum::<f64>() / result.actual_returns.len() as f64;

        let cov: f64 = result
            .predictions
            .iter()
            .zip(result.actual_returns.iter())
            .map(|(p, a)| (p - pred_mean) * (a - actual_mean))
            .sum::<f64>()
            / result.predictions.len() as f64;

        let pred_std = (result.predictions.iter().map(|p| (p - pred_mean).powi(2)).sum::<f64>()
            / result.predictions.len() as f64)
            .sqrt();
        let actual_std = (result.actual_returns.iter().map(|a| (a - actual_mean).powi(2)).sum::<f64>()
            / result.actual_returns.len() as f64)
            .sqrt();

        let correlation = if pred_std > 0.0 && actual_std > 0.0 {
            cov / (pred_std * actual_std)
        } else {
            0.0
        };

        println!("   Prediction-Return Correlation: {:.3}", correlation);
    }

    // Position analysis
    if !result.positions.is_empty() {
        let long_pct = result.positions.iter().filter(|&&p| p > 0.01).count() as f64
            / result.positions.len() as f64;
        let short_pct = result.positions.iter().filter(|&&p| p < -0.01).count() as f64
            / result.positions.len() as f64;
        let flat_pct = 1.0 - long_pct - short_pct;

        println!("   Time in Market:");
        println!("   - Long:  {:.1}%", long_pct * 100.0);
        println!("   - Short: {:.1}%", short_pct * 100.0);
        println!("   - Flat:  {:.1}%", flat_pct * 100.0);
    }

    // Equity curve statistics
    if result.equity_curve.len() > 1 {
        let equity_returns: Vec<f64> = result
            .equity_curve
            .windows(2)
            .map(|w| (w[1] / w[0]) - 1.0)
            .collect();

        let positive_periods = equity_returns.iter().filter(|&&r| r > 0.0).count();
        let hit_rate = positive_periods as f64 / equity_returns.len() as f64;

        println!("   Period Hit Rate: {:.1}%", hit_rate * 100.0);

        // Best and worst periods
        let best = equity_returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst = equity_returns.iter().cloned().fold(f64::INFINITY, f64::min);

        println!("   Best Period:  {:.2}%", best * 100.0);
        println!("   Worst Period: {:.2}%", worst * 100.0);
    }

    // Compare with buy-and-hold
    let buy_hold_return = (last.close / first.close) - 1.0;
    println!("\n   Benchmark Comparison:");
    println!("   - Strategy Return:  {:.2}%", result.metrics.total_return * 100.0);
    println!("   - Buy & Hold Return: {:.2}%", buy_hold_return * 100.0);
    println!(
        "   - Excess Return:     {:.2}%",
        (result.metrics.total_return - buy_hold_return) * 100.0
    );

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  Backtest completed successfully!");
    println!("═══════════════════════════════════════════════════════════════════");

    Ok(())
}
