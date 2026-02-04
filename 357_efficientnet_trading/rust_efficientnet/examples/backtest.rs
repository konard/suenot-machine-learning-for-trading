//! Example: Run a backtest with simulated signals
//!
//! This example demonstrates how to run a backtest using
//! the backtesting engine with pre-generated signals.

use chrono::{TimeZone, Utc};

use efficientnet_trading::prelude::*;
use efficientnet_trading::api::types::{Kline, KlineData};
use efficientnet_trading::backtest::engine::{BacktestConfig, BacktestEngine};
use efficientnet_trading::model::config::ModelConfig;
use efficientnet_trading::model::inference::{ModelInference, RuleBasedModel};

fn main() -> anyhow::Result<()> {
    println!("EfficientNet Trading - Backtest Example");
    println!("=======================================\n");

    // Generate simulated market data
    println!("1. Generating simulated market data...");
    let data = generate_market_data(500);
    println!("   Generated {} candles", data.len());
    println!("   Period: {} to {}",
        data.klines.first().map(|k| k.timestamp.to_string()).unwrap_or_default(),
        data.klines.last().map(|k| k.timestamp.to_string()).unwrap_or_default()
    );

    // Create image generator and model
    let generator = ImageStackGenerator::new(64);
    let model_config = ModelConfig::b2();
    let model = RuleBasedModel::new(model_config);
    let signal_gen = SignalGenerator::with_threshold(0.5);

    // Generate signals for all data points
    println!("\n2. Generating trading signals...");
    let lookback = 60;
    let mut signals = vec![Signal::new(SignalType::Hold, 0.0); lookback];

    let closes = data.close_prices();
    for i in lookback..closes.len() {
        let window = &closes[i - lookback..i];
        let image = generator.create_multi_transform_stack(window);
        let pred = model.predict(&image)?;
        let signal = signal_gen.generate(
            pred.prob_down,
            pred.prob_neutral,
            pred.prob_up,
            pred.magnitude,
        );
        signals.push(signal);
    }

    // Count signal types
    let buy_count = signals.iter().filter(|s| s.signal_type == SignalType::Buy).count();
    let sell_count = signals.iter().filter(|s| s.signal_type == SignalType::Sell).count();
    let hold_count = signals.iter().filter(|s| s.signal_type == SignalType::Hold).count();

    println!("   Total signals: {}", signals.len());
    println!("   BUY:  {} ({:.1}%)", buy_count, buy_count as f64 / signals.len() as f64 * 100.0);
    println!("   SELL: {} ({:.1}%)", sell_count, sell_count as f64 / signals.len() as f64 * 100.0);
    println!("   HOLD: {} ({:.1}%)", hold_count, hold_count as f64 / signals.len() as f64 * 100.0);

    // Run backtest with different configurations
    println!("\n3. Running backtests with different configurations...\n");

    // Configuration 1: Default (long only)
    println!("--- Configuration 1: Long Only ---");
    let config1 = BacktestConfig {
        initial_capital: 100000.0,
        position_size: 0.5,
        transaction_cost: 0.001,
        slippage: 0.0005,
        allow_short: false,
        stop_loss: None,
        take_profit: None,
    };
    let mut engine1 = BacktestEngine::new(config1);
    let report1 = engine1.run(&data, &signals);
    report1.print_summary();

    // Configuration 2: With stop-loss and take-profit
    println!("--- Configuration 2: With Stop-Loss/Take-Profit ---");
    let config2 = BacktestConfig {
        initial_capital: 100000.0,
        position_size: 0.5,
        transaction_cost: 0.001,
        slippage: 0.0005,
        allow_short: false,
        stop_loss: Some(0.03),     // 3% stop-loss
        take_profit: Some(0.05),   // 5% take-profit
    };
    let mut engine2 = BacktestEngine::new(config2);
    let report2 = engine2.run(&data, &signals);
    report2.print_summary();

    // Configuration 3: Allow shorting
    println!("--- Configuration 3: Long/Short ---");
    let config3 = BacktestConfig {
        initial_capital: 100000.0,
        position_size: 0.5,
        transaction_cost: 0.001,
        slippage: 0.0005,
        allow_short: true,
        stop_loss: Some(0.03),
        take_profit: Some(0.05),
    };
    let mut engine3 = BacktestEngine::new(config3);
    let report3 = engine3.run(&data, &signals);
    report3.print_summary();

    // Compare results
    println!("4. Results Comparison");
    println!("=====================\n");
    println!("{:<25} {:>12} {:>12} {:>12}",
        "Metric", "Long Only", "With SL/TP", "Long/Short");
    println!("{}", "-".repeat(60));
    println!("{:<25} {:>11.2}% {:>11.2}% {:>11.2}%",
        "Total Return", report1.total_return_pct, report2.total_return_pct, report3.total_return_pct);
    println!("{:<25} {:>12.2} {:>12.2} {:>12.2}",
        "Sharpe Ratio", report1.sharpe_ratio, report2.sharpe_ratio, report3.sharpe_ratio);
    println!("{:<25} {:>11.2}% {:>11.2}% {:>11.2}%",
        "Max Drawdown", report1.max_drawdown_pct, report2.max_drawdown_pct, report3.max_drawdown_pct);
    println!("{:<25} {:>12} {:>12} {:>12}",
        "Total Trades", report1.total_trades, report2.total_trades, report3.total_trades);
    println!("{:<25} {:>11.2}% {:>11.2}% {:>11.2}%",
        "Win Rate", report1.win_rate_pct, report2.win_rate_pct, report3.win_rate_pct);
    println!("{:<25} {:>12.2} {:>12.2} {:>12.2}",
        "Profit Factor", report1.profit_factor, report2.profit_factor, report3.profit_factor);

    println!("\nBacktest example complete!");

    Ok(())
}

/// Generate simulated market data
fn generate_market_data(n_candles: usize) -> KlineData {
    let mut klines = Vec::with_capacity(n_candles);
    let mut price = 45000.0;

    for i in 0..n_candles {
        // Timestamp: hourly candles starting from a fixed date
        let timestamp = Utc.timestamp_opt(1700000000 + (i as i64) * 3600, 0).unwrap();

        // Price movement with trend and mean reversion
        let trend = 10.0 * ((i as f64 * 0.02).sin());
        let noise = 100.0 * ((i as f64 * 0.5).cos() + (i as f64 * 0.3).sin() * 0.5);
        let mean_reversion = (45000.0 - price) * 0.001;

        price = (price + trend + noise + mean_reversion).max(30000.0).min(60000.0);

        // Generate OHLCV
        let volatility = 0.005;
        let open = price * (1.0 + volatility * (i as f64 * 0.7).sin());
        let high = price * (1.0 + volatility.abs());
        let low = price * (1.0 - volatility.abs());
        let close = price;
        let volume = 1000.0 + 500.0 * (i as f64 * 0.1).sin().abs();

        klines.push(Kline::new(
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            volume * close / 1000.0,
        ));
    }

    KlineData::new("BTCUSDT".to_string(), "60".to_string(), klines)
}
