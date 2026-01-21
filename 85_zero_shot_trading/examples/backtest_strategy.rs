//! Backtesting example: Run strategy backtest on historical data
//!
//! This example demonstrates the full backtesting workflow:
//! 1. Generate or load historical data
//! 2. Run the zero-shot trading strategy
//! 3. Analyze performance metrics
//!
//! Run with: cargo run --example backtest_strategy

use zero_shot_trading::prelude::*;
use zero_shot_trading::data::bybit::Kline;
use chrono::{Utc, Duration};

fn main() -> Result<()> {
    println!("Zero-Shot Trading: Backtest Strategy Example");
    println!("============================================\n");

    // Create model
    let model_config = zero_shot_trading::model::network::ModelConfig::default();
    let model = ZeroShotModel::new(model_config);

    // Create backtest configuration
    let strategy_config = StrategyConfig {
        symbols: vec!["BTCUSDT".to_string()],
        interval: "15".to_string(),
        lookback: 200,
        max_position_size: 0.1,   // 10% of portfolio per trade
        risk_per_trade: 0.02,     // 2% risk per trade
        stop_loss_pct: 0.02,      // 2% stop loss
        take_profit_pct: 0.04,    // 4% take profit
        min_confidence: 0.5,       // 50% minimum confidence
    };

    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        trading_fee: 0.001,        // 0.1% trading fee
        slippage: 0.0005,          // 0.05% slippage
        strategy_config,
        warmup_period: 100,
    };

    println!("Backtest Configuration:");
    println!("  Initial Capital: ${:.2}", backtest_config.initial_capital);
    println!("  Trading Fee: {:.2}%", backtest_config.trading_fee * 100.0);
    println!("  Slippage: {:.2}%", backtest_config.slippage * 100.0);
    println!("  Max Position: {:.1}%", backtest_config.strategy_config.max_position_size * 100.0);
    println!("  Stop Loss: {:.1}%", backtest_config.strategy_config.stop_loss_pct * 100.0);
    println!("  Take Profit: {:.1}%", backtest_config.strategy_config.take_profit_pct * 100.0);
    println!();

    // Create backtesting engine
    let mut engine = BacktestEngine::new(model, backtest_config);

    // Generate synthetic historical data for different market scenarios
    println!("Running backtests on synthetic data...\n");

    // Scenario 1: Bull market
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Scenario 1: BULL MARKET");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let bull_klines = generate_trending_klines(500, 0.001); // Uptrend
    run_backtest(&mut engine, &bull_klines, "Bull Market");

    // Recreate engine for fresh state
    let model = ZeroShotModel::new(zero_shot_trading::model::network::ModelConfig::default());
    let mut engine = BacktestEngine::new(model, BacktestConfig::default());

    // Scenario 2: Bear market
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Scenario 2: BEAR MARKET");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let bear_klines = generate_trending_klines(500, -0.001); // Downtrend
    run_backtest(&mut engine, &bear_klines, "Bear Market");

    // Recreate engine
    let model = ZeroShotModel::new(zero_shot_trading::model::network::ModelConfig::default());
    let mut engine = BacktestEngine::new(model, BacktestConfig::default());

    // Scenario 3: Sideways market
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Scenario 3: SIDEWAYS MARKET");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let sideways_klines = generate_sideways_klines(500);
    run_backtest(&mut engine, &sideways_klines, "Sideways Market");

    // Recreate engine
    let model = ZeroShotModel::new(zero_shot_trading::model::network::ModelConfig::default());
    let mut engine = BacktestEngine::new(model, BacktestConfig::default());

    // Scenario 4: Volatile market
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Scenario 4: VOLATILE MARKET");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let volatile_klines = generate_volatile_klines(500);
    run_backtest(&mut engine, &volatile_klines, "Volatile Market");

    println!("\n✓ All backtests completed!");
    Ok(())
}

fn run_backtest(engine: &mut BacktestEngine, klines: &[Kline], scenario: &str) {
    match engine.run(klines, "SYNTHETIC") {
        Ok(results) => {
            println!("\n{}", results.summary_report());
        }
        Err(e) => {
            println!("  Backtest error: {}", e);
        }
    }
}

/// Generate trending klines (bull or bear market)
fn generate_trending_klines(n: usize, daily_trend: f64) -> Vec<Kline> {
    let mut klines = Vec::with_capacity(n);
    let mut price = 50000.0; // Starting price (like BTC)

    for i in 0..n {
        // Add trend + random noise
        let change = daily_trend + (rand::random::<f64>() - 0.5) * 0.01;
        price *= 1.0 + change;
        price = price.max(100.0); // Prevent negative prices

        let volatility = 0.01 + rand::random::<f64>() * 0.02;
        klines.push(Kline {
            timestamp: Utc::now() + Duration::minutes(i as i64 * 15),
            open: price * (1.0 - volatility * 0.5),
            high: price * (1.0 + volatility),
            low: price * (1.0 - volatility),
            close: price,
            volume: 100.0 * (1.0 + rand::random::<f64>()),
            turnover: price * 100.0,
        });
    }

    klines
}

/// Generate sideways klines
fn generate_sideways_klines(n: usize) -> Vec<Kline> {
    let mut klines = Vec::with_capacity(n);
    let center_price = 50000.0;
    let range = 1000.0; // Price oscillates within this range

    for i in 0..n {
        // Oscillate around center with some randomness
        let cycle = (i as f64 * 0.1).sin() * range;
        let noise = (rand::random::<f64>() - 0.5) * range * 0.3;
        let price = center_price + cycle + noise;

        let volatility = 0.005 + rand::random::<f64>() * 0.01;
        klines.push(Kline {
            timestamp: Utc::now() + Duration::minutes(i as i64 * 15),
            open: price * (1.0 - volatility * 0.5),
            high: price * (1.0 + volatility),
            low: price * (1.0 - volatility),
            close: price,
            volume: 80.0 * (1.0 + rand::random::<f64>() * 0.5),
            turnover: price * 80.0,
        });
    }

    klines
}

/// Generate volatile klines with regime changes
fn generate_volatile_klines(n: usize) -> Vec<Kline> {
    let mut klines = Vec::with_capacity(n);
    let mut price = 50000.0;
    let mut trend = 0.001; // Start bullish

    for i in 0..n {
        // Change regime periodically
        if i % 100 == 0 && i > 0 {
            trend = -trend; // Flip between bull and bear
        }

        // Higher volatility
        let change = trend + (rand::random::<f64>() - 0.5) * 0.03;
        price *= 1.0 + change;
        price = price.max(100.0);

        let volatility = 0.02 + rand::random::<f64>() * 0.03;
        klines.push(Kline {
            timestamp: Utc::now() + Duration::minutes(i as i64 * 15),
            open: price * (1.0 - volatility * 0.5),
            high: price * (1.0 + volatility),
            low: price * (1.0 - volatility),
            close: price,
            volume: 150.0 * (1.0 + rand::random::<f64>()),
            turnover: price * 150.0,
        });
    }

    klines
}
