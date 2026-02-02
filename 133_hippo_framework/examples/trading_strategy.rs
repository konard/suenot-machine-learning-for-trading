//! Trading strategy example using HiPPO framework.
//!
//! Runs a HiPPO-based trend-following strategy on synthetic
//! BTC/USDT data and computes performance metrics.

use hippo_trading::backtest::engine::BacktestEngine;
use hippo_trading::data::bybit::generate_synthetic_data;
use hippo_trading::trading::strategy::{StrategyConfig, TradingStrategy};

fn main() {
    println!("=== HiPPO Framework - Trading Strategy Example ===\n");

    // Generate synthetic BTC/USDT hourly data
    let data = generate_synthetic_data("BTCUSDT", 2000, 50000.0, 0.02, 42);
    let prices = data.close_prices();

    println!(
        "Data: {} candles of {} ({})",
        prices.len(),
        data.symbol,
        data.interval
    );
    println!(
        "Price range: {:.2} - {:.2}\n",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // --- Strategy 1: HiPPO-LegS N=32 ---
    println!("Strategy 1: HiPPO-LegS N=32 (with vol filter)");
    let config1 = StrategyConfig {
        n_hippo: 32,
        dt: 0.1,
        trend_threshold: 0.0,
        vol_filter: true,
        max_position: 1.0,
    };
    let strategy1 = TradingStrategy::new(config1);
    let engine = BacktestEngine::new(8760.0); // hourly
    let metrics1 = engine.run_with_strategy(&prices, &strategy1);
    println!("{}", metrics1);

    // --- Strategy 2: HiPPO-LegS N=64 ---
    println!("Strategy 2: HiPPO-LegS N=64 (with vol filter)");
    let config2 = StrategyConfig {
        n_hippo: 64,
        dt: 0.1,
        trend_threshold: 0.0,
        vol_filter: true,
        max_position: 1.0,
    };
    let strategy2 = TradingStrategy::new(config2);
    let metrics2 = engine.run_with_strategy(&prices, &strategy2);
    println!("{}", metrics2);

    // --- Benchmark: Buy-and-Hold ---
    println!("Benchmark: Buy-and-Hold");
    let bh_positions = vec![1.0; prices.len()];
    let bh_metrics = engine.run_with_positions(&prices, &bh_positions);
    println!("{}", bh_metrics);

    println!("=== Done ===");
}
