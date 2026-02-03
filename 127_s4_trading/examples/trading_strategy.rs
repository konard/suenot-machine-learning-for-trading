//! Complete trading strategy example with backtesting.
//!
//! Demonstrates:
//! - Creating an S4-based trading strategy
//! - Running a full backtest
//! - Analyzing results

use s4_trading::prelude::*;
use s4_trading::data::bybit::generate_synthetic_data;
use s4_trading::model::s4::S4Config;
use s4_trading::trading::strategy::StrategyConfig;
use s4_trading::backtest::engine::{BacktestEngine, BacktestConfig};

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== S4 Trading: Full Strategy Backtest ===\n");

    // 1. Configure the model
    println!("1. Configuring S4 Model");
    let model_config = S4Config {
        d_model: 64,
        d_state: 64,
        n_layers: 4,
        dropout: 0.1,
        bidirectional: false,
        dt_min: 0.001,
        dt_max: 0.1,
    };
    println!("   d_model: {}, d_state: {}, n_layers: {}",
        model_config.d_model, model_config.d_state, model_config.n_layers);

    // 2. Configure the strategy
    println!("\n2. Configuring Trading Strategy");
    let strategy_config = StrategyConfig {
        min_confidence: 0.6,
        max_position_size: 0.1,
        lookback: 256,
        stop_loss: 0.02,
        take_profit: 0.04,
    };
    println!("   min_confidence: {:.1}%", strategy_config.min_confidence * 100.0);
    println!("   max_position_size: {:.1}%", strategy_config.max_position_size * 100.0);
    println!("   lookback: {} periods", strategy_config.lookback);
    println!("   stop_loss: {:.1}%", strategy_config.stop_loss * 100.0);
    println!("   take_profit: {:.1}%", strategy_config.take_profit * 100.0);

    // 3. Configure backtesting
    println!("\n3. Configuring Backtest");
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        trading_fee: 0.001,  // 0.1%
        slippage: 0.0005,    // 0.05%
    };
    println!("   initial_capital: ${:.0}", backtest_config.initial_capital);
    println!("   trading_fee: {:.2}%", backtest_config.trading_fee * 100.0);
    println!("   slippage: {:.2}%", backtest_config.slippage * 100.0);

    // 4. Generate test data
    println!("\n4. Generating Test Data");
    let data = generate_synthetic_data("BTCUSDT", "60", 2000);
    println!("   Symbol: {}", data.symbol);
    println!("   Data points: {}", data.candles.len());
    println!("   Timeframe: {} hours", data.interval);

    // Calculate data statistics
    let returns = data.returns();
    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let volatility: f64 = (returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>() / returns.len() as f64)
        .sqrt();

    println!("   Mean return: {:.4}%", mean_return * 100.0);
    println!("   Volatility: {:.4}%", volatility * 100.0);

    // 5. Run backtest
    println!("\n5. Running Backtest...");
    let mut engine = BacktestEngine::new(
        backtest_config.clone(),
        model_config.clone(),
        strategy_config.clone(),
    );

    let result = engine.run(&data);

    // 6. Display results
    println!("\n6. Backtest Results");
    println!("{}", result.summary());

    // 7. Analyze equity curve
    println!("7. Equity Curve Analysis");
    let equity_curve = &result.equity_curve;

    if equity_curve.len() >= 10 {
        println!("   Starting equity: ${:.2}", equity_curve.first().unwrap());
        println!("   Final equity: ${:.2}", equity_curve.last().unwrap());

        // Find highest and lowest points
        let max_equity = equity_curve.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_equity = equity_curve.iter().cloned().fold(f64::INFINITY, f64::min);

        println!("   Highest equity: ${:.2}", max_equity);
        println!("   Lowest equity: ${:.2}", min_equity);

        // Print equity at intervals
        println!("\n   Equity at key points:");
        let n = equity_curve.len();
        let intervals = [0, n/4, n/2, 3*n/4, n-1];
        for &i in &intervals {
            if i < n {
                println!("     Period {}: ${:.2}", i, equity_curve[i]);
            }
        }
    }

    // 8. Risk metrics deep dive
    println!("\n8. Risk Metrics Analysis");
    println!("   Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("   Sortino Ratio: {:.2}", result.sortino_ratio);
    println!("   Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("   Calmar Ratio: {:.2}", result.calmar_ratio);

    // Risk assessment
    let risk_rating = if result.sharpe_ratio > 1.5 && result.max_drawdown < 0.15 {
        "Low Risk"
    } else if result.sharpe_ratio > 1.0 && result.max_drawdown < 0.25 {
        "Medium Risk"
    } else {
        "High Risk"
    };
    println!("   Risk Rating: {}", risk_rating);

    // 9. Trade statistics
    println!("\n9. Trading Statistics");
    println!("   Total Trades: {}", result.total_trades);
    println!("   Win Rate: {:.1}%", result.win_rate * 100.0);
    println!("   Profit Factor: {:.2}", result.profit_factor);

    // Estimate trades per period
    let trades_per_day = if data.candles.len() > 0 {
        result.total_trades as f64 / (data.candles.len() as f64 / 24.0)
    } else {
        0.0
    };
    println!("   Avg Trades/Day: {:.2}", trades_per_day);

    // 10. Compare with buy-and-hold
    println!("\n10. Strategy vs Buy-and-Hold");
    let first_price = data.candles.first().map(|c| c.close).unwrap_or(1.0);
    let last_price = data.candles.last().map(|c| c.close).unwrap_or(1.0);
    let buy_hold_return = (last_price - first_price) / first_price;

    println!("   Buy & Hold Return: {:.2}%", buy_hold_return * 100.0);
    println!("   Strategy Return: {:.2}%", result.total_return * 100.0);
    println!("   Outperformance: {:.2}%",
        (result.total_return - buy_hold_return) * 100.0);

    // Conclusion
    println!("\n11. Strategy Assessment");
    let assessment = if result.sharpe_ratio > 1.5 {
        "EXCELLENT - Strong risk-adjusted returns"
    } else if result.sharpe_ratio > 1.0 {
        "GOOD - Acceptable risk-adjusted returns"
    } else if result.sharpe_ratio > 0.5 {
        "FAIR - Moderate risk-adjusted returns"
    } else {
        "POOR - Consider strategy refinement"
    };
    println!("   {}", assessment);

    println!("\n=== Trading Strategy Example Complete ===");
}
