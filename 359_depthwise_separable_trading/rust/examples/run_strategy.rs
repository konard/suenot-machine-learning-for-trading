//! Example: Run trading strategy with backtesting
//!
//! This example demonstrates how to create a DSC-based trading strategy,
//! fetch data from Bybit, and run a backtest.
//!
//! Run with: cargo run --example run_strategy

use dsc_trading::{
    convolution::DepthwiseSeparableConv1d,
    data::{BybitClient, CandleSeries},
    strategy::{Backtest, BacktestConfig, Signal, TradingStrategy},
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    println!("===========================================");
    println!("  DSC Trading Strategy Backtest");
    println!("===========================================\n");

    // Fetch data from Bybit
    println!("Fetching BTCUSDT historical data...");
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", "60", 1000).await?;
    println!("Received {} candles\n", candles.len());

    // Create DSC model
    // Input features: OHLCV (5) + Technical indicators (17) = 22
    println!("Creating DSC model...");
    let model = DepthwiseSeparableConv1d::new(22, 64, 3)?;
    println!("  Input channels: 22");
    println!("  Hidden channels: 64");
    println!("  Kernel size: 3");
    println!("  Parameters: {}\n", model.num_parameters());

    // Create trading strategy
    let strategy = TradingStrategy::new(model)
        .with_window_size(100)
        .with_confidence_threshold(0.5);

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: 0.001,      // 0.1% per trade
        position_size: 0.9,     // Use 90% of capital
        allow_short: true,
        stop_loss: Some(0.02),  // 2% stop loss
        take_profit: Some(0.05), // 5% take profit
        max_hold_time: Some(24), // Max 24 hours
        risk_free_rate: 0.02,
        trading_days: 365,      // Crypto trades 24/7
    };

    println!("Backtest Configuration:");
    println!("  Initial Capital: ${:.0}", config.initial_capital);
    println!("  Commission: {:.2}%", config.commission * 100.0);
    println!("  Position Size: {:.0}%", config.position_size * 100.0);
    println!("  Stop Loss: {:.0}%", config.stop_loss.unwrap_or(0.0) * 100.0);
    println!("  Take Profit: {:.0}%", config.take_profit.unwrap_or(0.0) * 100.0);
    println!();

    // Run backtest
    println!("Running backtest...\n");
    let backtest = Backtest::new(config);
    let result = backtest.run(&strategy, &candles)?;

    // Print results
    result.print_summary();

    // Calculate buy-and-hold benchmark
    let first_price = candles.first().map(|c| c.close).unwrap_or(1.0);
    let last_price = candles.last().map(|c| c.close).unwrap_or(1.0);
    let buy_hold_return = (last_price - first_price) / first_price;

    println!("\nBenchmark Comparison:");
    println!("  Buy & Hold Return: {:.2}%", buy_hold_return * 100.0);
    println!(
        "  Strategy vs B&H:   {:+.2}%",
        (result.total_return - buy_hold_return) * 100.0
    );

    if result.beats_benchmark(buy_hold_return) {
        println!("  ✓ Strategy outperforms buy-and-hold!");
    } else {
        println!("  ✗ Strategy underperforms buy-and-hold");
    }

    // Risk-adjusted performance
    println!("\nRisk Analysis:");
    println!(
        "  Return/Drawdown Ratio: {:.2}",
        result.risk_adjusted_score()
    );

    // Trade distribution
    if !result.trades.is_empty() {
        let long_trades: Vec<_> = result.trades.iter().filter(|t| t.is_long).collect();
        let short_trades: Vec<_> = result.trades.iter().filter(|t| !t.is_long).collect();

        println!("\nTrade Distribution:");
        println!("  Long trades: {}", long_trades.len());
        println!("  Short trades: {}", short_trades.len());

        let profitable_long = long_trades.iter().filter(|t| t.is_profitable()).count();
        let profitable_short = short_trades.iter().filter(|t| t.is_profitable()).count();

        if !long_trades.is_empty() {
            println!(
                "  Long win rate: {:.1}%",
                profitable_long as f64 / long_trades.len() as f64 * 100.0
            );
        }
        if !short_trades.is_empty() {
            println!(
                "  Short win rate: {:.1}%",
                profitable_short as f64 / short_trades.len() as f64 * 100.0
            );
        }
    }

    println!("\n===========================================");
    println!("  Backtest complete!");
    println!("===========================================");

    Ok(())
}
