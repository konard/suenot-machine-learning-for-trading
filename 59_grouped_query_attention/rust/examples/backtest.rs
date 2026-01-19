//! Backtest Example
//!
//! This example demonstrates how to run a backtest with the GQA trading model.

use gqa_trading::{
    data::generate_synthetic_data,
    model::GQATrader,
    strategy::{backtest_strategy, compare_strategies, BacktestConfig},
};

fn main() {
    println!("GQA Trading Backtest Example");
    println!("============================\n");

    // Generate synthetic data for demo
    // In production, use load_bybit_data or load_yahoo_data
    println!("Generating synthetic market data...");
    let data = generate_synthetic_data(1000, 50000.0, 0.02);
    println!("Generated {} candles\n", data.len());

    // Create the GQA model
    println!("Creating GQA Trading Model...");
    let model = GQATrader::new(
        5,   // OHLCV features
        64,  // model dimension
        8,   // query heads
        2,   // KV heads (GQA)
        4,   // layers
    );
    println!("Model created with ~{} parameters\n", model.param_count());

    // Run a single backtest
    println!("Running backtest with default configuration...");
    let config = BacktestConfig {
        seq_len: 60,
        initial_capital: 10000.0,
        confidence_threshold: 0.3,
        transaction_cost: 0.001,
        stop_loss: Some(0.02),
        take_profit: Some(0.05),
        ..Default::default()
    };

    let result = backtest_strategy(&model, &data.data, config);
    result.print_summary();

    // Show some trade details
    println!("\nFirst 5 trades:");
    for (i, trade) in result.trades.iter().take(5).enumerate() {
        if let (Some(exit_price), Some(pnl_pct)) = (trade.exit_price, trade.pnl_percent) {
            println!(
                "  {}. {} @ ${:.2} -> ${:.2} = {:+.2}%",
                i + 1,
                trade.direction,
                trade.entry_price,
                exit_price,
                pnl_pct * 100.0
            );
        }
    }

    // Compare different strategies
    println!("\n\nComparing multiple strategies...");
    let strategies = vec![
        (
            "Very Conservative",
            BacktestConfig {
                confidence_threshold: 0.6,
                stop_loss: Some(0.01),
                take_profit: Some(0.02),
                ..Default::default()
            },
        ),
        (
            "Conservative",
            BacktestConfig {
                confidence_threshold: 0.5,
                stop_loss: Some(0.015),
                take_profit: Some(0.03),
                ..Default::default()
            },
        ),
        (
            "Balanced",
            BacktestConfig {
                confidence_threshold: 0.3,
                stop_loss: Some(0.02),
                take_profit: Some(0.04),
                ..Default::default()
            },
        ),
        (
            "Aggressive",
            BacktestConfig {
                confidence_threshold: 0.2,
                stop_loss: Some(0.025),
                take_profit: Some(0.05),
                ..Default::default()
            },
        ),
        (
            "Very Aggressive",
            BacktestConfig {
                confidence_threshold: 0.1,
                stop_loss: Some(0.03),
                take_profit: Some(0.06),
                ..Default::default()
            },
        ),
    ];

    let _results = compare_strategies(&model, &data.data, &strategies);

    println!("\nBacktest example completed!");
}
