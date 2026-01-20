//! Backtesting example for anomaly-based trading strategies.
//!
//! Demonstrates how to run a backtest using historical data from Bybit.

use llm_anomaly_detection::{
    backtest::{Backtester, BacktesterConfig},
    data_loader::BybitLoader,
    detector::StatisticalDetector,
    signals::SignalStrategy,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("{}", "=".repeat(60));
    println!("Anomaly-Based Strategy Backtest");
    println!("{}", "=".repeat(60));

    // Load historical data from Bybit
    println!("\nLoading BTCUSDT data from Bybit...");
    let loader = BybitLoader::new();
    let candles = loader.get_klines("BTCUSDT", "4h", 1000).await?;
    println!("Loaded {} candles (4-hour timeframe)", candles.len());

    if candles.len() < 300 {
        println!("Not enough data for backtest. Need at least 300 candles.");
        return Ok(());
    }

    // Test different strategies
    let strategies = [
        ("Contrarian", SignalStrategy::Contrarian),
        ("Momentum", SignalStrategy::Momentum),
        ("Risk", SignalStrategy::Risk),
    ];

    let mut results = Vec::new();

    for (name, strategy) in strategies {
        println!("\n{}", "-".repeat(60));
        println!("Testing {} Strategy", name);
        println!("{}", "-".repeat(60));

        // Create detector
        let detector = StatisticalDetector::new(2.5);

        // Create backtester with configuration
        let config = BacktesterConfig {
            initial_capital: 10_000.0,
            position_size: 0.1,
            max_positions: 3,
            transaction_cost: 0.001,  // 0.1% (Bybit trading fee)
            slippage: 0.001,          // 0.1% slippage for crypto
            stop_loss: 0.08,          // 8% stop loss (wider for crypto)
            take_profit: 0.15,        // 15% take profit
        };

        let mut backtester = Backtester::new(detector)
            .with_config(config)
            .with_strategy(strategy);

        // Run backtest
        let result = backtester.run(&candles, 200)?;

        println!("Results:");
        println!("  Total Return: ${:.2} ({:.2}%)", result.total_return, result.total_return_pct);
        println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
        println!("  Max Drawdown: ${:.2} ({:.2}%)", result.max_drawdown, result.max_drawdown_pct);
        println!("  Win Rate: {:.1}%", result.win_rate);
        println!("  Trades: {}", result.num_trades);
        println!("  Anomalies: {}", result.anomaly_count);

        results.push((name, result));
    }

    // Print comparison summary
    println!("\n{}", "=".repeat(60));
    println!("STRATEGY COMPARISON");
    println!("{}", "=".repeat(60));
    println!(
        "{:<15} {:>12} {:>10} {:>12} {:>10} {:>8}",
        "Strategy", "Return %", "Sharpe", "MaxDD %", "Win Rate", "Trades"
    );
    println!("{}", "-".repeat(70));

    for (name, result) in &results {
        println!(
            "{:<15} {:>11.2}% {:>10.2} {:>11.2}% {:>9.1}% {:>8}",
            name,
            result.total_return_pct,
            result.sharpe_ratio,
            result.max_drawdown_pct,
            result.win_rate,
            result.num_trades
        );
    }

    // Find best strategy
    let best = results
        .iter()
        .max_by(|a, b| {
            a.1.sharpe_ratio
                .partial_cmp(&b.1.sharpe_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    if let Some((name, result)) = best {
        println!("\nBest Strategy: {} (Sharpe: {:.2})", name, result.sharpe_ratio);
    }

    println!("{}", "=".repeat(60));

    // Parameter sensitivity analysis
    println!("\n{}", "=".repeat(60));
    println!("PARAMETER SENSITIVITY (Z-Score Threshold)");
    println!("{}", "=".repeat(60));

    let z_thresholds = [1.5, 2.0, 2.5, 3.0, 3.5];
    println!(
        "{:<15} {:>12} {:>10} {:>10}",
        "Z-Threshold", "Return %", "Sharpe", "Trades"
    );
    println!("{}", "-".repeat(50));

    for z in z_thresholds {
        let detector = StatisticalDetector::new(z);

        let config = BacktesterConfig {
            initial_capital: 10_000.0,
            position_size: 0.1,
            max_positions: 3,
            transaction_cost: 0.001,
            slippage: 0.001,
            stop_loss: 0.08,
            take_profit: 0.15,
        };

        let mut backtester = Backtester::new(detector)
            .with_config(config)
            .with_strategy(SignalStrategy::Contrarian);

        let result = backtester.run(&candles, 200)?;

        println!(
            "{:<15} {:>11.2}% {:>10.2} {:>10}",
            format!("Z = {:.1}", z),
            result.total_return_pct,
            result.sharpe_ratio,
            result.num_trades
        );
    }

    println!("{}", "=".repeat(60));
    println!("\nBacktest completed!");

    Ok(())
}
