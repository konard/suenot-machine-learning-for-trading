//! Backtest Demo
//!
//! Demonstrates backtesting with full audit trails.

use chrono::{Utc, Duration};
use cot_trading::{
    Backtester, BacktestConfig, SignalGenerator, PositionSizer,
    MockLoader, DataLoader,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("====================================================");
    println!("Chain-of-Thought Trading Backtest Demo");
    println!("====================================================\n");

    let symbol = "AAPL";
    let initial_capital = 100_000.0;

    println!("Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Commission: 0.1%");
    println!("  Slippage: 0.05%");

    // Load historical data
    println!("\nLoading historical data...");
    let loader = MockLoader::new(42);
    let end = Utc::now();
    let start = end - Duration::days(365);

    let bars = loader.load(symbol, start, end, "1d").await?;
    println!("  Loaded {} bars", bars.len());
    println!("  Date range: {} to {}",
             bars.first().map(|b| b.timestamp.format("%Y-%m-%d").to_string()).unwrap_or_default(),
             bars.last().map(|b| b.timestamp.format("%Y-%m-%d").to_string()).unwrap_or_default());

    // Extract prices and timestamps
    let prices: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let timestamps: Vec<_> = bars.iter().map(|b| b.timestamp).collect();

    // Configure backtest
    let config = BacktestConfig {
        initial_capital,
        commission_pct: 0.001,
        slippage_pct: 0.0005,
        max_position_pct: 0.2,
        min_bars_between_trades: 5,
    };

    // Create backtester
    let signal_gen = SignalGenerator::new_mock();
    let position_sizer = PositionSizer::new(0.2, 0.02);
    let backtester = Backtester::new(signal_gen, position_sizer, config);

    // Run backtest
    println!("\nRunning backtest...");
    let result = backtester.run(&prices, &timestamps, symbol).await?;

    // Display results
    println!("\n====================================================");
    println!("Backtest Results");
    println!("====================================================\n");

    println!("Performance Metrics:");
    println!("  Total Return:    {:>8.2}%", result.total_return * 100.0);
    println!("  Annual Return:   {:>8.2}%", result.annual_return * 100.0);
    println!("  Sharpe Ratio:    {:>8.2}", result.sharpe_ratio);
    println!("  Max Drawdown:    {:>8.2}%", result.max_drawdown * 100.0);
    println!("  Win Rate:        {:>8.1}%", result.win_rate * 100.0);
    println!("  Profit Factor:   {:>8.2}", result.profit_factor);

    println!("\nTrading Statistics:");
    println!("  Total Trades:    {:>8}", result.total_trades);
    println!("  Winning Trades:  {:>8}", result.winning_trades);
    println!("  Losing Trades:   {:>8}", result.losing_trades);

    println!("\nCapital:");
    println!("  Initial:         ${:>12.2}", initial_capital);
    println!("  Final:           ${:>12.2}", result.final_capital);
    println!("  Profit/Loss:     ${:>12.2}", result.final_capital - initial_capital);

    // Display trade log
    if !result.trades.is_empty() {
        println!("\n----------------------------------------------------");
        println!("Trade Log (first 5 trades)");
        println!("----------------------------------------------------");
        println!("{:<4} {:<12} {:<6} {:>10} {:>10} {:>12} {:>8}",
                 "#", "Date", "Type", "Entry", "Exit", "P/L", "Return");
        println!("{}", "-".repeat(70));

        for (i, trade) in result.trades.iter().take(5).enumerate() {
            let direction = match trade.direction {
                cot_trading::TradeDirection::Long => "LONG",
                cot_trading::TradeDirection::Short => "SHORT",
            };

            println!("{:<4} {:<12} {:<6} ${:>9.2} ${:>9.2} ${:>11.2} {:>7.2}%",
                     i + 1,
                     trade.entry_time.format("%Y-%m-%d"),
                     direction,
                     trade.entry_price,
                     trade.exit_price,
                     trade.pnl,
                     trade.return_pct * 100.0);
        }

        if result.trades.len() > 5 {
            println!("... and {} more trades", result.trades.len() - 5);
        }

        // Show reasoning for first trade
        if let Some(trade) = result.trades.first() {
            println!("\n----------------------------------------------------");
            println!("Sample Trade Reasoning (Trade #1)");
            println!("----------------------------------------------------");
            println!("Direction: {:?}", trade.direction);
            println!("Entry: ${:.2} -> Exit: ${:.2}", trade.entry_price, trade.exit_price);
            println!("P/L: ${:.2} ({:.2}%)", trade.pnl, trade.return_pct * 100.0);
            println!("Confidence: {:.0}%", trade.confidence * 100.0);

            println!("\nReasoning Chain:");
            for (i, reason) in trade.reasoning_chain.iter().take(5).enumerate() {
                println!("  {}. {}", i + 1, reason);
            }
        }
    }

    // Equity curve summary
    println!("\n----------------------------------------------------");
    println!("Equity Curve (monthly snapshots)");
    println!("----------------------------------------------------");

    let step = result.equity_curve.len() / 12;
    let step = step.max(1);

    for i in (0..result.equity_curve.len()).step_by(step) {
        let timestamp = &timestamps[50 + i.min(timestamps.len() - 51)];
        let equity = result.equity_curve[i];
        let change_pct = (equity - initial_capital) / initial_capital * 100.0;

        let bar_len = ((equity / initial_capital - 0.8) * 50.0).max(0.0).min(50.0) as usize;
        let bar = "█".repeat(bar_len);

        println!("  {} ${:>12.2} ({:>+6.1}%) {}",
                 timestamp.format("%Y-%m"),
                 equity,
                 change_pct,
                 bar);
    }

    println!("\n====================================================");
    println!("Backtest complete!");
    println!("====================================================");
    println!("\nNote: This uses mock data. For real trading:");
    println!("  1. Use YahooLoader or BybitLoader for real data");
    println!("  2. Configure with your API keys for real LLM analysis");
    println!("  3. Always paper trade before going live!");

    Ok(())
}
