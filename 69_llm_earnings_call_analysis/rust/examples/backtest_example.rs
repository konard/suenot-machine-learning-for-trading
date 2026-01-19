//! Backtesting example
//!
//! This example demonstrates how to backtest an earnings-based trading strategy.

use earnings_call_analyzer::{
    backtest::{Backtester, BacktestConfig, EarningsEvent},
};

fn main() {
    println!("=== Earnings Call Trading Backtest ===\n");

    // Create sample earnings events
    let events = generate_sample_events();

    println!("Number of earnings events: {}\n", events.len());

    // Run backtest with default configuration
    let backtester = Backtester::new();
    let results = backtester.run(&events);

    println!("{}", results);

    // Run backtest with custom configuration
    println!("\n--- Custom Configuration ---\n");

    let custom_config = BacktestConfig {
        initial_capital: 50_000.0,
        position_size: 0.2,
        holding_period_days: 3,
        transaction_cost: 0.0005,
        risk_free_rate: 0.04,
    };

    let custom_backtester = Backtester::with_config(custom_config);
    let custom_results = custom_backtester.run(&events);

    println!("{}", custom_results);

    // Show individual trades
    println!("\n--- Individual Trades ---\n");
    for (i, trade) in results.trades.iter().take(5).enumerate() {
        println!("Trade {}: {} | Entry: ${:.2} | Exit: ${:.2} | P&L: ${:.2} ({:.2}%)",
            i + 1,
            trade.signal,
            trade.entry_price,
            trade.exit_price,
            trade.pnl,
            trade.pnl_pct * 100.0
        );
    }
    if results.trades.len() > 5 {
        println!("... and {} more trades", results.trades.len() - 5);
    }

    println!("\n=== Backtest Complete ===");
}

fn generate_sample_events() -> Vec<EarningsEvent> {
    // Sample earnings transcripts and price movements
    vec![
        // Q1 - Strong results
        EarningsEvent {
            timestamp: 1704067200000, // Jan 1, 2024
            transcript: r#"
CEO: Exceptional quarter with 25% revenue growth.
We exceeded all expectations and are raising guidance.
Strong confidence in the business trajectory.
            "#.to_string(),
            price_before: 100.0,
            price_after: 112.0,
        },
        // Q2 - Mixed results
        EarningsEvent {
            timestamp: 1711929600000, // Apr 1, 2024
            transcript: r#"
CFO: Results were in line with our revised expectations.
We faced some headwinds but managed through them.
Maintaining guidance for the year.
            "#.to_string(),
            price_before: 108.0,
            price_after: 105.0,
        },
        // Q3 - Weak results
        EarningsEvent {
            timestamp: 1719792000000, // Jul 1, 2024
            transcript: r#"
CEO: Challenging quarter with declining revenue.
Market conditions were more difficult than anticipated.
We are lowering guidance due to uncertainty.
            "#.to_string(),
            price_before: 102.0,
            price_after: 88.0,
        },
        // Q4 - Recovery
        EarningsEvent {
            timestamp: 1727740800000, // Oct 1, 2024
            transcript: r#"
CFO: We delivered solid results this quarter.
Revenue grew modestly as conditions improved.
We are cautiously optimistic about the outlook.
            "#.to_string(),
            price_before: 92.0,
            price_after: 98.0,
        },
        // Next Q1 - Strong rebound
        EarningsEvent {
            timestamp: 1735689600000, // Jan 1, 2025
            transcript: r#"
CEO: Outstanding quarter with record results.
We achieved significant growth across all segments.
Very confident in our strong position.
            "#.to_string(),
            price_before: 100.0,
            price_after: 118.0,
        },
        // Additional bearish event
        EarningsEvent {
            timestamp: 1743552000000, // Apr 1, 2025
            transcript: r#"
CFO: Revenue declined significantly this quarter.
We face continued headwinds and challenges.
Outlook remains uncertain and volatile.
            "#.to_string(),
            price_before: 115.0,
            price_after: 98.0,
        },
    ]
}
