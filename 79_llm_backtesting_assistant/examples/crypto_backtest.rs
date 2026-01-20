//! Crypto Backtest Example
//!
//! This example demonstrates how to fetch cryptocurrency data from Bybit,
//! run a backtest with a moving average crossover strategy, and analyze
//! the results using an LLM.
//!
//! Run with: `cargo run --example crypto_backtest`

use chrono::{Duration, Utc};
use llm_backtesting_assistant::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== LLM Backtesting Assistant - Crypto Backtest Example ===\n");

    // Configuration
    let symbol = "BTCUSDT";
    let interval = "1d";
    let lookback_days = 365;

    println!("Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Interval: {}", interval);
    println!("  Lookback: {} days", lookback_days);
    println!();

    // Fetch data from Bybit
    println!("1. Fetching historical data from Bybit...");

    let fetcher = BybitDataFetcher::new();
    let end = Utc::now();
    let start = end - Duration::days(lookback_days);

    let candles = match fetcher.fetch_candles(symbol, interval, start, end).await {
        Ok(data) => {
            println!("   Fetched {} candles from Bybit", data.len());
            data
        }
        Err(e) => {
            println!("   Warning: Could not fetch from Bybit ({})", e);
            println!("   Using generated sample data instead...");
            generate_sample_candles(lookback_days as usize, 40000.0)
        }
    };

    if candles.is_empty() {
        println!("   No data available, using sample data...");
        let candles = generate_sample_candles(lookback_days as usize, 40000.0);
        run_backtest_and_analysis(&candles, symbol).await?;
    } else {
        run_backtest_and_analysis(&candles, symbol).await?;
    }

    Ok(())
}

async fn run_backtest_and_analysis(
    candles: &[Candle],
    symbol: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Display data summary
    println!("\n2. Data Summary:");
    if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
        println!("   Start: {} @ ${:.2}", first.timestamp.format("%Y-%m-%d"), first.close);
        println!("   End: {} @ ${:.2}", last.timestamp.format("%Y-%m-%d"), last.close);
        let price_change = (last.close - first.close) / first.close * 100.0;
        println!("   Price Change: {:.2}%", price_change);
    }
    println!();

    // Create and run backtest
    println!("3. Running backtest with MA Crossover strategy...");
    println!("   Fast MA: 10 periods");
    println!("   Slow MA: 30 periods");

    let mut strategy = MovingAverageCrossover::new(10, 30);
    let backtester = Backtester::new(10000.0)
        .with_position_size(0.2)  // 20% position size
        .with_commission(0.001);   // 0.1% commission (typical for crypto)

    let results = backtester.run(&mut strategy, candles, symbol, MarketType::Crypto);

    // Display backtest results
    println!("\n4. Backtest Results:");
    println!("   Initial Capital: ${:.2}", results.initial_capital);
    println!("   Final Capital: ${:.2}", results.final_capital);
    println!("   Total Return: {:.2}%", results.metrics.total_return * 100.0);
    println!("   Sharpe Ratio: {:.2}", results.metrics.sharpe_ratio);
    println!("   Sortino Ratio: {:.2}", results.metrics.sortino_ratio);
    println!("   Calmar Ratio: {:.2}", results.metrics.calmar_ratio);
    println!("   Max Drawdown: {:.2}%", results.metrics.max_drawdown * 100.0);
    println!("   Win Rate: {:.2}%", results.metrics.win_rate * 100.0);
    println!("   Profit Factor: {:.2}", results.metrics.profit_factor);
    println!("   Total Trades: {}", results.metrics.total_trades);
    println!();

    // Display trade summary
    if !results.trades.is_empty() {
        println!("5. Recent Trades:");
        for (i, trade) in results.trades.iter().rev().take(5).enumerate() {
            let pnl_sign = if trade.pnl >= 0.0 { "+" } else { "" };
            println!("   {}. {} {:?}: ${:.2} -> ${:.2} ({}${:.2})",
                i + 1,
                trade.entry_time.format("%Y-%m-%d"),
                trade.side,
                trade.entry_price,
                trade.exit_price,
                pnl_sign,
                trade.pnl
            );
        }
        println!();
    }

    // Create assistant and run analysis
    println!("6. Running LLM analysis...");

    // Check for API key
    let assistant: BacktestingAssistant<MockLlmClient>;
    let use_mock = std::env::var("OPENAI_API_KEY").is_err()
        && std::env::var("ANTHROPIC_API_KEY").is_err();

    if use_mock {
        println!("   (No API key found, using mock LLM provider)");
        assistant = BacktestingAssistant::with_provider(MockLlmClient::new());
    } else {
        println!("   (Using mock provider for this example)");
        assistant = BacktestingAssistant::with_provider(MockLlmClient::new());
    }

    let analysis = assistant.analyze(&results).await?;

    println!("\n7. Analysis Result:");
    println!("{}", "-".repeat(60));
    println!("{}", analysis.analysis);
    println!("{}", "-".repeat(60));

    // Generate markdown report
    println!("\n8. Generating report...");
    let report = ReportBuilder::new(results)
        .with_analysis(analysis)
        .with_title(format!("{} MA Crossover Backtest Report", symbol))
        .build();

    let md_report = report.generate(ReportFormat::Markdown)?;
    println!("   Report generated ({} characters)", md_report.len());

    // Optionally save report
    // std::fs::write("crypto_backtest_report.md", &md_report)?;
    // println!("   Saved to crypto_backtest_report.md");

    println!("\n=== Crypto Backtest Example Complete ===");

    Ok(())
}
