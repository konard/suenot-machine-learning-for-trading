//! Stock Backtest Example
//!
//! This example demonstrates how to fetch stock market data,
//! run a backtest with a moving average crossover strategy, and analyze
//! the results using an LLM.
//!
//! Run with: `cargo run --example stock_backtest`

use chrono::{Duration, Utc};
use llm_backtesting_assistant::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== LLM Backtesting Assistant - Stock Backtest Example ===\n");

    // Configuration - popular stock symbols
    let symbols = vec!["AAPL", "MSFT", "GOOGL"];
    let interval = "1d";
    let lookback_days = 365;

    println!("Configuration:");
    println!("  Symbols: {:?}", symbols);
    println!("  Interval: {}", interval);
    println!("  Lookback: {} days", lookback_days);
    println!();

    // Process each symbol
    for symbol in symbols {
        println!("{}", "=".repeat(60));
        println!("Processing: {}", symbol);
        println!("{}", "=".repeat(60));

        // Fetch data
        println!("\n1. Fetching historical data...");

        let fetcher = StockDataFetcher::new();
        let end = Utc::now();
        let start = end - Duration::days(lookback_days);

        let candles = match fetcher.fetch_candles(symbol, interval, start, end).await {
            Ok(data) => {
                println!("   Fetched {} candles for {}", data.len(), symbol);
                data
            }
            Err(e) => {
                println!("   Warning: Could not fetch data for {} ({})", symbol, e);
                println!("   Using generated sample data instead...");
                // Generate sample data with typical stock price
                generate_sample_candles(lookback_days as usize, 150.0)
            }
        };

        if candles.len() < 50 {
            println!("   Insufficient data for backtest (need at least 50 candles)");
            println!("   Using sample data...");
            let sample_candles = generate_sample_candles(lookback_days as usize, 150.0);
            run_stock_analysis(&sample_candles, symbol).await?;
        } else {
            run_stock_analysis(&candles, symbol).await?;
        }

        println!();
    }

    println!("=== Stock Backtest Example Complete ===");

    Ok(())
}

async fn run_stock_analysis(
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

        // Calculate volatility
        let returns: Vec<f64> = candles.windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let daily_vol = variance.sqrt();
        let annual_vol = daily_vol * (252.0_f64).sqrt();
        println!("   Annualized Volatility: {:.2}%", annual_vol * 100.0);
    }

    // Run backtest
    println!("\n3. Running backtest with MA Crossover strategy...");
    println!("   Fast MA: 20 periods (approximately 1 month)");
    println!("   Slow MA: 50 periods (approximately 2.5 months)");

    let mut strategy = MovingAverageCrossover::new(20, 50);
    let backtester = Backtester::new(100000.0)  // $100k initial capital
        .with_position_size(0.1)   // 10% position size (conservative)
        .with_commission(0.0005);  // 0.05% commission (typical for stocks)

    let results = backtester.run(&mut strategy, candles, symbol, MarketType::Stock);

    // Display results
    println!("\n4. Backtest Results:");
    println!("   Initial Capital: ${:.2}", results.initial_capital);
    println!("   Final Capital: ${:.2}", results.final_capital);
    println!("   Total Return: {:.2}%", results.metrics.total_return * 100.0);
    println!("   Annualized Return: {:.2}%", results.metrics.annualized_return * 100.0);
    println!("   Sharpe Ratio: {:.2}", results.metrics.sharpe_ratio);
    println!("   Max Drawdown: {:.2}%", results.metrics.max_drawdown * 100.0);
    println!("   Win Rate: {:.2}%", results.metrics.win_rate * 100.0);
    println!("   Profit Factor: {:.2}", results.metrics.profit_factor);
    println!("   Total Trades: {}", results.metrics.total_trades);

    // Compare with buy-and-hold
    if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
        let buy_hold_return = (last.close - first.close) / first.close * 100.0;
        let strategy_return = results.metrics.total_return * 100.0;
        println!("\n5. Strategy vs Buy-and-Hold:");
        println!("   Strategy Return: {:.2}%", strategy_return);
        println!("   Buy-and-Hold Return: {:.2}%", buy_hold_return);
        if strategy_return > buy_hold_return {
            println!("   Strategy outperforms by {:.2}%", strategy_return - buy_hold_return);
        } else {
            println!("   Buy-and-hold outperforms by {:.2}%", buy_hold_return - strategy_return);
        }
    }

    // Run LLM analysis
    println!("\n6. Running LLM analysis...");
    let assistant = BacktestingAssistant::with_provider(MockLlmClient::new());
    let analysis = assistant.analyze(&results).await?;

    println!("\n   Analysis Summary:");
    // Print first few lines of analysis
    for line in analysis.analysis.lines().take(10) {
        println!("   {}", line);
    }
    println!("   ...");

    Ok(())
}
