//! Example: Simple vectorized backtest
//!
//! This example demonstrates a simple vectorized backtest approach,
//! similar to the Python notebook in this chapter.
//!
//! Usage:
//!   cargo run --example simple_backtest

use anyhow::Result;
use chrono::{Duration, Utc};
use rust_backtester::{
    api::BybitClient,
    backtest::BacktestEngine,
    models::Timeframe,
    utils::{extract_closes, returns, sma, PerformanceMetrics},
};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           SIMPLE VECTORIZED BACKTEST EXAMPLE                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Fetch data from Bybit
    println!("Fetching historical data from Bybit...");
    let client = BybitClient::new();
    let end = Utc::now();
    let start = end - Duration::days(90);

    let candles = client
        .get_historical_klines("BTCUSDT", Timeframe::H4, start, end)
        .await?;

    if candles.is_empty() {
        println!("No data available. Exiting.");
        return Ok(());
    }

    println!("Fetched {} candles\n", candles.len());

    // Extract closing prices
    let closes = extract_closes(&candles);
    let price_returns = returns(&closes);

    // Strategy 1: Buy and Hold
    println!("═══════════════════════════════════════");
    println!("Strategy 1: Buy and Hold");
    println!("═══════════════════════════════════════");

    let bnh_returns = price_returns.clone();
    let bnh_metrics = PerformanceMetrics::from_returns(&bnh_returns, 365.0 * 6.0); // 4h = 6 per day
    bnh_metrics.print();

    // Strategy 2: SMA Crossover (Vectorized)
    println!("\n═══════════════════════════════════════");
    println!("Strategy 2: SMA Crossover (10/50)");
    println!("═══════════════════════════════════════");

    let fast_sma = sma(&closes, 10);
    let slow_sma = sma(&closes, 50);

    // Generate signals: 1 when fast > slow, 0 otherwise
    let signals: Vec<f64> = fast_sma
        .iter()
        .zip(slow_sma.iter())
        .map(|(f, s)| match (f, s) {
            (Some(fast), Some(slow)) if fast > slow => 1.0,
            (Some(_), Some(_)) => 0.0,
            _ => 0.0,
        })
        .collect();

    // Skip first element to align with returns
    let strategy_returns = BacktestEngine::run_vectorized(
        &signals[1..],
        &price_returns,
        0.001, // 0.1% transaction cost
    );

    let sma_metrics = PerformanceMetrics::from_returns(&strategy_returns, 365.0 * 6.0);
    sma_metrics.print();

    // Strategy 3: Mean Reversion (Vectorized)
    println!("\n═══════════════════════════════════════");
    println!("Strategy 3: Mean Reversion (20-period)");
    println!("═══════════════════════════════════════");

    let sma_20 = sma(&closes, 20);

    // Go long when price is below SMA, short when above
    let mr_signals: Vec<f64> = closes
        .iter()
        .zip(sma_20.iter())
        .map(|(price, sma_val)| match sma_val {
            Some(sma) if price < sma => 1.0,  // Below SMA: expect reversion up
            Some(sma) if price > sma => -1.0, // Above SMA: expect reversion down
            _ => 0.0,
        })
        .collect();

    let mr_returns = BacktestEngine::run_vectorized(
        &mr_signals[1..],
        &price_returns,
        0.001,
    );

    let mr_metrics = PerformanceMetrics::from_returns(&mr_returns, 365.0 * 6.0);
    mr_metrics.print();

    // Strategy 4: Momentum
    println!("\n═══════════════════════════════════════");
    println!("Strategy 4: Momentum (10-period)");
    println!("═══════════════════════════════════════");

    let lookback = 10;
    let mut momentum_signals = vec![0.0; closes.len()];

    for i in lookback..closes.len() {
        let momentum = (closes[i] - closes[i - lookback]) / closes[i - lookback];
        momentum_signals[i] = if momentum > 0.0 { 1.0 } else { -1.0 };
    }

    let momentum_returns = BacktestEngine::run_vectorized(
        &momentum_signals[1..],
        &price_returns,
        0.001,
    );

    let momentum_metrics = PerformanceMetrics::from_returns(&momentum_returns, 365.0 * 6.0);
    momentum_metrics.print();

    // Summary comparison
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    STRATEGY COMPARISON                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Strategy          │ Return   │ Sharpe │ MaxDD  │ WinRate    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Buy & Hold        │ {:>7.2}% │ {:>6.2} │ {:>5.1}% │ {:>6.1}%    ║",
        bnh_metrics.total_return * 100.0,
        bnh_metrics.sharpe_ratio,
        bnh_metrics.max_drawdown * 100.0,
        bnh_metrics.win_rate * 100.0
    );
    println!(
        "║ SMA Crossover     │ {:>7.2}% │ {:>6.2} │ {:>5.1}% │ {:>6.1}%    ║",
        sma_metrics.total_return * 100.0,
        sma_metrics.sharpe_ratio,
        sma_metrics.max_drawdown * 100.0,
        sma_metrics.win_rate * 100.0
    );
    println!(
        "║ Mean Reversion    │ {:>7.2}% │ {:>6.2} │ {:>5.1}% │ {:>6.1}%    ║",
        mr_metrics.total_return * 100.0,
        mr_metrics.sharpe_ratio,
        mr_metrics.max_drawdown * 100.0,
        mr_metrics.win_rate * 100.0
    );
    println!(
        "║ Momentum          │ {:>7.2}% │ {:>6.2} │ {:>5.1}% │ {:>6.1}%    ║",
        momentum_metrics.total_return * 100.0,
        momentum_metrics.sharpe_ratio,
        momentum_metrics.max_drawdown * 100.0,
        momentum_metrics.win_rate * 100.0
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
