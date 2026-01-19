//! Backtesting Demo
//!
//! Demonstrates backtesting of LLM-derived trading strategies.
//!
//! Run with: cargo run --bin backtest_demo

use bloomberggpt_trading::{
    Backtester, BacktestConfig, TradingSignal,
    backtest::print_backtest_report,
    data::{MockLoader, OHLCVBar, Interval},
    signals::{SignalType, SignalSource},
};
use chrono::{Duration, Utc};
use std::collections::HashMap;
use rand::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=".repeat(60));
    println!("BloombergGPT Trading - Backtesting Demo");
    println!("=".repeat(60));

    // Configuration
    let symbols = vec!["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"];
    let n_days = 252; // One trading year
    let start_date = Utc::now() - Duration::days(n_days as i64);

    println!("\nBacktest Configuration:");
    println!("  Symbols: {}", symbols.join(", "));
    println!("  Period: {} days", n_days);
    println!("  Start Date: {}", start_date.format("%Y-%m-%d"));

    // Generate synthetic price data
    println!("\nGenerating synthetic market data...");

    let mut prices: HashMap<String, Vec<OHLCVBar>> = HashMap::new();

    for symbol in &symbols {
        let initial_price = match *symbol {
            "AAPL" => 175.0,
            "MSFT" => 380.0,
            "GOOGL" => 140.0,
            "TSLA" => 250.0,
            "NVDA" => 500.0,
            _ => 100.0,
        };

        let bars = MockLoader::generate_random_walk(
            symbol,
            start_date,
            n_days,
            Interval::Day1,
            initial_price,
        );
        prices.insert(symbol.to_string(), bars);
    }

    // Generate synthetic signals
    println!("Generating synthetic trading signals...");

    let signals = generate_realistic_signals(&symbols, start_date, n_days, 200, &prices);
    println!("  Generated {} signals", signals.len());

    // Signal distribution
    println!("\n  Signal distribution by symbol:");
    for symbol in &symbols {
        let count = signals.iter().filter(|s| s.symbol == *symbol).count();
        println!("    {}: {} signals", symbol, count);
    }

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        max_position_size: 0.15,
        max_total_exposure: 0.8,
        transaction_cost_bps: 10.0,
        slippage_bps: 5.0,
        signal_decay_hours: 48.0,
        rebalance_frequency: "daily".to_string(),
        min_signal_confidence: 0.6,
        short_selling: true,
    };

    println!("\nBacktest Settings:");
    println!("  Initial Capital: ${:.2}", config.initial_capital);
    println!("  Max Position Size: {:.0}%", config.max_position_size * 100.0);
    println!("  Transaction Costs: {:.0}bps", config.transaction_cost_bps);
    println!("  Signal Decay: {:.0}h", config.signal_decay_hours);
    println!("  Short Selling: {}", if config.short_selling { "Enabled" } else { "Disabled" });

    // Run backtest
    println!("\nRunning backtest...");
    let mut backtester = Backtester::new(config.clone());
    let result = backtester.run(&signals, &prices)?;

    // Print results
    print_backtest_report(&result);

    // Additional analysis
    if !result.trades.is_empty() {
        println!("\nTRADE ANALYSIS");
        println!("{}", "-".repeat(40));

        // Trades by symbol
        let mut trades_by_symbol: HashMap<String, (usize, f64)> = HashMap::new();
        for trade in &result.trades {
            let entry = trades_by_symbol
                .entry(trade.symbol.clone())
                .or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += trade.value;
        }

        println!("\nTrades by Symbol:");
        for (symbol, (count, value)) in &trades_by_symbol {
            println!("  {}: {} trades, ${:.2} total value", symbol, count, value);
        }

        // Buy vs Sell
        let buys = result.trades
            .iter()
            .filter(|t| matches!(t.trade_type, bloomberggpt_trading::backtest::TradeType::Buy))
            .count();
        let sells = result.trades
            .iter()
            .filter(|t| matches!(t.trade_type, bloomberggpt_trading::backtest::TradeType::Sell))
            .count();

        println!("\nTrade Direction: {} buys, {} sells", buys, sells);
    }

    // Benchmark comparison
    println!("\n{}", "-".repeat(40));
    println!("BENCHMARK COMPARISON");
    println!("{}", "-".repeat(40));

    // Calculate equal-weight buy and hold returns
    let mut bh_returns = Vec::new();
    for i in 1..n_days {
        let mut daily_return = 0.0;
        let mut count = 0;
        for bars in prices.values() {
            if i < bars.len() {
                daily_return += (bars[i].close - bars[i-1].close) / bars[i-1].close;
                count += 1;
            }
        }
        if count > 0 {
            bh_returns.push(daily_return / count as f64);
        }
    }

    let bh_total_return: f64 = bh_returns.iter()
        .fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;

    let mean_return = bh_returns.iter().sum::<f64>() / bh_returns.len() as f64;
    let variance: f64 = bh_returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>() / bh_returns.len() as f64;
    let bh_vol = variance.sqrt() * (252.0_f64).sqrt();
    let bh_sharpe = if bh_vol > 0.0 {
        (bh_total_return * 252.0 / bh_returns.len() as f64 - 0.02) / bh_vol
    } else {
        0.0
    };

    println!("\nEqual Weight Buy & Hold:");
    println!("  Total Return: {:.2}%", bh_total_return * 100.0);
    println!("  Volatility: {:.2}%", bh_vol * 100.0);
    println!("  Sharpe Ratio: {:.2}", bh_sharpe);

    let alpha = result.metrics.total_return - bh_total_return;
    let sharpe_diff = result.metrics.sharpe_ratio - bh_sharpe;

    println!("\nLLM Strategy vs Benchmark:");
    println!("  Alpha: {:+.2}%", alpha * 100.0);
    println!("  Sharpe Improvement: {:+.2}", sharpe_diff);

    println!("\n{}", "=".repeat(60));
    println!("Demo complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}

/// Generate realistic-looking trading signals for demonstration
fn generate_realistic_signals(
    symbols: &[&str],
    start: chrono::DateTime<Utc>,
    n_days: usize,
    n_signals: usize,
    prices: &HashMap<String, Vec<OHLCVBar>>,
) -> Vec<TradingSignal> {
    let mut rng = rand::thread_rng();
    let mut signals = Vec::with_capacity(n_signals);

    for _ in 0..n_signals {
        let day = rng.gen_range(0..n_days);
        let timestamp = start + Duration::days(day as i64);
        let symbol = symbols[rng.gen_range(0..symbols.len())];

        // Look at price movement to generate correlated signals
        let price_bars = prices.get(*symbol);
        let price_trend = if let Some(bars) = price_bars {
            if day > 0 && day < bars.len() {
                let lookback = 5.min(day);
                let recent_return = (bars[day].close - bars[day - lookback].close)
                    / bars[day - lookback].close;
                recent_return * 10.0 // Scale to signal range
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Add some noise but keep correlation with price
        let noise = rng.gen_range(-0.4..0.4);
        let signal_score = (price_trend * 0.3 + noise).clamp(-1.0, 1.0);

        // Skip weak signals
        if signal_score.abs() < 0.2 {
            continue;
        }

        let signal_type = SignalType::from_score(signal_score);
        let confidence = rng.gen_range(0.5..0.95);

        let source = match rng.gen_range(0..4) {
            0 => SignalSource::News,
            1 => SignalSource::Earnings,
            2 => SignalSource::Social,
            _ => SignalSource::Analyst,
        };

        signals.push(TradingSignal {
            symbol: symbol.to_string(),
            signal_type,
            strength: signal_score.abs(),
            confidence,
            source,
            timestamp,
            expiry: timestamp + Duration::hours(48),
            reason: format!("Synthetic signal from {:?}", source),
            sentiment_result: None,
        });
    }

    signals
}
