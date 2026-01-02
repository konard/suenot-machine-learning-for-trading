//! Backtest example
//!
//! This example demonstrates how to:
//! 1. Run a backtest on historical data
//! 2. Analyze trading performance
//! 3. Generate performance report

use subgraph_mining_trading::{
    data::Candle,
    trading::{BacktestConfig, Backtester, StrategyConfig},
    utils::Stats,
};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("=== Subgraph Mining Strategy Backtest ===\n");

    // Create synthetic historical data
    println!("Generating synthetic historical data...");
    let data = generate_historical_data();
    println!("  Created {} days of data for {} symbols",
        data.values().next().map(|v| v.len()).unwrap_or(0),
        data.len()
    );

    // Configure backtest
    let config = BacktestConfig {
        start_time: 0,
        end_time: i64::MAX,
        rebalance_interval: 86400, // Daily rebalance
        correlation_threshold: 0.6,
        window_size: 20,
        min_support: 1,
        strategy_config: StrategyConfig {
            initial_capital: 10000.0,
            max_position_size: 0.2,
            min_signal_strength: 0.5,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.10,
            max_positions: 5,
            allow_short: false,
        },
        trading_fee_pct: 0.001,
        slippage_pct: 0.0005,
    };

    // Run backtest
    println!("\nRunning backtest...");
    let backtester = Backtester::new(config);

    match backtester.run(&data) {
        Ok(result) => {
            // Print results
            println!("\n=== Backtest Results ===\n");

            println!("Performance Metrics:");
            println!("  Total Return:       {:.2}%", result.total_return * 100.0);
            println!("  Annualized Return:  {:.2}%", result.annualized_return * 100.0);
            println!("  Sharpe Ratio:       {:.2}", result.sharpe_ratio);
            println!("  Sortino Ratio:      {:.2}", result.sortino_ratio);
            println!("  Max Drawdown:       {:.2}%", result.max_drawdown * 100.0);

            println!("\nTrading Statistics:");
            println!("  Total Trades:       {}", result.total_trades);
            println!("  Winning Trades:     {}", result.winning_trades);
            println!("  Losing Trades:      {}", result.losing_trades);
            println!("  Win Rate:           {:.2}%", result.win_rate * 100.0);
            println!("  Profit Factor:      {:.2}", result.profit_factor);
            println!("  Avg Trade Return:   {:.2}%", result.avg_trade_return * 100.0);

            println!("\nCapital:");
            println!("  Initial:            ${:.2}", 10000.0);
            println!("  Final:              ${:.2}", result.final_capital);

            // Show equity curve summary
            if !result.equity_curve.is_empty() {
                let min_equity = result.equity_curve.iter()
                    .map(|(_, v)| *v)
                    .fold(f64::INFINITY, f64::min);
                let max_equity = result.equity_curve.iter()
                    .map(|(_, v)| *v)
                    .fold(f64::NEG_INFINITY, f64::max);

                println!("\nEquity Curve:");
                println!("  Min:                ${:.2}", min_equity);
                println!("  Max:                ${:.2}", max_equity);
                println!("  Data Points:        {}", result.equity_curve.len());
            }

            // Show trades
            if !result.trades.is_empty() {
                println!("\nRecent Trades (last 10):");
                for trade in result.trades.iter().rev().take(10) {
                    println!("  {} {} {} @ ${:.2} ({})",
                        trade.side,
                        trade.quantity,
                        trade.symbol,
                        trade.price,
                        trade.reason
                    );
                }
            }

            // Show daily returns stats
            if !result.daily_returns.is_empty() {
                println!("\nDaily Returns Analysis:");
                println!("  Mean:               {:.4}%", Stats::mean(&result.daily_returns) * 100.0);
                println!("  Std Dev:            {:.4}%", Stats::std_dev(&result.daily_returns) * 100.0);
                println!("  Skewness:           {:.4}", Stats::skewness(&result.daily_returns));
                println!("  Kurtosis:           {:.4}", Stats::kurtosis(&result.daily_returns));
            }

            println!("\nSummary: {}", result.summary());
        }
        Err(e) => {
            println!("Backtest failed: {}", e);
        }
    }

    println!("\n=== Backtest Complete ===");
    Ok(())
}

/// Generate synthetic historical data
fn generate_historical_data() -> HashMap<String, Vec<Candle>> {
    let symbols = vec![
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "AVAXUSDT", "DOTUSDT",
    ];

    let num_days = 365;
    let base_time = 1672531200; // 2023-01-01

    // Generate BTC as base asset
    let btc_candles = generate_trending_candles(
        num_days,
        base_time,
        50000.0, // Initial price
        0.0002,  // Slight upward drift
        0.02,    // Daily volatility
    );

    let mut data = HashMap::new();
    data.insert("BTCUSDT".to_string(), btc_candles.clone());

    // Generate correlated assets
    for (i, symbol) in symbols.iter().skip(1).enumerate() {
        let correlation = 0.8 - (i as f64 * 0.05);
        let initial_price = match i {
            0 => 3000.0,   // ETH
            1 => 100.0,    // SOL
            2 => 300.0,    // BNB
            3 => 0.50,     // XRP
            4 => 0.40,     // ADA
            5 => 30.0,     // AVAX
            _ => 5.0,      // DOT
        };

        let candles = generate_correlated_candles(
            &btc_candles,
            initial_price,
            correlation,
            0.03, // Slightly higher volatility
        );

        data.insert(symbol.to_string(), candles);
    }

    data
}

/// Generate trending candles
fn generate_trending_candles(
    n: usize,
    base_time: i64,
    initial_price: f64,
    drift: f64,
    volatility: f64,
) -> Vec<Candle> {
    let mut candles = Vec::new();
    let mut price = initial_price;

    for i in 0..n {
        // Generate daily return with drift and mean reversion
        let random_component = pseudo_random(i) * volatility;
        let ret = drift + random_component;

        let open = price;
        price *= 1.0 + ret;
        let close = price;

        let high = open.max(close) * (1.0 + random_component.abs() * 0.5);
        let low = open.min(close) * (1.0 - random_component.abs() * 0.5);

        let volume = 10000.0 + pseudo_random(i * 2 + 1).abs() * 5000.0;

        candles.push(Candle::new(
            base_time + (i as i64 * 86400),
            open,
            high,
            low,
            close,
            volume,
        ));
    }

    candles
}

/// Generate correlated candles based on reference
fn generate_correlated_candles(
    reference: &[Candle],
    initial_price: f64,
    correlation: f64,
    extra_volatility: f64,
) -> Vec<Candle> {
    let mut candles = Vec::new();
    let mut price = initial_price;

    for (i, ref_candle) in reference.iter().enumerate() {
        // Get reference return
        let ref_return = if i > 0 {
            (ref_candle.close - reference[i - 1].close) / reference[i - 1].close
        } else {
            0.0
        };

        // Generate correlated return
        let idiosyncratic = pseudo_random(i * 3) * extra_volatility;
        let ret = ref_return * correlation + idiosyncratic * (1.0 - correlation);

        let open = price;
        price *= 1.0 + ret;
        let close = price;

        let high = open.max(close) * (1.0 + ret.abs() * 0.3);
        let low = open.min(close) * (1.0 - ret.abs() * 0.3);

        let volume = ref_candle.volume * (0.5 + pseudo_random(i * 4).abs());

        candles.push(Candle::new(
            ref_candle.timestamp,
            open,
            high,
            low,
            close,
            volume,
        ));
    }

    candles
}

/// Simple pseudo-random number generator
fn pseudo_random(seed: usize) -> f64 {
    let x = seed as f64;
    ((x * 0.1).sin() + (x * 0.23).cos() + (x * 0.37).sin()) / 3.0
}
