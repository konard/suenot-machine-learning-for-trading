//! Example: Backtesting a Graph Transformer trading strategy
//!
//! This example demonstrates how to backtest a trading strategy
//! using historical data and the Graph Transformer model.

use graph_transformer_trading::prelude::*;
use graph_transformer_trading::api::types::Kline;
use graph_transformer_trading::strategy::portfolio::{Portfolio, Position, PortfolioBuilder};
use std::collections::HashMap;

fn main() {
    println!("=== Graph Transformer Backtest Example ===\n");

    // Generate simulated historical data
    println!("1. Generating simulated market data...");
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"];
    let num_days = 100;

    let mut prices: HashMap<String, Vec<Kline>> = HashMap::new();

    for (i, symbol) in symbols.iter().enumerate() {
        let base_price = match i {
            0 => 45000.0,  // BTC
            1 => 2500.0,   // ETH
            2 => 100.0,    // SOL
            3 => 300.0,    // BNB
            _ => 0.08,     // DOGE
        };

        let klines = generate_synthetic_klines(base_price, num_days, i as u64);
        prices.insert(symbol.to_string(), klines);
    }

    println!("   Generated {} days of data for {} symbols", num_days, symbols.len());

    // Create signals based on momentum
    println!("\n2. Generating trading signals...");
    let signals = generate_momentum_signals(&prices, &symbols);
    println!("   Generated {} signals", signals.len());

    // Build portfolio
    println!("\n3. Building portfolio from signals...");
    let portfolio_builder = PortfolioBuilder::new();
    let portfolio = portfolio_builder.build(&signals);

    println!("   Portfolio positions:");
    for pos in &portfolio.positions {
        let direction = if pos.direction > 0 { "LONG" } else { "SHORT" };
        println!("      {} {} - Weight: {:.2}%", direction, pos.symbol, pos.weight * 100.0);
    }
    println!("   Cash weight: {:.2}%", portfolio.cash_weight * 100.0);

    // Run backtest
    println!("\n4. Running backtest...");
    let backtest_config = graph_transformer_trading::backtest::engine::BacktestConfig {
        initial_capital: 100000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        rebalance_frequency: 1,
    };

    let engine = BacktestEngine::new(backtest_config);
    let report = engine.run_simple(&portfolio, &prices);

    // Display results
    println!("\n5. Backtest Results:");
    println!("{}", report.summary());

    // Compare with buy-and-hold BTC
    println!("\n6. Comparison with Buy-and-Hold BTC:");
    let btc_portfolio = Portfolio {
        positions: vec![Position {
            symbol: "BTCUSDT".to_string(),
            weight: 1.0,
            direction: 1,
        }],
        cash_weight: 0.0,
    };

    let btc_report = engine.run_simple(&btc_portfolio, &prices);

    println!("   Strategy Return: {:.2}%", (report.final_capital / report.initial_capital - 1.0) * 100.0);
    println!("   BTC Buy-Hold Return: {:.2}%", (btc_report.final_capital / btc_report.initial_capital - 1.0) * 100.0);
    println!("   Strategy Sharpe: {:.2}", report.metrics.sharpe_ratio);
    println!("   BTC Sharpe: {:.2}", btc_report.metrics.sharpe_ratio);
    println!("   Strategy Max DD: {:.2}%", report.metrics.max_drawdown * 100.0);
    println!("   BTC Max DD: {:.2}%", btc_report.metrics.max_drawdown * 100.0);

    // Equity curve summary
    println!("\n7. Equity Curve (sampled):");
    let step = report.equity_curve.len() / 10;
    for (i, &equity) in report.equity_curve.iter().step_by(step.max(1)).enumerate() {
        let bar_len = ((equity / report.initial_capital - 0.8) * 100.0).max(0.0).min(40.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("   Day {:>3}: ${:>10.2} [{:<40}]", i * step, equity, bar);
    }

    println!("\n=== Backtest complete! ===");
}

/// Generate synthetic kline data with trend and volatility
fn generate_synthetic_klines(base_price: f64, num_days: usize, seed: u64) -> Vec<Kline> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut klines = Vec::with_capacity(num_days);
    let mut price = base_price;

    for day in 0..num_days {
        let trend = 0.0005; // Slight upward trend
        let volatility = 0.02; // 2% daily volatility

        let return_pct = trend + volatility * (rng.gen::<f64>() * 2.0 - 1.0);
        let close = price * (1.0 + return_pct);

        let high = close * (1.0 + rng.gen::<f64>() * 0.01);
        let low = close * (1.0 - rng.gen::<f64>() * 0.01);
        let open = price;
        let volume = rng.gen::<f64>() * 1000.0 + 500.0;

        klines.push(Kline {
            start_time: (day as i64) * 86400000,
            open,
            high,
            low,
            close,
            volume,
            turnover: close * volume,
        });

        price = close;
    }

    klines
}

/// Generate signals based on momentum
fn generate_momentum_signals(
    prices: &HashMap<String, Vec<Kline>>,
    symbols: &[&str],
) -> Vec<Signal> {
    let mut signals = Vec::new();

    for symbol in symbols {
        if let Some(klines) = prices.get(*symbol) {
            if klines.len() < 20 {
                continue;
            }

            // Calculate 10-day momentum
            let recent = &klines[klines.len() - 10..];
            let momentum = (recent.last().unwrap().close - recent.first().unwrap().close)
                / recent.first().unwrap().close;

            // Calculate volatility-adjusted signal
            let returns: Vec<f64> = recent
                .windows(2)
                .map(|w| (w[1].close - w[0].close) / w[0].close)
                .collect();
            let volatility: f64 = returns.iter().map(|r| r.abs()).sum::<f64>() / returns.len() as f64;

            let signal_strength = momentum / volatility.max(0.001);

            let (signal_type, confidence) = if signal_strength > 1.0 {
                (SignalType::Long, 0.5 + signal_strength.min(1.0) * 0.2)
            } else if signal_strength < -1.0 {
                (SignalType::Short, 0.5 + signal_strength.abs().min(1.0) * 0.2)
            } else {
                continue;
            };

            signals.push(Signal::new(
                symbol,
                signal_type,
                confidence,
                momentum,
            ));
        }
    }

    signals
}
