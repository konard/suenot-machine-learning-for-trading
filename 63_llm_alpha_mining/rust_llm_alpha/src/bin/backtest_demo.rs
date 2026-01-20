//! Backtesting Demo
//!
//! Demonstrates how to backtest alpha factors with realistic trading simulation.

use llm_alpha_mining::data::generate_synthetic_data;
use llm_alpha_mining::alpha::{AlphaEvaluator, predefined_factors};
use llm_alpha_mining::backtest::Backtester;

fn main() -> anyhow::Result<()> {
    println!("====================================================");
    println!("LLM Alpha Mining - Backtesting Demo (Rust)");
    println!("====================================================");

    // 1. Load data
    println!("\n1. LOADING DATA");
    println!("{}", "-".repeat(40));

    let data = generate_synthetic_data("BTCUSDT", 500, 42);
    println!("Loaded {} records for BTCUSDT", data.len());
    println!("Date range: {} to {}",
             data.candles.first().map(|c| c.timestamp.date_naive()).unwrap(),
             data.candles.last().map(|c| c.timestamp.date_naive()).unwrap());

    // 2. Initialize backtester
    println!("\n2. INITIALIZING BACKTESTER");
    println!("{}", "-".repeat(40));

    let backtester = Backtester::new(100_000.0)
        .position_size(0.5)
        .commission(0.001)
        .slippage(0.0005)
        .long_threshold(0.5)
        .short_threshold(Some(-0.5))
        .max_holding_periods(10);

    println!("Initial capital: $100,000");
    println!("Position size: 50%");
    println!("Commission: 0.1%");
    println!("Slippage: 0.05%");

    // 3. Backtest predefined factors
    println!("\n3. BACKTESTING PREDEFINED FACTORS");
    println!("{}", "-".repeat(40));

    let factors = predefined_factors();
    let evaluator = AlphaEvaluator::new(&data);
    let prices = data.close_prices();
    let timestamps = data.timestamps();

    let mut results = Vec::new();

    for factor in &factors {
        let values = match evaluator.evaluate(factor) {
            Ok(v) => v,
            Err(e) => {
                println!("\n{}: Error - {}", factor.name, e);
                continue;
            }
        };

        // Normalize to z-score
        let valid_vals: Vec<f64> = values.iter().filter(|v| !v.is_nan()).cloned().collect();
        if valid_vals.is_empty() {
            continue;
        }

        let mean: f64 = valid_vals.iter().sum::<f64>() / valid_vals.len() as f64;
        let variance: f64 = valid_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / valid_vals.len() as f64;
        let std = variance.sqrt();

        let signals: Vec<f64> = values.iter()
            .map(|v| if v.is_nan() || std == 0.0 { f64::NAN } else { (v - mean) / std })
            .collect();

        // Run backtest
        match backtester.run(&signals, &prices, &timestamps) {
            Ok(result) => {
                println!("\n{}:", factor.name);
                println!("  Total Return: {:+.2}%", result.total_return * 100.0);
                println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
                println!("  Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
                println!("  Win Rate: {:.2}%", result.win_rate * 100.0);
                println!("  Total Trades: {}", result.total_trades);

                results.push((factor.name.clone(), result));
            }
            Err(e) => {
                println!("\n{}: Backtest error - {}", factor.name, e);
            }
        }
    }

    // 4. Compare all factors
    println!("\n4. FACTOR COMPARISON");
    println!("{}", "-".repeat(40));

    if !results.is_empty() {
        println!("\n{:<20} {:>10} {:>10} {:>10} {:>10}",
                 "Factor", "Return", "Sharpe", "MaxDD", "Trades");
        println!("{}", "-".repeat(65));

        for (name, result) in &results {
            println!("{:<20} {:>9.1}% {:>10.2} {:>9.1}% {:>10}",
                     name,
                     result.total_return * 100.0,
                     result.sharpe_ratio,
                     result.max_drawdown * 100.0,
                     result.total_trades);
        }
    }

    // 5. Best factor details
    println!("\n5. BEST FACTOR DETAILS");
    println!("{}", "-".repeat(40));

    if let Some((name, best)) = results.iter()
        .max_by(|a, b| a.1.sharpe_ratio.partial_cmp(&b.1.sharpe_ratio).unwrap())
    {
        println!("\nBest factor by Sharpe: {}", name);
        println!("{}", best.summary());

        // Show recent trades
        println!("\nRecent Trades:");
        let recent: Vec<_> = best.trades.iter().rev().take(5).collect();
        for trade in recent.iter().rev() {
            let side = match trade.side {
                llm_alpha_mining::backtest::TradeSide::Long => "LONG",
                llm_alpha_mining::backtest::TradeSide::Short => "SHORT",
            };
            println!("  {} {} @ ${:.2} -> ${:.2}, PnL: ${:+.2} ({})",
                     trade.timestamp.date_naive(),
                     side,
                     trade.entry_price,
                     trade.exit_price.unwrap_or(0.0),
                     trade.pnl,
                     trade.exit_reason);
        }
    }

    // 6. Strategy variations
    println!("\n6. STRATEGY VARIATIONS");
    println!("{}", "-".repeat(40));

    if let Some(factor) = factors.first() {
        let values = evaluator.evaluate(factor)?;
        let valid_vals: Vec<f64> = values.iter().filter(|v| !v.is_nan()).cloned().collect();
        let mean: f64 = valid_vals.iter().sum::<f64>() / valid_vals.len() as f64;
        let variance: f64 = valid_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / valid_vals.len() as f64;
        let std = variance.sqrt();

        let signals: Vec<f64> = values.iter()
            .map(|v| if v.is_nan() || std == 0.0 { f64::NAN } else { (v - mean) / std })
            .collect();

        println!("\nFactor: {}", factor.name);
        println!("\n{:<20} {:>10} {:>10} {:>10}",
                 "Strategy", "Return", "Sharpe", "Trades");
        println!("{}", "-".repeat(55));

        // Long only
        let long_only = Backtester::new(100_000.0)
            .position_size(0.5)
            .commission(0.001)
            .long_threshold(0.5)
            .short_threshold(None)
            .max_holding_periods(10);

        if let Ok(r) = long_only.run(&signals, &prices, &timestamps) {
            println!("{:<20} {:>9.1}% {:>10.2} {:>10}",
                     "Long Only", r.total_return * 100.0, r.sharpe_ratio, r.total_trades);
        }

        // Long-Short
        let long_short = Backtester::new(100_000.0)
            .position_size(0.5)
            .commission(0.001)
            .long_threshold(0.5)
            .short_threshold(Some(-0.5))
            .max_holding_periods(10);

        if let Ok(r) = long_short.run(&signals, &prices, &timestamps) {
            println!("{:<20} {:>9.1}% {:>10.2} {:>10}",
                     "Long-Short", r.total_return * 100.0, r.sharpe_ratio, r.total_trades);
        }

        // With Stop Loss
        let with_sl = Backtester::new(100_000.0)
            .position_size(0.5)
            .commission(0.001)
            .long_threshold(0.5)
            .short_threshold(Some(-0.5))
            .max_holding_periods(10)
            .stop_loss(Some(0.03));

        if let Ok(r) = with_sl.run(&signals, &prices, &timestamps) {
            println!("{:<20} {:>9.1}% {:>10.2} {:>10}",
                     "With 3% Stop Loss", r.total_return * 100.0, r.sharpe_ratio, r.total_trades);
        }

        // High conviction
        let high_conv = Backtester::new(100_000.0)
            .position_size(0.5)
            .commission(0.001)
            .long_threshold(1.5)
            .short_threshold(Some(-1.5))
            .max_holding_periods(10);

        if let Ok(r) = high_conv.run(&signals, &prices, &timestamps) {
            println!("{:<20} {:>9.1}% {:>10.2} {:>10}",
                     "High Conviction", r.total_return * 100.0, r.sharpe_ratio, r.total_trades);
        }
    }

    println!("\n====================================================");
    println!("Demo complete!");

    Ok(())
}
