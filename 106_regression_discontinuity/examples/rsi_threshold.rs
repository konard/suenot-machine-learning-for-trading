//! RSI Threshold Trading Example.
//!
//! Demonstrates RDD analysis on RSI indicator thresholds
//! using cryptocurrency data from Bybit.

use rdd_trading::backtest::engine::BacktestEngine;
use rdd_trading::data::bybit::generate_synthetic_data;
use rdd_trading::data::indicators::compute_rsi;
use rdd_trading::model::rdd::{Kernel, RDDModel};
use rdd_trading::trading::strategy::RDDStrategy;

fn main() {
    println!("=== RSI Threshold RDD Trading Example ===\n");

    // Generate synthetic market data (in production, use BybitClient)
    println!("Generating synthetic BTC/USDT hourly data...");
    let market_data = generate_synthetic_data("BTCUSDT", 2000, 50000.0, 0.015, 42);

    let prices = market_data.close_prices();
    let rsi = compute_rsi(&prices, 14);

    println!("Data points: {}", prices.len());
    println!(
        "Price range: {:.2} - {:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!();

    // --- Analyze RSI distribution ---
    let valid_rsi: Vec<f64> = rsi.iter().filter(|&&r| r > 0.0 && r < 100.0).cloned().collect();
    let rsi_below_30 = valid_rsi.iter().filter(|&&r| r < 30.0).count();
    let rsi_above_70 = valid_rsi.iter().filter(|&&r| r > 70.0).count();

    println!("RSI Distribution:");
    println!(
        "  Mean RSI: {:.2}",
        valid_rsi.iter().sum::<f64>() / valid_rsi.len() as f64
    );
    println!(
        "  Observations below 30: {} ({:.1}%)",
        rsi_below_30,
        rsi_below_30 as f64 / valid_rsi.len() as f64 * 100.0
    );
    println!(
        "  Observations above 70: {} ({:.1}%)",
        rsi_above_70,
        rsi_above_70 as f64 / valid_rsi.len() as f64 * 100.0
    );
    println!();

    // --- RDD Analysis at RSI = 30 (oversold) ---
    println!("=== RDD Analysis: RSI = 30 (Oversold) ===");

    // Calculate 24-hour forward returns
    let forward_returns = market_data.forward_returns(24);

    // Filter valid observations
    let valid_obs: Vec<(f64, f64)> = rsi
        .iter()
        .zip(forward_returns.iter())
        .filter(|(r, f)| **r > 0.0 && **r < 100.0 && !f.is_nan())
        .map(|(&r, &f)| (r, f))
        .collect();

    let running: Vec<f64> = valid_obs.iter().map(|(r, _)| *r).collect();
    let outcomes: Vec<f64> = valid_obs.iter().map(|(_, f)| *f).collect();

    println!("Valid observations for RDD: {}", valid_obs.len());

    // Fit RDD at RSI = 30
    let mut model_30 = RDDModel::new(30.0, 5.0, Kernel::Triangular);
    match model_30.fit(&running, &outcomes) {
        Ok(results) => {
            println!("\nRDD Results at RSI = 30:");
            println!("  Treatment Effect: {:.4}% ({:.4}%)", results.tau * 100.0, results.se * 100.0);
            println!("  95% CI: [{:.4}%, {:.4}%]", results.ci_lower * 100.0, results.ci_upper * 100.0);
            println!("  p-value: {:.4}", results.p_value);
            println!(
                "  Interpretation: Crossing below RSI 30 is associated with a {:.2}% 24h return",
                results.tau * 100.0
            );
        }
        Err(e) => println!("Error fitting RSI=30 model: {}", e),
    }

    // --- RDD Analysis at RSI = 70 (overbought) ---
    println!("\n=== RDD Analysis: RSI = 70 (Overbought) ===");

    let mut model_70 = RDDModel::new(70.0, 5.0, Kernel::Triangular);
    match model_70.fit(&running, &outcomes) {
        Ok(results) => {
            println!("RDD Results at RSI = 70:");
            println!("  Treatment Effect: {:.4}% ({:.4}%)", results.tau * 100.0, results.se * 100.0);
            println!("  95% CI: [{:.4}%, {:.4}%]", results.ci_lower * 100.0, results.ci_upper * 100.0);
            println!("  p-value: {:.4}", results.p_value);
            println!(
                "  Interpretation: Crossing above RSI 70 is associated with a {:.2}% 24h return",
                results.tau * 100.0
            );
        }
        Err(e) => println!("Error fitting RSI=70 model: {}", e),
    }

    // --- Backtest RSI Strategy ---
    println!("\n=== Backtesting RSI Oversold Strategy ===");

    let mut strategy = RDDStrategy::rsi_oversold()
        .with_min_observations(50)
        .with_holding_period(24);

    match strategy.fit(&running, &outcomes) {
        Ok(_) => {
            println!("{}", strategy.summary());

            // Run backtest
            let engine = BacktestEngine::new(100_000.0, 0.001)
                .with_slippage(0.0002)
                .with_max_position(0.1);

            match engine.run_strategy(&prices[14..], &rsi[14..], &strategy) {
                Ok(results) => {
                    println!("\n{}", results.summary());
                }
                Err(e) => println!("Backtest error: {}", e),
            }
        }
        Err(e) => println!("Strategy fitting error: {}", e),
    }

    println!("\n=== Done ===");
}
