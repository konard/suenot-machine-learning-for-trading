//! Trading strategy with SHAP-enhanced signals.
//!
//! This example demonstrates:
//! 1. Building a SHAP-enhanced trading strategy
//! 2. Generating signals with explanations
//! 3. Running a backtest
//! 4. Comparing results with and without SHAP filtering

use shap_trading::backtest::engine::{BacktestConfig, BacktestEngine};
use shap_trading::data::bybit::BybitClient;
use shap_trading::trading::signals::{SignalGenerator, TradingSignal};
use shap_trading::trading::strategy::StrategyBuilder;
use rand::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== SHAP-Enhanced Trading Strategy ===\n");

    // Create Bybit client and fetch data
    let client = BybitClient::new();
    let data = client.fetch_klines("BTCUSDT", "60", 500).await?;

    println!("Data source: {}", data.source);
    println!("Candles: {}", data.candles.len());

    let prices = data.close_prices();
    let volumes = data.volumes();

    if prices.len() < 50 {
        println!("Not enough data for backtesting");
        return Ok(());
    }

    // Build trading strategy
    let mut strategy = StrategyBuilder::new()
        .weights(vec![0.5, 0.3, 0.2, -0.4, 0.1, 0.15, 0.25])
        .min_confidence(0.1)
        .min_stability(0.3)
        .build();

    // Generate background data for SHAP explainer
    let signal_gen = SignalGenerator::new(20, 0.1);
    let mut background_data = Vec::new();

    for i in 30..prices.len().saturating_sub(100) {
        let features = signal_gen.compute_features(&prices[..=i], &volumes[..=i]);
        if features.len() == 7 {
            background_data.push(features);
        }
    }

    if !background_data.is_empty() {
        strategy.init_explainer(background_data.clone(), 0.5);
    }

    // Generate signals for backtesting
    println!("\nGenerating trading signals...");
    let mut signals = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..prices.len() {
        if i < 30 {
            // Not enough data for features
            signals.push(TradingSignal::new(0.5, None));
            continue;
        }

        match strategy.generate_signal(&prices[..=i], &volumes[..=i]) {
            Some(mut signal) => {
                // Simulate SHAP stability (in practice, compute from actual SHAP values)
                let stability = 0.6 + 0.3 * rng.gen::<f64>();
                signal = signal.with_stability(stability);
                signals.push(signal);
            }
            None => {
                signals.push(TradingSignal::new(0.5, None));
            }
        }
    }

    // Backtest without SHAP filter
    println!("\n--- Backtest: No SHAP Filter ---");
    let config_no_shap = BacktestConfig {
        use_shap_filter: false,
        ..Default::default()
    };
    let engine_no_shap = BacktestEngine::new(config_no_shap);
    let result_no_shap = engine_no_shap.run(&prices, &signals);
    result_no_shap.print_summary();

    // Backtest with SHAP filter
    println!("\n--- Backtest: With SHAP Filter ---");
    let config_shap = BacktestConfig {
        use_shap_filter: true,
        shap_stability_threshold: 0.5,
        ..Default::default()
    };
    let engine_shap = BacktestEngine::new(config_shap);
    let result_shap = engine_shap.run(&prices, &signals);
    result_shap.print_summary();

    // Compare results
    println!("\n--- Comparison ---");
    let sharpe_no_shap = result_no_shap.metrics.get("sharpe_ratio").unwrap_or(&0.0);
    let sharpe_shap = result_shap.metrics.get("sharpe_ratio").unwrap_or(&0.0);

    let improvement = if sharpe_no_shap.abs() > 0.01 {
        (sharpe_shap - sharpe_no_shap) / sharpe_no_shap.abs() * 100.0
    } else {
        0.0
    };

    println!("Sharpe (No Filter): {:.4}", sharpe_no_shap);
    println!("Sharpe (SHAP Filter): {:.4}", sharpe_shap);
    println!("Improvement: {:.1}%", improvement);

    let dd_no_shap = result_no_shap.metrics.get("max_drawdown").unwrap_or(&0.0);
    let dd_shap = result_shap.metrics.get("max_drawdown").unwrap_or(&0.0);
    println!("\nMax Drawdown (No Filter): {:.2}%", dd_no_shap * 100.0);
    println!("Max Drawdown (SHAP Filter): {:.2}%", dd_shap * 100.0);

    // Show sample signal explanations
    println!("\n--- Sample Signal Explanations ---");
    for (i, signal) in signals.iter().enumerate().skip(prices.len() - 5) {
        println!(
            "Candle {}: prob={:.3}, direction={}, confidence={:.3}, stability={:.3}",
            i, signal.probability, signal.direction, signal.confidence, signal.shap_stability
        );

        if let Some(ref shap_values) = signal.shap_values {
            println!("  Top drivers:");
            for (name, value) in shap_values.top_contributors(3) {
                let dir = if value > 0.0 { "+" } else { "" };
                println!("    {}: {}{:.4}", name, dir, value);
            }
        }
    }

    println!("\n=== Strategy Demo Complete ===");
    Ok(())
}
