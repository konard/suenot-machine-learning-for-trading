//! Trading Strategy Example with Meta-Gradient Optimization
//!
//! Demonstrates a complete trading workflow:
//! 1. Generate simulated market data (mimicking Bybit crypto data)
//! 2. Train with meta-gradient optimization
//! 3. Backtest the strategy
//! 4. Report performance metrics

use meta_gradient_trading::prelude::*;
use meta_gradient_trading::backtest::engine::{BacktestEngine, PerformanceMetrics};
use meta_gradient_trading::data::bybit::SimulatedDataGenerator;
use meta_gradient_trading::data::features::FeatureGenerator;

fn print_metrics(name: &str, metrics: &PerformanceMetrics) {
    println!("  Total Return:       {:.2}%", metrics.total_return * 100.0);
    println!("  Annualized Return:  {:.2}%", metrics.annualized_return * 100.0);
    println!("  Ann. Volatility:    {:.2}%", metrics.annualized_volatility * 100.0);
    println!("  Sharpe Ratio:       {:.3}", metrics.sharpe_ratio);
    println!("  Sortino Ratio:      {:.3}", metrics.sortino_ratio);
    println!("  Max Drawdown:       {:.2}%", metrics.max_drawdown * 100.0);
    println!("  Win Rate:           {:.1}%", metrics.win_rate * 100.0);
    println!("  Profit Factor:      {:.3}", metrics.profit_factor);
    println!("  Num Trades:         {}", metrics.num_trades);
}

fn main() {
    println!("=== Trading Strategy with Meta-Gradient Optimization ===\n");

    // Generate market data simulating Bybit crypto assets
    println!("Generating simulated market data...");
    let btc_klines = SimulatedDataGenerator::generate_regime_changing_klines(500, 50000.0);
    let eth_klines = SimulatedDataGenerator::generate_regime_changing_klines(500, 3000.0);

    println!(
        "  BTC: {} klines, price range {:.0}-{:.0}",
        btc_klines.len(),
        btc_klines.iter().map(|k| k.close).fold(f64::INFINITY, f64::min),
        btc_klines.iter().map(|k| k.close).fold(f64::NEG_INFINITY, f64::max),
    );
    println!(
        "  ETH: {} klines, price range {:.0}-{:.0}",
        eth_klines.len(),
        eth_klines.iter().map(|k| k.close).fold(f64::INFINITY, f64::min),
        eth_klines.iter().map(|k| k.close).fold(f64::NEG_INFINITY, f64::max),
    );

    // === Strategy 1: Meta-Gradient Optimized ===
    println!("\n--- Strategy 1: Meta-Gradient Optimized ---");
    let model_meta = TradingModel::new(11, 32, 1);
    let optimizer_meta = MetaGradientOptimizer::new(
        model_meta,
        0.01,   // initial inner LR
        0.001,  // meta LR
        3,      // inner steps
        false,  // per-param LR
        true,   // learn loss
    );
    let feature_gen = FeatureGenerator::new(20);

    let mut engine_meta = BacktestEngine::new(
        optimizer_meta,
        feature_gen,
        0.001,  // prediction threshold
        0.001,  // transaction cost
        25,     // adaptation window
        5,      // validation window
    );

    println!("Running backtest on BTC...");
    let results_meta = engine_meta.run(&btc_klines, 10000.0);
    if !results_meta.is_empty() {
        let metrics_meta = BacktestEngine::calculate_metrics(&results_meta);
        println!("\nMeta-Gradient Strategy (BTC):");
        print_metrics("Meta-Gradient", &metrics_meta);

        // Show LR evolution
        println!("\n  Learning rate evolution:");
        let step = results_meta.len() / 5;
        for i in (0..results_meta.len()).step_by(step.max(1)) {
            println!(
                "    Step {:4}: LR={:.6}, Capital={:.2}",
                i, results_meta[i].learning_rate, results_meta[i].capital,
            );
        }
    }

    // === Strategy 2: Fixed LR Baseline ===
    println!("\n--- Strategy 2: Fixed LR Baseline ---");
    let model_fixed = TradingModel::new(11, 32, 1);
    let optimizer_fixed = MetaGradientOptimizer::new(
        model_fixed,
        0.01,   // fixed inner LR
        0.0,    // zero meta LR = no meta-learning
        3,      // inner steps
        false,
        false,
    );
    let feature_gen_fixed = FeatureGenerator::new(20);

    let mut engine_fixed = BacktestEngine::new(
        optimizer_fixed,
        feature_gen_fixed,
        0.001,
        0.001,
        25,
        5,
    );

    println!("Running backtest on BTC...");
    let results_fixed = engine_fixed.run(&btc_klines, 10000.0);
    if !results_fixed.is_empty() {
        let metrics_fixed = BacktestEngine::calculate_metrics(&results_fixed);
        println!("\nFixed LR Strategy (BTC):");
        print_metrics("Fixed LR", &metrics_fixed);
    }

    // === Compare on ETH ===
    println!("\n--- Comparison on ETH ---");

    let model_meta_eth = TradingModel::new(11, 32, 1);
    let optimizer_meta_eth = MetaGradientOptimizer::new(
        model_meta_eth, 0.01, 0.001, 3, false, true,
    );
    let mut engine_meta_eth = BacktestEngine::new(
        optimizer_meta_eth,
        FeatureGenerator::new(20),
        0.001, 0.001, 25, 5,
    );

    let results_meta_eth = engine_meta_eth.run(&eth_klines, 10000.0);
    if !results_meta_eth.is_empty() {
        let metrics_eth = BacktestEngine::calculate_metrics(&results_meta_eth);
        println!("\nMeta-Gradient Strategy (ETH):");
        print_metrics("Meta-Gradient ETH", &metrics_eth);
    }

    println!("\n=== Complete ===");
}
