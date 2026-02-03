//! TE-based trading strategy backtest example.
//!
//! Demonstrates the full pipeline: generate data, compute TE network,
//! identify leader-follower pairs, generate signals, and backtest.

use te_trading::prelude::*;

fn main() {
    println!("=== TE Trading Strategy Backtest ===\n");

    // Phase 1: Generate training data
    println!("Phase 1: Data Generation");
    let mut gen = SimulatedDataGenerator::new(42);
    let (symbols, returns) = gen.generate_crypto_returns(1000);

    println!("  Generated {} periods for {} assets", returns[0].len(), symbols.len());

    // Phase 2: Build TE network and identify pairs
    println!("\nPhase 2: TE Network Analysis");
    let return_refs: Vec<&[f64]> = returns.iter().map(|r| r.as_slice()).collect();
    let mut network = TENetwork::new(symbols.clone(), 3);
    network.compute_all_pairs(&return_refs);

    let (leaders, followers) = network.identify_leaders_followers();
    println!("  Leaders:   {:?}", leaders);
    println!("  Followers: {:?}", followers);

    let leader_idx = symbols.iter().position(|s| s == &leaders[0]).unwrap();
    let follower_idx = symbols.iter().position(|s| s == &followers[followers.len() - 1]).unwrap();
    println!(
        "  Trading pair: {} (leader) -> {} (follower)",
        symbols[leader_idx], symbols[follower_idx]
    );

    let te_value = network.te_matrix[leader_idx][follower_idx];
    println!("  TE({} -> {}) = {:.4} bits", symbols[leader_idx], symbols[follower_idx], te_value);

    // Phase 3: Generate trading signals
    println!("\nPhase 3: Signal Generation");
    let strategy = TEStrategy::new(0.005, 0.01, 60);
    let signals = strategy.generate_signals(&returns[leader_idx], &returns[follower_idx]);

    let long_count = signals.iter().filter(|s| **s == TradingSignal::Long).count();
    let short_count = signals.iter().filter(|s| **s == TradingSignal::Short).count();
    let neutral_count = signals.iter().filter(|s| **s == TradingSignal::Neutral).count();
    println!("  Long signals:    {}", long_count);
    println!("  Short signals:   {}", short_count);
    println!("  Neutral signals: {}", neutral_count);

    // Phase 4: Backtest
    println!("\nPhase 4: Backtesting");

    // Generate follower prices from returns
    let mut follower_prices = vec![100.0f64; returns[follower_idx].len()];
    for t in 1..follower_prices.len() {
        follower_prices[t] = follower_prices[t - 1] * (1.0 + returns[follower_idx][t]);
    }

    let engine = BacktestEngine::new(100_000.0, 0.001);
    let steps = engine.run(&signals, &returns[follower_idx], &follower_prices);
    let metrics = engine.compute_metrics(&steps);

    println!("\nTE Strategy Results:");
    println!("{}", metrics);

    // Benchmark: Buy and hold the follower
    let buy_hold_signals = vec![TradingSignal::Long; returns[follower_idx].len()];
    let bh_engine = BacktestEngine::new(100_000.0, 0.001);
    let bh_steps = bh_engine.run(&buy_hold_signals, &returns[follower_idx], &follower_prices);
    let bh_metrics = bh_engine.compute_metrics(&bh_steps);

    println!("Benchmark (Buy & Hold {}):", symbols[follower_idx]);
    println!("{}", bh_metrics);

    // Capital curve summary
    println!("Capital Curve (every 100 steps):");
    println!("  {:>6}  {:>12}  {:>12}  {:>8}", "Step", "TE Strategy", "Buy & Hold", "Signal");
    for t in (0..steps.len()).step_by(100) {
        let signal = match steps[t].signal {
            TradingSignal::Long => "Long",
            TradingSignal::Short => "Short",
            TradingSignal::Neutral => "Neutral",
        };
        println!(
            "  {:>6}  {:>12.2}  {:>12.2}  {:>8}",
            t, steps[t].capital, bh_steps[t].capital, signal
        );
    }

    println!("\nDone!");
}
