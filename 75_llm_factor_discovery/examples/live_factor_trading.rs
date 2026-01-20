//! Live factor trading example
//!
//! Demonstrates how to use factors for live trading with risk management.
//!
//! Run with: cargo run --example live_factor_trading
//!
//! WARNING: This is a demonstration only. Do not use for actual trading
//! without proper testing and risk management.

use llm_factor_discovery::{
    data::{BybitClient, BybitConfig, DataBuffer, TimeFrame},
    evaluation::IcCalculator,
    parser::FactorExpr,
    strategy::{
        ExecutionConfig, FactorCombiner, FactorSignal, FactorStrategy,
        SignalGenerator, TradeAction,
        execution::{Position, Side, TradeExecutor},
    },
};
use std::collections::HashMap;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== Live Factor Trading Example ===");
    info!("WARNING: This is a demonstration only!\n");

    // Define our trading strategy
    let mut strategy = FactorStrategy::new("Multi-Factor Crypto Strategy");

    // Add factors with weights
    strategy.add_factor("rank(returns(close, 4))", 1.0);      // Momentum
    strategy.add_factor("sub(0, ts_rank(close, 24))", 0.5);   // Mean reversion
    strategy.add_factor("div(volume, ts_mean(volume, 12))", 0.3); // Volume

    strategy.set_thresholds(0.3, 0.5); // min_strength, min_confidence

    info!("Strategy: {}", strategy.name);
    info!("Factors:");
    for (expr, weight) in strategy.factors.iter().zip(strategy.weights.iter()) {
        info!("  {} (weight: {:.1})", expr, weight);
    }
    info!("");

    // Execution configuration
    let exec_config = ExecutionConfig {
        max_position_size: 0.05,      // 5% of capital
        min_position_size: 0.01,      // 1% minimum
        stop_loss_pct: 0.02,          // 2% stop loss
        take_profit_pct: 0.04,        // 4% take profit
        max_holding_hours: 24,        // 24 hour max hold
        use_limit_orders: true,
        limit_offset_bps: 5.0,        // 5 bps from mid
        min_adjustment_threshold: 0.2,
    };

    info!("Execution Config:");
    info!("  Max Position: {:.1}%", exec_config.max_position_size * 100.0);
    info!("  Stop Loss: {:.1}%", exec_config.stop_loss_pct * 100.0);
    info!("  Take Profit: {:.1}%", exec_config.take_profit_pct * 100.0);
    info!("  Max Hold: {} hours", exec_config.max_holding_hours);
    info!("");

    // Initialize components
    let executor = TradeExecutor::new(exec_config);
    let signal_generator = SignalGenerator::new();
    let factor_combiner = FactorCombiner::equal_weight();

    // Universe
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    // Track positions
    let mut positions: HashMap<String, Position> = HashMap::new();
    for symbol in &symbols {
        positions.insert(symbol.to_string(), Position::default());
    }

    // Simulate a few trading iterations
    info!("=== Simulated Trading Loop ===\n");

    for iteration in 1..=3 {
        info!("--- Iteration {} ---", iteration);

        for symbol in &symbols {
            info!("\nProcessing {}...", symbol);

            // In real trading, you would fetch live data
            // Here we simulate factor signals
            let factor_signals = simulate_factor_signals(symbol, iteration);

            info!("  Factor Signals:");
            for sig in &factor_signals {
                info!("    {}: z={:.2}", sig.factor_name, sig.z_score);
            }

            // Generate trading signal
            let weights = strategy.weights.clone();
            let trading_signal = signal_generator.generate(symbol, &factor_signals, &weights);

            let position = positions.get_mut(*symbol).unwrap();

            match trading_signal {
                Some(signal) => {
                    info!("  Trading Signal:");
                    info!("    Direction: {:.2}", signal.direction);
                    info!("    Strength: {:.2}", signal.strength);
                    info!("    Confidence: {:.2}", signal.confidence);

                    if strategy.should_trade(&signal) {
                        // Simulated current price
                        let current_price = simulate_price(symbol, iteration);

                        // Determine action
                        let action = executor.determine_action(&signal, position, current_price);

                        match action {
                            TradeAction::Open { side, size, limit_price, .. } => {
                                info!("  Action: OPEN {:?}", side);
                                info!("    Size: {:.2}%", size * 100.0);
                                if let Some(limit) = limit_price {
                                    info!("    Limit Price: ${:.2}", limit);
                                }

                                // Update position (simulated)
                                position.symbol = symbol.to_string();
                                position.side = Some(side);
                                position.size = size;
                                position.entry_price = current_price;
                                position.entry_time = Some(chrono::Utc::now());
                            }
                            TradeAction::Close { reason, .. } => {
                                info!("  Action: CLOSE (reason: {:?})", reason);

                                // Reset position
                                *position = Position::default();
                            }
                            TradeAction::Adjust { new_size, .. } => {
                                info!("  Action: ADJUST to {:.2}%", new_size * 100.0);
                                position.size = new_size;
                            }
                            TradeAction::Hold => {
                                info!("  Action: HOLD");
                            }
                        }
                    } else {
                        info!("  Signal below thresholds, skipping");
                    }
                }
                None => {
                    info!("  No signal generated (factors not significant)");
                }
            }

            // Update position P&L
            if position.is_open() {
                let current_price = simulate_price(symbol, iteration);
                position.update_pnl(current_price);
                info!("  Current Position: {:?} {:.2}%, PnL: {:.2}%",
                    position.side.unwrap(),
                    position.size * 100.0,
                    position.unrealized_pnl * 100.0);
            }
        }

        // Brief pause between iterations
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    // Final portfolio summary
    info!("\n=== Portfolio Summary ===\n");

    for (symbol, position) in &positions {
        if position.is_open() {
            info!("{}: {:?} {:.2}%, Unrealized PnL: {:.2}%",
                symbol,
                position.side.unwrap(),
                position.size * 100.0,
                position.unrealized_pnl * 100.0);
        } else {
            info!("{}: No position", symbol);
        }
    }

    // Risk checks
    info!("\n=== Risk Management ===\n");

    let total_exposure: f64 = positions.values()
        .filter(|p| p.is_open())
        .map(|p| p.size)
        .sum();

    info!("Total Exposure: {:.2}%", total_exposure * 100.0);

    if total_exposure > 0.15 {
        warn!("WARNING: Total exposure exceeds 15% limit!");
    } else {
        info!("Exposure within limits");
    }

    let max_loss: f64 = positions.values()
        .filter(|p| p.is_open() && p.unrealized_pnl < 0.0)
        .map(|p| p.unrealized_pnl.abs())
        .fold(0.0, f64::max);

    if max_loss > 0.01 {
        info!("Max Single Loss: {:.2}%", max_loss * 100.0);
    }

    info!("\n=== Live Trading Example Complete ===");
    info!("Remember: Always test strategies thoroughly before live trading!");

    Ok(())
}

/// Simulate factor signals for demonstration
fn simulate_factor_signals(symbol: &str, iteration: i32) -> Vec<FactorSignal> {
    // Create different signals based on symbol and iteration
    let base_momentum = match symbol {
        "BTCUSDT" => 1.5 - iteration as f64 * 0.3,
        "ETHUSDT" => 0.8 + iteration as f64 * 0.2,
        "SOLUSDT" => -0.5 + iteration as f64 * 0.8,
        _ => 0.0,
    };

    let base_reversion = -base_momentum * 0.3;
    let base_volume = 1.0 + (iteration as f64 * 0.5).sin();

    vec![
        FactorSignal::new("momentum".to_string(), base_momentum, 0.0, 1.0),
        FactorSignal::new("mean_reversion".to_string(), base_reversion, 0.0, 1.0),
        FactorSignal::new("volume".to_string(), base_volume, 1.0, 0.5),
    ]
}

/// Simulate price for demonstration
fn simulate_price(symbol: &str, iteration: i32) -> f64 {
    match symbol {
        "BTCUSDT" => 65000.0 + iteration as f64 * 500.0,
        "ETHUSDT" => 3500.0 + iteration as f64 * 30.0,
        "SOLUSDT" => 150.0 + iteration as f64 * 5.0,
        _ => 100.0,
    }
}
