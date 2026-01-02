//! Backtesting Example for Temporal GNN Trading
//!
//! This example demonstrates how to backtest a TGN-based trading strategy
//! using historical data.

use temporal_gnn_trading::prelude::*;
use temporal_gnn_trading::{TemporalGNN, TGNConfig, MarketEvent};
use std::collections::HashMap;

fn main() {
    println!("=== Temporal GNN Trading - Backtest Example ===\n");

    // Generate synthetic historical data
    let historical_data = generate_historical_data();

    println!("Generated {} historical data points\n", historical_data.len());

    // Create TGN model
    let tgn_config = TGNConfig {
        memory_dim: 64,
        time_dim: 32,
        message_dim: 64,
        embedding_dim: 32,
        num_heads: 4,
        memory_decay: 0.005,
        ..Default::default()
    };

    let mut tgn = TemporalGNN::new(tgn_config);

    // Create trading strategy with conservative parameters
    let strategy_config = StrategyConfig {
        min_confidence: 0.55,
        max_position_size: 0.1,
        max_exposure: 0.3,
        stop_loss_pct: 0.015,
        take_profit_pct: 0.03,
        max_leverage: 2.0,
        cooldown_ms: 60000, // 1 minute
    };

    let mut strategy = TradingStrategy::new(strategy_config);

    // Performance tracker
    let initial_capital = 10000.0;
    let mut tracker = PerformanceTracker::new(initial_capital, 100);

    // Backtest loop
    println!("Running backtest...\n");
    println!("{:-<80}", "");

    let mut total_signals = 0;
    let mut actionable_signals = 0;

    for (i, data_point) in historical_data.iter().enumerate() {
        // Create events from data point
        let events = create_events_from_data(data_point);

        // Process each event
        for event in &events {
            if let Some(signal) = tgn.process_event(event) {
                total_signals += 1;

                // Get current price
                let current_price = data_point.prices.get(&event.target).copied().unwrap_or(0.0);

                if current_price > 0.0 {
                    // Process signal through strategy
                    if let Some(action) = strategy.process_signal(&signal, current_price) {
                        actionable_signals += 1;
                        execute_backtest_action(
                            &mut strategy,
                            &action,
                            current_price,
                            data_point.timestamp,
                            &mut tracker,
                        );
                    }
                }
            }
        }

        // Update positions with current prices
        strategy.update_positions(&data_point.prices);

        // Check for stop losses and take profits
        check_exits(&mut strategy, &data_point.prices, data_point.timestamp, &mut tracker);

        // Print progress every 100 data points
        if (i + 1) % 100 == 0 {
            let current_equity = tracker.current_equity();
            let return_pct = (current_equity - initial_capital) / initial_capital * 100.0;
            println!(
                "Progress: {}/{} | Equity: ${:.2} ({:+.2}%) | Positions: {}",
                i + 1,
                historical_data.len(),
                current_equity,
                return_pct,
                strategy.get_all_positions().len()
            );
        }
    }

    // Close all remaining positions at the end
    let final_prices = &historical_data.last().unwrap().prices;
    let final_timestamp = historical_data.last().unwrap().timestamp;
    close_all_positions(&mut strategy, final_prices, final_timestamp, &mut tracker);

    println!("{:-<80}", "");
    println!("\n=== Backtest Results ===\n");

    // Performance metrics
    let metrics = tracker.get_metrics(0.02, 365.0 * 24.0); // Hourly data, annualized
    let trade_metrics = strategy.calculate_metrics();

    println!("Portfolio Performance:");
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Final Equity: ${:.2}", tracker.current_equity());
    println!("  Total Return: {:.2}%", metrics.total_return * 100.0);
    println!("  Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
    println!("  Sortino Ratio: {:.3}", metrics.sortino_ratio);
    println!("  Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    println!("  Calmar Ratio: {:.3}", metrics.calmar_ratio);

    println!("\nTrading Statistics:");
    println!("  Total Signals: {}", total_signals);
    println!("  Actionable Signals: {}", actionable_signals);
    println!("  Total Trades: {}", trade_metrics.total_trades);
    println!("  Winning Trades: {}", trade_metrics.winning_trades);
    println!("  Losing Trades: {}", trade_metrics.losing_trades);
    println!("  Win Rate: {:.2}%", trade_metrics.win_rate * 100.0);
    println!("  Average Win: ${:.2}", trade_metrics.avg_win);
    println!("  Average Loss: ${:.2}", trade_metrics.avg_loss);
    println!("  Profit Factor: {:.2}", trade_metrics.profit_factor);
    println!("  Total PnL: ${:.2}", trade_metrics.total_pnl);

    // Trade history summary
    let history = strategy.get_trade_history();
    if !history.is_empty() {
        println!("\nRecent Trades (last 10):");
        for trade in history.iter().rev().take(10) {
            let side = if trade.side > 0 { "LONG" } else { "SHORT" };
            println!(
                "  {} {} Entry: ${:.2} Exit: ${:.2} PnL: ${:+.2} ({:+.2}%) - {}",
                trade.symbol,
                side,
                trade.entry_price,
                trade.exit_price,
                trade.pnl,
                trade.return_pct * 100.0,
                trade.exit_reason
            );
        }
    }

    // Model statistics
    println!("\nModel Statistics:");
    let stats = tgn.stats();
    println!("  Nodes in Graph: {}", stats.num_nodes);
    println!("  Events Processed: {}", stats.total_events);

    // Equity curve analysis
    let equity_curve = tracker.equity_curve();
    println!("\nEquity Curve Summary:");
    println!("  Data Points: {}", equity_curve.len());
    println!("  Peak Equity: ${:.2}", equity_curve.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!("  Min Equity: ${:.2}", equity_curve.iter().cloned().fold(f64::INFINITY, f64::min));

    println!("\n=== Backtest Complete ===");
}

/// Data point for backtesting
struct DataPoint {
    timestamp: u64,
    prices: HashMap<String, f64>,
    volumes: HashMap<String, f64>,
    returns: HashMap<String, f64>,
}

/// Generate synthetic historical data
fn generate_historical_data() -> Vec<DataPoint> {
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let mut data = Vec::new();

    // Initial prices
    let mut prices: HashMap<String, f64> = HashMap::new();
    prices.insert("BTCUSDT".to_string(), 40000.0);
    prices.insert("ETHUSDT".to_string(), 2000.0);
    prices.insert("SOLUSDT".to_string(), 80.0);

    let base_timestamp = 1704067200000u64; // 2024-01-01
    let num_periods = 500; // ~500 hours of data

    for i in 0..num_periods {
        let timestamp = base_timestamp + i as u64 * 3600000; // Hourly data

        let mut returns: HashMap<String, f64> = HashMap::new();
        let mut volumes: HashMap<String, f64> = HashMap::new();

        // Generate correlated returns with some randomness
        let market_return = (rand::random::<f64>() - 0.5) * 0.04; // Market factor

        for symbol in &symbols {
            // Asset-specific return with market correlation
            let beta = match symbol.as_str() {
                "BTCUSDT" => 1.0,
                "ETHUSDT" => 1.2,
                "SOLUSDT" => 1.5,
                _ => 1.0,
            };

            let idiosyncratic = (rand::random::<f64>() - 0.5) * 0.02;
            let ret = market_return * beta + idiosyncratic;

            // Add some trend momentum
            let momentum = if i > 0 && i % 24 == 0 {
                (rand::random::<f64>() - 0.5) * 0.02
            } else {
                0.0
            };

            let total_return = ret + momentum;
            returns.insert(symbol.to_string(), total_return);

            // Update price
            let current_price = prices.get_mut(&symbol.to_string()).unwrap();
            *current_price *= 1.0 + total_return;

            // Generate volume (higher when price moves more)
            let base_volume = match symbol.as_str() {
                "BTCUSDT" => 1000.0,
                "ETHUSDT" => 5000.0,
                "SOLUSDT" => 50000.0,
                _ => 1000.0,
            };
            let volume = base_volume * (1.0 + total_return.abs() * 10.0) * (0.5 + rand::random::<f64>());
            volumes.insert(symbol.to_string(), volume);
        }

        data.push(DataPoint {
            timestamp,
            prices: prices.clone(),
            volumes,
            returns,
        });
    }

    data
}

/// Create events from a data point
fn create_events_from_data(data: &DataPoint) -> Vec<MarketEvent> {
    let mut events = Vec::new();

    for (symbol, &price) in &data.prices {
        let volume = data.volumes.get(symbol).copied().unwrap_or(0.0);
        let ret = data.returns.get(symbol).copied().unwrap_or(0.0);

        events.push(MarketEvent {
            source: "historical".to_string(),
            target: symbol.clone(),
            timestamp: data.timestamp,
            features: EventFeatures {
                price,
                volume,
                side: if ret > 0.0 { 1.0 } else { -1.0 },
                price_change: ret,
                volume_ratio: 1.0 + (rand::random::<f64>() - 0.5) * 0.5,
                imbalance: (rand::random::<f64>() - 0.5) * 0.6,
                spread: price * 0.0001,
                volatility: ret.abs() * 2.0,
            },
        });
    }

    // Add some cross-asset events
    if rand::random::<f64>() > 0.7 {
        events.push(MarketEvent {
            source: "BTCUSDT".to_string(),
            target: "ETHUSDT".to_string(),
            timestamp: data.timestamp,
            features: EventFeatures {
                price: data.prices.get("BTCUSDT").copied().unwrap_or(0.0),
                volume: data.volumes.get("BTCUSDT").copied().unwrap_or(0.0) * 0.1,
                side: if data.returns.get("BTCUSDT").copied().unwrap_or(0.0) > 0.0 { 1.0 } else { -1.0 },
                price_change: data.returns.get("BTCUSDT").copied().unwrap_or(0.0),
                volume_ratio: 1.0,
                imbalance: 0.0,
                spread: 0.0,
                volatility: 0.02,
            },
        });
    }

    events
}

/// Execute a backtest trade action
fn execute_backtest_action(
    strategy: &mut TradingStrategy,
    action: &TradeAction,
    current_price: f64,
    timestamp: u64,
    tracker: &mut PerformanceTracker,
) {
    match action {
        TradeAction::Open { symbol, side, size } => {
            strategy.open_position(symbol, *side, *size, current_price, timestamp);
        }
        TradeAction::Close { symbol, reason } => {
            if let Some(record) = strategy.close_position(symbol, current_price, timestamp, reason) {
                tracker.update(record.pnl);
            }
        }
        TradeAction::Reverse { symbol, new_side, size } => {
            if let Some(record) = strategy.close_position(symbol, current_price, timestamp, "reversal") {
                tracker.update(record.pnl);
            }
            strategy.open_position(symbol, *new_side, *size, current_price, timestamp);
        }
    }
}

/// Check for stop losses and take profits
fn check_exits(
    strategy: &mut TradingStrategy,
    prices: &HashMap<String, f64>,
    timestamp: u64,
    tracker: &mut PerformanceTracker,
) {
    let positions: Vec<_> = strategy.get_all_positions().keys().cloned().collect();

    for symbol in positions {
        if let Some(&current_price) = prices.get(&symbol) {
            if let Some(position) = strategy.get_position(&symbol) {
                if position.is_stop_loss_hit(current_price) {
                    if let Some(record) = strategy.close_position(&symbol, current_price, timestamp, "stop_loss") {
                        tracker.update(record.pnl);
                    }
                } else if position.is_take_profit_hit(current_price) {
                    if let Some(record) = strategy.close_position(&symbol, current_price, timestamp, "take_profit") {
                        tracker.update(record.pnl);
                    }
                }
            }
        }
    }
}

/// Close all remaining positions
fn close_all_positions(
    strategy: &mut TradingStrategy,
    prices: &HashMap<String, f64>,
    timestamp: u64,
    tracker: &mut PerformanceTracker,
) {
    let positions: Vec<_> = strategy.get_all_positions().keys().cloned().collect();

    for symbol in positions {
        if let Some(&price) = prices.get(&symbol) {
            if let Some(record) = strategy.close_position(&symbol, price, timestamp, "backtest_end") {
                tracker.update(record.pnl);
            }
        }
    }
}
