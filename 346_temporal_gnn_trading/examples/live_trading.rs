//! Live Trading Demo with Bybit
//!
//! This example demonstrates how to use TGN with real-time
//! market data from Bybit exchange.

use temporal_gnn_trading::prelude::*;
use temporal_gnn_trading::{TemporalGNN, TGNConfig, MarketEvent};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Temporal GNN Trading - Live Trading Demo ===\n");

    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Trading symbols to monitor
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOGEUSDT"];

    println!("Monitoring symbols: {:?}\n", symbols);

    // Create Bybit client
    let bybit_config = BybitConfig::default();
    let client = BybitClient::new(bybit_config)?;

    // Create TGN model
    let tgn_config = TGNConfig {
        memory_dim: 128,
        time_dim: 64,
        message_dim: 128,
        embedding_dim: 64,
        num_heads: 4,
        memory_decay: 0.001,
        use_time_encoding: true,
        num_neighbors: 10,
        learning_rate: 0.001,
        dropout: 0.1,
    };

    let mut tgn = TemporalGNN::new(tgn_config);

    // Create trading strategy
    let strategy_config = StrategyConfig {
        min_confidence: 0.65,
        max_position_size: 0.05,
        max_exposure: 0.15,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.04,
        max_leverage: 3.0,
        cooldown_ms: 30000, // 30 seconds
    };

    let mut strategy = TradingStrategy::new(strategy_config);

    // Performance tracker
    let mut tracker = PerformanceTracker::new(10000.0, 100);

    println!("Fetching initial market data...\n");

    // Fetch initial data
    let tickers: HashMap<String, Ticker> = client
        .get_tickers(&symbols.iter().map(|s| *s).collect::<Vec<_>>())
        .await?;

    println!("Current prices:");
    for (symbol, ticker) in &tickers {
        println!("  {}: ${:.2} (24h: {:.2}%)",
            symbol,
            ticker.last_price,
            ticker.price_change_24h
        );
    }
    println!();

    // Simulate live trading loop (in production, use WebSocket)
    println!("Starting trading simulation...\n");
    println!("{:-<80}", "");

    let mut iteration = 0;
    let max_iterations = 20; // Limit for demo

    while iteration < max_iterations {
        iteration += 1;

        // Fetch current tickers
        let current_tickers = match client
            .get_tickers(&symbols.iter().map(|s| *s).collect::<Vec<_>>())
            .await
        {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error fetching tickers: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                continue;
            }
        };

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Process each symbol
        for symbol in &symbols {
            if let Some(ticker) = current_tickers.get(*symbol) {
                // Create market event from ticker update
                let event = create_event_from_ticker(ticker, timestamp);

                // Process through TGN
                if let Some(signal) = tgn.process_event(&event) {
                    // Get current price for strategy
                    let current_price = ticker.last_price;

                    // Process signal through strategy
                    if let Some(action) = strategy.process_signal(&signal, current_price) {
                        execute_action(&mut strategy, &action, current_price, timestamp, &mut tracker);
                    }

                    // Print signal if strong enough
                    if signal.confidence > 0.5 {
                        println!(
                            "[{}] {} - {:?} (strength: {:.2}, confidence: {:.2})",
                            format_timestamp(timestamp),
                            symbol,
                            signal.signal_type,
                            signal.strength,
                            signal.confidence
                        );
                    }
                }
            }
        }

        // Update positions with current prices
        let price_map: HashMap<String, f64> = current_tickers
            .iter()
            .map(|(k, v)| (k.clone(), v.last_price))
            .collect();
        strategy.update_positions(&price_map);

        // Print status every 5 iterations
        if iteration % 5 == 0 {
            print_status(&tgn, &strategy, &tracker);
        }

        // Wait before next iteration (simulating real-time delay)
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    println!("\n{:-<80}", "");
    println!("\n=== Final Performance Summary ===\n");

    // Final metrics
    let metrics = tracker.get_metrics(0.02, 252.0 * 24.0 * 60.0); // Assuming minute data
    let trade_metrics = strategy.calculate_metrics();

    println!("Portfolio Performance:");
    println!("  Total Return: {:.2}%", metrics.total_return * 100.0);
    println!("  Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    println!("  Current Drawdown: {:.2}%", metrics.current_drawdown * 100.0);

    println!("\nTrading Statistics:");
    println!("  Total Trades: {}", trade_metrics.total_trades);
    println!("  Win Rate: {:.2}%", trade_metrics.win_rate * 100.0);
    println!("  Profit Factor: {:.2}", trade_metrics.profit_factor);
    println!("  Total PnL: ${:.2}", trade_metrics.total_pnl);

    println!("\nModel Statistics:");
    let stats = tgn.stats();
    println!("  Nodes in Graph: {}", stats.num_nodes);
    println!("  Events Processed: {}", stats.total_events);

    println!("\n=== Demo Complete ===");

    Ok(())
}

/// Create a market event from a ticker update
fn create_event_from_ticker(ticker: &Ticker, timestamp: u64) -> MarketEvent {
    MarketEvent {
        source: "ticker_update".to_string(),
        target: ticker.symbol.clone(),
        timestamp,
        features: EventFeatures {
            price: ticker.last_price,
            volume: ticker.volume_24h / 1440.0, // Approximate minute volume
            side: if ticker.price_change_24h > 0.0 { 1.0 } else { -1.0 },
            price_change: ticker.price_change_24h / 100.0,
            volume_ratio: 1.0,
            imbalance: if ticker.bid_price > 0.0 && ticker.ask_price > 0.0 {
                (ticker.bid_price - ticker.mid_price()) / ticker.spread()
            } else {
                0.0
            },
            spread: ticker.spread(),
            volatility: (ticker.high_24h - ticker.low_24h) / ticker.last_price,
        },
    }
}

/// Execute a trade action
fn execute_action(
    strategy: &mut TradingStrategy,
    action: &TradeAction,
    current_price: f64,
    timestamp: u64,
    tracker: &mut PerformanceTracker,
) {
    match action {
        TradeAction::Open { symbol, side, size } => {
            println!(
                "  [OPEN] {:?} {} @ ${:.2} (size: {:.4})",
                side, symbol, current_price, size
            );
            strategy.open_position(symbol, *side, *size, current_price, timestamp);
        }
        TradeAction::Close { symbol, reason } => {
            if let Some(record) = strategy.close_position(symbol, current_price, timestamp, reason) {
                println!(
                    "  [CLOSE] {} @ ${:.2} (PnL: ${:.2}, reason: {})",
                    symbol, current_price, record.pnl, reason
                );
                tracker.update(record.pnl);
            }
        }
        TradeAction::Reverse { symbol, new_side, size } => {
            // Close existing position first
            if let Some(record) = strategy.close_position(symbol, current_price, timestamp, "reversal") {
                println!(
                    "  [CLOSE for REVERSAL] {} @ ${:.2} (PnL: ${:.2})",
                    symbol, current_price, record.pnl
                );
                tracker.update(record.pnl);
            }
            // Open new position
            println!(
                "  [OPEN] {:?} {} @ ${:.2} (size: {:.4})",
                new_side, symbol, current_price, size
            );
            strategy.open_position(symbol, *new_side, *size, current_price, timestamp);
        }
    }
}

/// Print current status
fn print_status(tgn: &TemporalGNN, strategy: &TradingStrategy, tracker: &PerformanceTracker) {
    let stats = tgn.stats();
    let positions = strategy.get_all_positions();

    println!("\n--- Status Update ---");
    println!("  Events processed: {}", stats.total_events);
    println!("  Current equity: ${:.2}", tracker.current_equity());
    println!("  Open positions: {}", positions.len());

    for (symbol, pos) in positions {
        let side = if pos.side > 0 { "LONG" } else { "SHORT" };
        println!(
            "    {} {} @ ${:.2} (PnL: ${:.2})",
            symbol, side, pos.entry_price, pos.unrealized_pnl
        );
    }
    println!();
}

/// Format timestamp for display
fn format_timestamp(timestamp: u64) -> String {
    let secs = timestamp / 1000;
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let secs = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}
