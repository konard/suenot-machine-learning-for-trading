//! Backtesting Example
//!
//! This example demonstrates:
//! 1. Loading historical data
//! 2. Running the strategy in simulation
//! 3. Calculating performance metrics
//!
//! Run with: cargo run --example backtest

use dynamic_gnn_trading::graph::{DynamicGraph, GraphConfig, NodeFeatures, EdgeFeatures};
use dynamic_gnn_trading::gnn::{DynamicGNN, GNNConfig};
use dynamic_gnn_trading::strategy::{TradingStrategy, StrategyConfig, OrderSide};
use dynamic_gnn_trading::utils::{PerformanceTracker, TradeRecord};
use std::collections::HashMap;

fn main() {
    println!("=== Dynamic GNN Backtesting Example ===\n");

    // Configuration
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let initial_capital = 10000.0;
    let num_periods = 100;

    // Generate synthetic price data
    println!("Step 1: Generating synthetic market data...");
    let price_data = generate_synthetic_data(&symbols, num_periods);

    for (symbol, prices) in &price_data {
        let first = prices.first().unwrap();
        let last = prices.last().unwrap();
        let change = (last - first) / first * 100.0;
        println!("  {}: ${:.2} -> ${:.2} ({:+.2}%)", symbol, first, last, change);
    }

    // Initialize components
    println!("\nStep 2: Initializing components...");

    let graph_config = GraphConfig::default();
    let mut graph = DynamicGraph::with_config(graph_config);

    let gnn_config = GNNConfig {
        input_dim: 10,
        hidden_dims: vec![32, 16],
        output_dim: 8,
        num_heads: 2,
        ..Default::default()
    };
    let mut model = DynamicGNN::new(gnn_config);

    let strategy_config = StrategyConfig {
        min_confidence: 0.55,
        max_position_pct: 0.1,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.03,
        max_positions: 3,
        ..Default::default()
    };
    let mut strategy = TradingStrategy::new(strategy_config);

    let mut tracker = PerformanceTracker::new(initial_capital);

    println!("  Capital: ${:.2}", initial_capital);
    println!("  Model params: {}", model.param_count());

    // Run backtest
    println!("\nStep 3: Running backtest...");

    let mut trades_executed = 0;

    for t in 0..num_periods {
        let timestamp = (t as u64) * 60000; // 1 minute intervals

        // Update graph with current prices
        for symbol in &symbols {
            let price = price_data[*symbol][t];
            let features = NodeFeatures::new(price, 100000.0, timestamp);

            if t == 0 {
                graph.add_node(*symbol, features);
            } else {
                graph.update_node(*symbol, features);
            }
        }

        // Add edges if needed
        if t == 0 {
            graph.add_edge("BTCUSDT", "ETHUSDT", EdgeFeatures::with_correlation(0.8, timestamp));
            graph.add_edge("ETHUSDT", "SOLUSDT", EdgeFeatures::with_correlation(0.7, timestamp));
        }

        graph.tick(timestamp);

        // Run GNN
        let (features, node_ids) = graph.feature_matrix();
        let (adjacency, _) = graph.adjacency_matrix();
        let output = model.forward(&features, &adjacency, None);

        // Generate signals and execute trades
        for (i, symbol) in node_ids.iter().enumerate() {
            let embedding = output.row(i).to_owned();
            let (p_down, p_neutral, p_up) = model.predict_direction(&embedding);
            let price = price_data[symbol.as_str()][t];

            // Process through strategy
            if let Some(order) = strategy.process_predictions(
                symbol,
                price,
                (p_down, p_neutral, p_up),
                0.7,
                timestamp,
            ) {
                // Simulate execution
                strategy.execute_order(&order, price, timestamp);
                trades_executed += 1;

                if t % 20 == 0 {
                    println!(
                        "  [t={}] {} {} @ ${:.2}",
                        t, order.side, symbol, price
                    );
                }
            }

            // Check exits
            if let Some(exit_order) = strategy.check_exits(symbol, price) {
                // Record trade
                if let Some(pos) = strategy.positions().iter().find(|p| p.symbol == *symbol) {
                    let trade = TradeRecord::new(
                        symbol.clone(),
                        match pos.side {
                            OrderSide::Buy => "BUY",
                            OrderSide::Sell => "SELL",
                        },
                        pos.entry_price,
                        price,
                        pos.size,
                        pos.entry_time,
                        timestamp,
                    );
                    tracker.record_trade(trade);
                }

                strategy.execute_order(&exit_order, price, timestamp);
            }
        }
    }

    // Close remaining positions
    println!("\nStep 4: Closing remaining positions...");
    let final_prices: HashMap<String, f64> = symbols
        .iter()
        .map(|s| (s.to_string(), *price_data[*s].last().unwrap()))
        .collect();

    for symbol in &symbols {
        if let Some(pos) = strategy.positions().iter().find(|p| p.symbol == *symbol) {
            let price = final_prices[*symbol];
            let trade = TradeRecord::new(
                symbol.to_string(),
                match pos.side {
                    OrderSide::Buy => "BUY",
                    OrderSide::Sell => "SELL",
                },
                pos.entry_price,
                price,
                pos.size,
                pos.entry_time,
                (num_periods as u64) * 60000,
            );
            tracker.record_trade(trade);
        }
    }

    // Calculate metrics
    println!("\nStep 5: Calculating performance metrics...");

    let metrics = tracker.metrics();

    println!("\n╔══════════════════════════════════════╗");
    println!("║         BACKTEST RESULTS             ║");
    println!("╠══════════════════════════════════════╣");
    println!("║ Total Trades:      {:>16} ║", metrics.total_trades);
    println!("║ Winning Trades:    {:>16} ║", metrics.winning_trades);
    println!("║ Losing Trades:     {:>16} ║", metrics.losing_trades);
    println!("║ Win Rate:          {:>15.1}% ║", metrics.win_rate * 100.0);
    println!("╠══════════════════════════════════════╣");
    println!("║ Total PnL:        ${:>15.2} ║", metrics.total_pnl);
    println!("║ Return:            {:>15.2}% ║", tracker.total_return() * 100.0);
    println!("║ Sharpe Ratio:      {:>16.2} ║", metrics.sharpe_ratio);
    println!("║ Sortino Ratio:     {:>16.2} ║", metrics.sortino_ratio);
    println!("║ Max Drawdown:      {:>15.1}% ║", metrics.max_drawdown * 100.0);
    println!("║ Profit Factor:     {:>16.2} ║", metrics.profit_factor);
    println!("╠══════════════════════════════════════╣");
    println!("║ Final Equity:     ${:>15.2} ║", tracker.current_equity());
    println!("╚══════════════════════════════════════╝");

    // Recent trades
    println!("\nRecent Trades:");
    for trade in tracker.recent_trades(5) {
        let emoji = if trade.is_winner() { "✓" } else { "✗" };
        println!(
            "  {} {} {}: ${:.2} -> ${:.2} (PnL: ${:.2})",
            emoji, trade.side, trade.symbol, trade.entry_price, trade.exit_price, trade.pnl
        );
    }

    println!("\n=== Backtest Complete ===");
}

/// Generate synthetic price data for testing
fn generate_synthetic_data(symbols: &[&str], periods: usize) -> HashMap<&str, Vec<f64>> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut data = HashMap::new();

    let initial_prices: HashMap<&str, f64> = [
        ("BTCUSDT", 50000.0),
        ("ETHUSDT", 3000.0),
        ("SOLUSDT", 100.0),
    ]
    .into_iter()
    .collect();

    for symbol in symbols {
        let mut prices = Vec::with_capacity(periods);
        let mut price = *initial_prices.get(symbol).unwrap_or(&100.0);

        for _ in 0..periods {
            // Random walk with slight upward bias
            let drift = 0.0001;
            let volatility = 0.02;
            let change = drift + volatility * rng.gen_range(-1.0..1.0);
            price *= 1.0 + change;
            prices.push(price);
        }

        data.insert(*symbol, prices);
    }

    data
}
