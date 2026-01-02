//! Backtesting Example with Heterogeneous GNN
//!
//! This example demonstrates how to:
//! 1. Run a historical backtest with HGNN signals
//! 2. Track performance metrics
//! 3. Analyze results by metapath type
//!
//! Run with: cargo run --example backtest

use heterogeneous_gnn_trading::prelude::*;
use heterogeneous_gnn_trading::gnn::HGNNConfig;
use heterogeneous_gnn_trading::graph::{HeterogeneousGraph, GraphSchema, NodeType, EdgeType};
use heterogeneous_gnn_trading::strategy::{TradingStrategy, StrategyConfig, SignalAggregator};
use heterogeneous_gnn_trading::utils::{Metrics, PerformanceTracker};
use std::collections::HashMap;
use rand::Rng;

fn main() {
    println!("=== Heterogeneous GNN Backtesting Demo ===\n");

    // Configuration
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "ARBUSDT"];
    let initial_capital = 100_000.0;
    let num_periods = 252;  // 1 year of daily data

    println!("Backtest Configuration:");
    println!("  Symbols: {:?}", symbols);
    println!("  Initial Capital: ${:.0}", initial_capital);
    println!("  Periods: {} days", num_periods);
    println!();

    // Step 1: Generate simulated historical data
    println!("Step 1: Generating simulated historical data...");
    let historical_data = generate_simulated_data(&symbols, num_periods);
    println!("  Generated {} periods of data for {} symbols", num_periods, symbols.len());

    // Step 2: Initialize model and strategy
    println!("\nStep 2: Initializing HGNN model and strategy...");

    let config = HGNNConfig::default();
    let model = HeterogeneousGNN::new(config);

    let strategy_config = StrategyConfig {
        max_position_size: 0.1,
        max_total_exposure: 0.8,
        max_leverage: 2.0,
        stop_loss_pct: 0.03,
        take_profit_pct: 0.06,
        min_signal_strength: 0.25,
        min_confidence: 0.4,
        symbols: symbols.iter().map(|s| s.to_string()).collect(),
    };

    let mut strategy = TradingStrategy::new(strategy_config, initial_capital);
    let mut tracker = PerformanceTracker::new(initial_capital, num_periods);
    let aggregator = SignalAggregator::new();

    println!("  Model parameters: {}", model.param_count());
    println!("  Strategy initialized");

    // Step 3: Run backtest
    println!("\nStep 3: Running backtest...");
    println!("  Progress: ");

    let mut returns = Vec::new();
    let mut trade_log: Vec<TradeRecord> = Vec::new();
    let progress_interval = num_periods / 20;

    for period in 0..num_periods {
        // Progress bar
        if period % progress_interval == 0 {
            print!(".");
        }

        // Build graph for this period
        let graph = build_period_graph(&symbols, &historical_data, period);

        // Get embeddings and signals
        let embeddings = model.get_embeddings(&graph);
        let mut period_signals = Vec::new();

        for symbol in &symbols {
            if let Some(embedding) = embeddings.get(*symbol) {
                let (p_down, p_neutral, p_up) = model.predict_direction(embedding);
                let magnitude = model.predict_magnitude(embedding);

                let signal_type = SignalType::from_probs(p_down, p_neutral, p_up);
                let strength = (p_up - p_down).abs();
                let confidence = 1.0 - p_neutral;

                let signal = Signal::new(*symbol, signal_type, strength)
                    .with_confidence(confidence)
                    .with_expected_return(magnitude)
                    .with_source("hgnn");

                period_signals.push(signal);
            }
        }

        // Update prices
        for symbol in &symbols {
            if let Some(price) = historical_data.get(*symbol)
                .and_then(|prices| prices.get(period))
            {
                strategy.update_price(symbol, *price);
            }
        }

        // Generate and execute orders
        let orders = strategy.generate_orders(&period_signals);

        for order in &orders {
            if let Some(&price) = historical_data.get(order.symbol.as_str())
                .and_then(|prices| prices.get(period))
            {
                // Simulate execution with slippage
                let slippage = 0.0005;  // 5 bps
                let fill_price = if order.side == heterogeneous_gnn_trading::strategy::OrderSide::Buy {
                    price * (1.0 + slippage)
                } else {
                    price * (1.0 - slippage)
                };

                strategy.execute_order(order, fill_price);

                trade_log.push(TradeRecord {
                    period,
                    symbol: order.symbol.clone(),
                    side: format!("{:?}", order.side),
                    size: order.size,
                    price: fill_price,
                });
            }
        }

        // Calculate period return
        let total_pnl = strategy.total_pnl();
        let equity = initial_capital + total_pnl;
        let period_return = if period > 0 {
            let prev_equity = tracker.current_equity();
            if prev_equity > 0.0 {
                equity / prev_equity - 1.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        returns.push(period_return);
        tracker.update(equity);
    }

    println!(" Done!\n");

    // Step 4: Calculate and display results
    println!("Step 4: Backtest Results\n");

    let metrics = Metrics::from_returns(&returns, 0.02);  // 2% risk-free rate

    println!("  Performance Summary:");
    println!("  {}", "=".repeat(50));
    println!("  Total Return:       {:>+10.2}%", metrics.total_return * 100.0);
    println!("  Annualized Return:  {:>+10.2}%", metrics.annualized_return * 100.0);
    println!("  Sharpe Ratio:       {:>10.2}", metrics.sharpe_ratio);
    println!("  Sortino Ratio:      {:>10.2}", metrics.sortino_ratio);
    println!("  Max Drawdown:       {:>10.2}%", metrics.max_drawdown * 100.0);
    println!("  Calmar Ratio:       {:>10.2}", metrics.calmar_ratio);
    println!("  {}", "-".repeat(50));
    println!("  Win Rate:           {:>10.2}%", metrics.win_rate * 100.0);
    println!("  Profit Factor:      {:>10.2}", metrics.profit_factor);
    println!("  Total Trades:       {:>10}", trade_log.len());
    println!("  Avg Win:            {:>+10.2}%", metrics.avg_win * 100.0);
    println!("  Avg Loss:           {:>10.2}%", metrics.avg_loss * 100.0);
    println!("  {}", "=".repeat(50));

    // Equity curve summary
    let final_equity = tracker.current_equity();
    println!("\n  Equity Curve:");
    println!("    Start:  ${:.2}", initial_capital);
    println!("    End:    ${:.2}", final_equity);
    println!("    P&L:    ${:+.2}", final_equity - initial_capital);

    // Trade analysis
    if !trade_log.is_empty() {
        println!("\n  Trade Analysis:");

        // Trades by symbol
        let mut by_symbol: HashMap<String, usize> = HashMap::new();
        for trade in &trade_log {
            *by_symbol.entry(trade.symbol.clone()).or_default() += 1;
        }

        println!("    Trades by Symbol:");
        for (symbol, count) in by_symbol.iter() {
            println!("      {}: {}", symbol, count);
        }

        // Sample trades
        println!("\n    Sample Trades (first 5):");
        println!("    {:>6} {:<12} {:>6} {:>10} {:>12}", "Period", "Symbol", "Side", "Size", "Price");
        for trade in trade_log.iter().take(5) {
            println!(
                "    {:>6} {:<12} {:>6} {:>10.4} {:>12.2}",
                trade.period, trade.symbol, trade.side, trade.size, trade.price
            );
        }
    }

    // Step 5: Metapath analysis
    println!("\n  Metapath Contribution Analysis:");
    println!("    (Simulated - in production, track by signal source)");
    println!("    Asset-Correlation-Asset: High contribution");
    println!("    Asset-Exchange-Asset: Medium contribution");
    println!("    Wallet-Asset-Asset: Low contribution");

    println!("\n=== Backtest Complete ===");
    println!("\nKey Insights:");
    println!("  - Heterogeneous GNN captures multi-type relationships");
    println!("  - Different metapaths contribute different signal quality");
    println!("  - Type-aware attention focuses on relevant relationships");

    if metrics.sharpe_ratio > 1.0 {
        println!("\n  Strategy shows promising risk-adjusted returns!");
    } else {
        println!("\n  Strategy needs optimization for better risk-adjusted returns.");
    }
}

/// Trade record for logging
#[derive(Debug)]
struct TradeRecord {
    period: usize,
    symbol: String,
    side: String,
    size: f64,
    price: f64,
}

/// Generate simulated historical price data
fn generate_simulated_data(
    symbols: &[&str],
    num_periods: usize,
) -> HashMap<&str, Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut data = HashMap::new();

    // Base prices and volatilities
    let base_prices = [50000.0, 3000.0, 100.0, 35.0, 1.2];
    let volatilities = [0.02, 0.025, 0.04, 0.035, 0.05];

    // Generate correlated random walks
    for (i, symbol) in symbols.iter().enumerate() {
        let mut prices = Vec::with_capacity(num_periods);
        let mut price = base_prices[i];

        for _ in 0..num_periods {
            // Random return with drift
            let drift = 0.0001;  // Small positive drift
            let vol = volatilities[i];
            let random: f64 = rng.gen_range(-1.0..1.0);
            let return_: f64 = drift + vol * random;

            price *= 1.0 + return_;
            prices.push(price);
        }

        data.insert(*symbol, prices);
    }

    data
}

/// Build heterogeneous graph for a specific period
fn build_period_graph(
    symbols: &[&str],
    historical_data: &HashMap<&str, Vec<f64>>,
    period: usize,
) -> HeterogeneousGraph {
    let schema = GraphSchema::trading_schema();
    let mut graph = HeterogeneousGraph::new(schema);

    // Add asset nodes
    for symbol in symbols {
        if let Some(prices) = historical_data.get(symbol) {
            if let Some(&price) = prices.get(period) {
                // Calculate simple volatility from recent data
                let lookback = 20.min(period);
                let vol = if lookback > 1 {
                    let returns: Vec<f64> = (period.saturating_sub(lookback)..period)
                        .filter_map(|i| {
                            if i > 0 {
                                Some((prices[i] / prices[i-1]).ln())
                            } else {
                                None
                            }
                        })
                        .collect();

                    if !returns.is_empty() {
                        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                        let var: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                        var.sqrt()
                    } else {
                        0.02
                    }
                } else {
                    0.02
                };

                let features = AssetFeatures::new(price, 1_000_000.0, vol);
                graph.add_node(*symbol, NodeType::Asset, features.into());
            }
        }
    }

    // Add exchange node
    let exchange_features = ExchangeFeatures::new(5_000_000.0, 100, 0.95);
    graph.add_node("Bybit", NodeType::Exchange, exchange_features.into());

    // Add correlation edges based on recent data
    let lookback = 30.min(period);
    if lookback > 5 {
        for i in 0..symbols.len() {
            for j in (i+1)..symbols.len() {
                if let (Some(prices_i), Some(prices_j)) = (
                    historical_data.get(symbols[i]),
                    historical_data.get(symbols[j]),
                ) {
                    let start = period.saturating_sub(lookback);
                    let returns_i: Vec<f64> = (start..period)
                        .filter_map(|k| {
                            if k > 0 {
                                Some((prices_i[k] / prices_i[k-1]).ln())
                            } else {
                                None
                            }
                        })
                        .collect();

                    let returns_j: Vec<f64> = (start..period)
                        .filter_map(|k| {
                            if k > 0 {
                                Some((prices_j[k] / prices_j[k-1]).ln())
                            } else {
                                None
                            }
                        })
                        .collect();

                    if returns_i.len() > 5 && returns_j.len() > 5 {
                        let corr = compute_correlation(&returns_i, &returns_j);
                        if corr.abs() > 0.3 {
                            let features = EdgeFeatures::with_correlation(corr, period as u64);
                            graph.add_edge(symbols[i], symbols[j], EdgeType::Correlation, features);
                        }
                    }
                }
            }
        }
    }

    // Add TradesOn edges
    for symbol in symbols {
        let features = EdgeFeatures::with_volume(1_000_000.0, period as u64);
        graph.add_edge(*symbol, "Bybit", EdgeType::TradesOn, features);
    }

    graph
}

/// Compute Pearson correlation
fn compute_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }

    let mean_a: f64 = a.iter().take(n).sum::<f64>() / n as f64;
    let mean_b: f64 = b.iter().take(n).sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a == 0.0 || var_b == 0.0 {
        return 0.0;
    }

    cov / (var_a.sqrt() * var_b.sqrt())
}
