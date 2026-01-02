//! Backtest E-GNN trading strategy
//!
//! Example: cargo run --bin backtest_strategy

use equivariant_gnn_trading::{
    EquivariantGNN, MarketGraph, Backtester, TradingSignal, TradeDirection, Candle,
};
use std::collections::HashMap;

fn main() {
    println!("=== E-GNN Strategy Backtest ===\n");

    // Generate synthetic data
    println!("Generating synthetic market data...");
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let num_periods = 500;
    let mut candles_map = HashMap::new();

    for (i, symbol) in symbols.iter().enumerate() {
        let mut candles = Vec::new();
        let base = 100.0 * (i + 1) as f64;

        for t in 0..num_periods {
            let trend = (t as f64 * 0.01).sin() * 20.0;
            let vol = (t as f64 * 0.05).cos().abs() * 5.0;
            let noise = (rand::random::<f64>() - 0.5) * vol;

            let close = base + trend + noise;
            candles.push(Candle::new(
                t as u64 * 3600000,
                close - 1.0,
                close + 2.0,
                close - 2.0,
                close,
                1000.0,
                close * 1000.0,
            ));
        }
        candles_map.insert(symbol.to_string(), candles);
    }

    // Create model and generate signals
    println!("Building graph and model...");
    let graph_builder = MarketGraph::new(0.3);
    let model = EquivariantGNN::new(10, 64, 3, 4);

    // Generate signals for each period
    println!("Generating trading signals...\n");
    let mut all_signals: Vec<Vec<TradingSignal>> = Vec::new();

    for t in 168..num_periods {
        // Create window of data
        let mut window_map = HashMap::new();
        for (sym, candles) in &candles_map {
            window_map.insert(sym.clone(), candles[t-168..t].to_vec());
        }

        let graph = graph_builder.from_candles(&window_map);
        let output = model.forward(&graph);
        let signals_vec = model.signals_from_output(&output, 0.35);

        let signals: Vec<TradingSignal> = graph.nodes.iter().enumerate()
            .map(|(i, node)| TradingSignal::new(
                node.symbol.clone(),
                TradeDirection::from_signal(signals_vec[i]),
                output.position_sizes[i],
                output.direction_probs[[i, signals_vec[i].max(0) as usize]],
                output.volatility[i],
                t as u64 * 3600000,
            ))
            .collect();

        all_signals.push(signals);
    }

    // Run backtest
    println!("Running backtest...\n");
    let backtester = Backtester::new(0.0004, 0.0001, 10000.0);
    let result = backtester.run(&all_signals, &candles_map);

    // Print results
    println!("=== Backtest Results ===\n");
    println!("Initial Capital:  $10,000.00");
    println!("Final Capital:    ${:.2}", result.final_capital);
    println!("Total Return:     {:.2}%", result.metrics.total_return * 100.0);
    println!("Sharpe Ratio:     {:.2}", result.metrics.sharpe_ratio);
    println!("Sortino Ratio:    {:.2}", result.metrics.sortino_ratio);
    println!("Max Drawdown:     {:.2}%", result.metrics.max_drawdown * 100.0);
    println!("Win Rate:         {:.1}%", result.metrics.win_rate * 100.0);
    println!("Profit Factor:    {:.2}", result.metrics.profit_factor);
    println!("Number of Trades: {}", result.trades.len());

    // Sample trades
    if !result.trades.is_empty() {
        println!("\n=== Sample Trades ===\n");
        for trade in result.trades.iter().take(5) {
            println!("{} {:?}: entry=${:.2} exit=${:.2} pnl={:.2}%",
                trade.symbol,
                trade.direction,
                trade.entry_price,
                trade.exit_price,
                trade.pnl * 100.0);
        }
    }

    // Equity curve stats
    if result.equity_curve.len() > 1 {
        println!("\n=== Equity Curve ===\n");
        println!("Start: ${:.2}", result.equity_curve.first().unwrap());
        println!("End:   ${:.2}", result.equity_curve.last().unwrap());
        println!("Min:   ${:.2}", result.equity_curve.iter().cloned().fold(f64::INFINITY, f64::min));
        println!("Max:   ${:.2}", result.equity_curve.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    }

    println!("\nBacktest complete!");
}
