//! Trading strategy example using graph-based signals.
//!
//! This example demonstrates a complete trading workflow:
//! 1. Generate simulated market data
//! 2. Build correlation graphs
//! 3. Generate trading signals
//! 4. Run backtest

use graph_generation_trading::{
    data::{MarketData, OHLCV},
    graph::{GraphBuilder, GraphType, CorrelationMethod},
    trading::{BacktestEngine, GraphSignals, Portfolio, PortfolioOptimizer},
};
use chrono::{Utc, Duration};
use rand::Rng;
use std::collections::HashMap;

fn main() {
    println!("=== Graph-Based Trading Strategy Example ===\n");

    // Configuration
    let symbols = vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "SOLUSDT".to_string(),
        "AVAXUSDT".to_string(),
        "DOGEUSDT".to_string(),
    ];
    let num_candles = 1000;
    let initial_capital = 10000.0;

    // Generate market data
    println!("1. Generating simulated market data...");
    let market_data = generate_market_data(&symbols, num_candles);
    println!("   Generated {} candles for {} symbols\n", num_candles, symbols.len());

    // Build rolling graphs and generate signals
    println!("2. Building rolling correlation graphs...");
    let window_size = 100;
    let step_size = 1;

    let mut all_signals: Vec<HashMap<String, f64>> = Vec::new();
    let mut prev_graph_density = 0.0;

    for t in window_size..num_candles {
        // Create windowed data
        let windowed_data = create_window(&market_data, t - window_size, t);

        // Build graph for this window
        let graph = GraphBuilder::new()
            .with_method(CorrelationMethod::Pearson)
            .with_graph_type(GraphType::Threshold)
            .with_threshold(0.5)
            .build(&windowed_data)
            .unwrap();

        // Generate signals
        let signals = GraphSignals::new(&graph);
        let centrality_signals = signals.centrality_signals(0.2, 0.2);

        // Adjust signals based on regime
        let regime = signals.regime_indicator();
        let multiplier = regime.position_multiplier();

        let adjusted_signals: HashMap<String, f64> = centrality_signals
            .into_iter()
            .map(|(s, v)| (s, v * multiplier))
            .collect();

        all_signals.push(adjusted_signals);

        // Log periodically
        if t % 200 == 0 {
            let current_density = graph.density();
            let density_change = current_density - prev_graph_density;
            prev_graph_density = current_density;

            println!("   t={}: density={:.3} (Δ{:.3}), regime={:?}",
                t, current_density, density_change, regime);
        }
    }

    println!("   Generated {} signal snapshots\n", all_signals.len());

    // Prepare data for backtest
    println!("3. Running backtest...");
    let backtest_data = prepare_backtest_data(&market_data, window_size);

    let engine = BacktestEngine::new(initial_capital)
        .with_commission(0.001)
        .with_slippage(0.0005);

    let result = engine.run(&backtest_data, &all_signals);

    // Display results
    println!("\n=== Backtest Results ===\n");
    println!("Initial Capital: ${:.2}", initial_capital);
    println!("Final Equity: ${:.2}", result.equity_curve.last().unwrap_or(&initial_capital));
    println!("Total Return: {:.2}%", result.total_return * 100.0);
    println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("Number of Trades: {}", result.num_trades);

    // Equity curve summary
    println!("\n--- Equity Curve Summary ---");
    let equity = &result.equity_curve;
    if !equity.is_empty() {
        let step = equity.len() / 5;
        for i in (0..equity.len()).step_by(step.max(1)) {
            let pct = (equity[i] - initial_capital) / initial_capital * 100.0;
            println!("  t={}: ${:.2} ({:+.2}%)", i, equity[i], pct);
        }
    }

    // Portfolio optimization example
    println!("\n4. Portfolio Optimization...");
    let optimizer = PortfolioOptimizer::new()
        .with_max_weight(0.3)
        .with_risk_aversion(1.5);

    // Get final signals
    if let Some(final_signals) = all_signals.last() {
        let weights = optimizer.signal_weight(final_signals);

        println!("\n--- Optimized Portfolio Weights ---");
        for (symbol, weight) in &weights {
            let direction = if *weight > 0.0 { "LONG" } else if *weight < 0.0 { "SHORT" } else { "FLAT" };
            println!("  {}: {:.1}% ({})", symbol, weight * 100.0, direction);
        }
    }

    println!("\n=== Example Complete ===");
}

/// Generate correlated market data
fn generate_market_data(symbols: &[String], num_candles: usize) -> MarketData {
    let mut rng = rand::thread_rng();
    let mut market_data = MarketData::new(symbols.to_vec(), "1h");

    let base_prices: Vec<f64> = vec![40000.0, 2000.0, 100.0, 30.0, 0.08];

    // Generate common market factor with regime changes
    let market_factor: Vec<f64> = (0..num_candles)
        .map(|i| {
            let regime = (i / 200) % 4;
            let base_vol = match regime {
                0 => 0.01,  // Low volatility
                1 => 0.02,  // Medium volatility
                2 => 0.03,  // High volatility
                _ => 0.015, // Trending
            };

            let trend = match regime {
                3 => 0.001,  // Uptrend
                _ => 0.0,
            };

            trend + rng.gen_range(-base_vol..base_vol)
        })
        .collect();

    for (idx, symbol) in symbols.iter().enumerate() {
        let base_price = base_prices.get(idx).copied().unwrap_or(100.0);
        let mut price = base_price;

        // Symbol-specific correlation to market
        let beta = match idx {
            0 => 1.0,   // BTC is the market
            1 => 1.2,   // ETH higher beta
            2 => 1.5,   // SOL higher beta
            3 => 1.3,   // AVAX
            _ => 2.0,   // DOGE very high beta
        };

        let candles: Vec<OHLCV> = (0..num_candles)
            .map(|i| {
                let now = Utc::now() - Duration::hours((num_candles - i) as i64);

                let market_return = market_factor[i] * beta;
                let idiosyncratic = rng.gen_range(-0.005..0.005);
                let total_return = market_return + idiosyncratic;

                price *= 1.0 + total_return;

                let volatility = price * 0.01;
                let high = price + rng.gen_range(0.0..volatility);
                let low = price - rng.gen_range(0.0..volatility);
                let open = price * (1.0 + rng.gen_range(-0.005..0.005));
                let volume = rng.gen_range(1000.0..10000.0);

                OHLCV::new(now, open, high, low, price, volume)
            })
            .collect();

        market_data.add_candles(symbol, candles);
    }

    market_data
}

/// Create a windowed subset of market data
fn create_window(data: &MarketData, start: usize, end: usize) -> MarketData {
    let mut windowed = MarketData::new(data.symbols.clone(), &data.timeframe);

    for (symbol, candles) in data.symbols.iter().zip(data.data.iter()) {
        let window: Vec<OHLCV> = candles[start..end].to_vec();
        windowed.add_candles(symbol, window);
    }

    windowed
}

/// Prepare data for backtesting
fn prepare_backtest_data(data: &MarketData, skip: usize) -> HashMap<String, Vec<OHLCV>> {
    data.symbols
        .iter()
        .zip(data.data.iter())
        .map(|(symbol, candles)| {
            (symbol.clone(), candles[skip..].to_vec())
        })
        .collect()
}
