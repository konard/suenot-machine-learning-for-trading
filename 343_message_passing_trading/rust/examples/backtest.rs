//! Backtesting example with MPNN strategy.
//!
//! This example demonstrates:
//! - Running a complete backtest
//! - Calculating performance metrics
//! - Comparing against benchmarks

use mpnn_trading::{
    data::Candle,
    backtest::{Backtester, BacktestConfig, PerformanceMetrics, RollingMetrics},
    graph::{GraphBuilder, MarketGraph},
    mpnn::{AggregationType, MPNN, MPNNConfig},
    strategy::{MPNNStrategy, Signal},
};
use ndarray::Array1;
use std::collections::HashMap;

fn main() {
    println!("=== MPNN Strategy Backtest ===\n");

    // Step 1: Generate synthetic market data
    println!("1. Generating synthetic market data...");
    let candles = generate_synthetic_data(500); // 500 bars
    println!("   Generated {} bars for {} symbols\n",
        candles.values().next().map(|c| c.len()).unwrap_or(0),
        candles.len());

    // Step 2: Configure backtest
    println!("2. Configuring backtest...");
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission_rate: 0.001,
        slippage: 0.0005,
        max_position_size: 0.2,
        rebalance_frequency: 5, // Rebalance every 5 bars
    };
    println!("   Initial capital: ${:.0}", config.initial_capital);
    println!("   Commission: {:.2}%", config.commission_rate * 100.0);
    println!("   Slippage: {:.3}%", config.slippage * 100.0);
    println!("   Max position size: {:.0}%\n", config.max_position_size * 100.0);

    // Step 3: Build MPNN model and strategy
    println!("3. Building MPNN model...");
    let mpnn_config = MPNNConfig {
        input_dim: 8,
        hidden_dim: 32,
        output_dim: 16,
        num_layers: 2,
        aggregation: AggregationType::Mean,
        ..Default::default()
    };
    let mpnn = MPNN::from_config(mpnn_config);
    let strategy = MPNNStrategy::new(mpnn)
        .with_thresholds(0.15, -0.15)
        .with_min_confidence(0.3);
    println!("   MPNN configured\n");

    // Step 4: Generate signals for each time step
    println!("4. Generating signals...");
    let signals = generate_signals_for_backtest(&candles, &strategy);
    println!("   Generated signals for {} periods\n", signals.len());

    // Step 5: Run backtest
    println!("5. Running backtest...");
    let mut backtester = Backtester::new(config.clone());
    let result = backtester.run(&candles, &signals);

    // Step 6: Display results
    println!("\n=== Backtest Results ===\n");
    println!("Final Portfolio Value: ${:.2}", result.final_value);
    println!("Total Return: {:.2}%", result.total_return * 100.0);
    println!("Number of Trades: {}", result.trades.len());

    println!("\n{}", result.metrics.display());

    // Step 7: Compare with buy-and-hold
    println!("=== Benchmark Comparison ===\n");
    let btc_candles = candles.get("BTCUSDT").unwrap();
    let btc_start = btc_candles.first().unwrap().close;
    let btc_end = btc_candles.last().unwrap().close;
    let btc_return = (btc_end - btc_start) / btc_start;

    println!("Buy & Hold BTC Return: {:.2}%", btc_return * 100.0);
    println!("Strategy Return:       {:.2}%", result.total_return * 100.0);
    println!("Outperformance:        {:.2}%", (result.total_return - btc_return) * 100.0);

    // Step 8: Rolling analysis
    println!("\n=== Rolling Analysis (30-bar window) ===\n");
    let equity_returns: Vec<f64> = result.equity_curve
        .windows(2)
        .map(|w| (w[1].1 - w[0].1) / w[0].1)
        .collect();

    let rolling = RollingMetrics::new(&equity_returns, 30, 252);
    println!("Average Rolling Sharpe: {:.2}", rolling.avg_sharpe());
    println!("Sharpe Stability (std): {:.2}", rolling.sharpe_stability());

    // Step 9: Drawdown analysis
    println!("\n=== Drawdown Analysis ===\n");
    let max_dd = result.drawdowns.iter().cloned().fold(0.0_f64, f64::max);
    let avg_dd = result.drawdowns.iter().sum::<f64>() / result.drawdowns.len() as f64;

    println!("Maximum Drawdown: {:.2}%", max_dd * 100.0);
    println!("Average Drawdown: {:.2}%", avg_dd * 100.0);

    // Find longest drawdown period
    let mut max_dd_length = 0;
    let mut current_dd_length = 0;
    for &dd in &result.drawdowns {
        if dd > 0.0 {
            current_dd_length += 1;
            max_dd_length = max_dd_length.max(current_dd_length);
        } else {
            current_dd_length = 0;
        }
    }
    println!("Longest Drawdown Period: {} bars", max_dd_length);

    // Step 10: Trade analysis
    println!("\n=== Trade Analysis ===\n");
    let winning_trades: Vec<_> = result.trades.iter().filter(|t| t.pnl > 0.0).collect();
    let losing_trades: Vec<_> = result.trades.iter().filter(|t| t.pnl < 0.0).collect();

    println!("Total Trades: {}", result.trades.len());
    println!("Winning Trades: {} ({:.1}%)",
        winning_trades.len(),
        winning_trades.len() as f64 / result.trades.len().max(1) as f64 * 100.0);
    println!("Losing Trades: {} ({:.1}%)",
        losing_trades.len(),
        losing_trades.len() as f64 / result.trades.len().max(1) as f64 * 100.0);

    if !winning_trades.is_empty() {
        let avg_win = winning_trades.iter().map(|t| t.pnl).sum::<f64>() / winning_trades.len() as f64;
        println!("Average Win: ${:.2}", avg_win);
    }
    if !losing_trades.is_empty() {
        let avg_loss = losing_trades.iter().map(|t| t.pnl).sum::<f64>() / losing_trades.len() as f64;
        println!("Average Loss: ${:.2}", avg_loss);
    }

    println!("\n=== Backtest Complete ===");
}

/// Generate synthetic market data with realistic properties.
fn generate_synthetic_data(n_bars: usize) -> HashMap<String, Vec<Candle>> {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();

    let symbols = vec![
        ("BTCUSDT", 40000.0, 0.02),
        ("ETHUSDT", 2500.0, 0.025),
        ("SOLUSDT", 100.0, 0.035),
        ("BNBUSDT", 300.0, 0.02),
        ("XRPUSDT", 0.5, 0.03),
    ];

    let mut candles = HashMap::new();

    // Generate correlated random walks
    let btc_returns = generate_returns(n_bars, 0.0002, 0.02);

    for (symbol, start_price, vol_factor) in symbols {
        let mut price = start_price;
        let mut symbol_candles = Vec::with_capacity(n_bars);
        let normal = Normal::new(0.0, 0.01).unwrap();

        for i in 0..n_bars {
            // Correlate with BTC
            let btc_component = btc_returns[i] * 0.7;
            let idio_component = normal.sample(&mut rng) * vol_factor;
            let ret = btc_component + idio_component;

            let open = price;
            price *= 1.0 + ret;

            let high = open.max(price) * (1.0 + rng.gen::<f64>() * 0.005);
            let low = open.min(price) * (1.0 - rng.gen::<f64>() * 0.005);
            let volume = 1000.0 * (1.0 + rng.gen::<f64>());

            symbol_candles.push(Candle {
                timestamp: 1704067200 + (i as u64) * 3600,
                open,
                high,
                low,
                close: price,
                volume,
                symbol: symbol.to_string(),
            });
        }

        candles.insert(symbol.to_string(), symbol_candles);
    }

    candles
}

/// Generate return series.
fn generate_returns(n: usize, drift: f64, vol: f64) -> Vec<f64> {
    use rand_distr::{Distribution, Normal};
    let mut rng = rand::thread_rng();
    let normal = Normal::new(drift, vol).unwrap();

    (0..n).map(|_| normal.sample(&mut rng)).collect()
}

/// Generate signals for each time step in the backtest.
fn generate_signals_for_backtest(
    candles: &HashMap<String, Vec<Candle>>,
    strategy: &MPNNStrategy,
) -> Vec<Vec<Signal>> {
    let min_len = candles.values().map(|c| c.len()).min().unwrap_or(0);
    let lookback = 20;

    let mut all_signals = Vec::with_capacity(min_len);

    for i in 0..min_len {
        if i < lookback {
            // Not enough data yet
            all_signals.push(Vec::new());
            continue;
        }

        // Build graph from recent data
        let mut subset_candles = HashMap::new();
        for (symbol, symbol_candles) in candles {
            let recent = symbol_candles[i.saturating_sub(lookback)..=i].to_vec();
            subset_candles.insert(symbol.clone(), recent);
        }

        // Build graph
        let builder = GraphBuilder::new()
            .correlation_threshold(0.3)
            .min_data_points(10);

        match builder.build_from_candles(&subset_candles) {
            Ok(mut graph) => {
                let timestamp = candles.values()
                    .next()
                    .and_then(|c| c.get(i))
                    .map(|c| c.timestamp)
                    .unwrap_or(0);

                // Generate signals
                match strategy.generate_signals(&mut graph, timestamp) {
                    Ok(signals) => all_signals.push(signals),
                    Err(_) => all_signals.push(Vec::new()),
                }
            }
            Err(_) => all_signals.push(Vec::new()),
        }
    }

    all_signals
}
