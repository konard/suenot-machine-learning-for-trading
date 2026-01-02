//! Backtest: Graph Pooling Trading Strategy
//!
//! This example demonstrates backtesting the graph pooling strategy
//! using historical data from Bybit.

use graph_pooling_trading::prelude::*;
use graph_pooling_trading::strategy::{BacktestResult, Regime, StrategyConfig};
use graph_pooling_trading::BybitClient;
use std::collections::HashMap;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// Simulated historical data for backtesting
/// In production, you would fetch this from Bybit's historical API
fn generate_simulated_data(
    symbols: &[&str],
    n_periods: usize,
) -> HashMap<String, Vec<graph_pooling_trading::Kline>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut data = HashMap::new();

    // Generate correlated price series
    for (idx, symbol) in symbols.iter().enumerate() {
        let base_price = match *symbol {
            s if s.starts_with("BTC") => 45000.0,
            s if s.starts_with("ETH") => 2500.0,
            s if s.starts_with("SOL") => 100.0,
            s if s.starts_with("BNB") => 300.0,
            _ => 50.0 + rng.gen::<f64>() * 100.0,
        };

        let mut price = base_price;
        let mut klines = Vec::with_capacity(n_periods);

        // Create cluster-correlated movements
        let cluster = idx % 3; // 3 clusters

        for t in 0..n_periods {
            // Market-wide factor
            let market_factor = (t as f64 * 0.1).sin() * 0.02;

            // Cluster factor
            let cluster_factor = ((t as f64 + cluster as f64 * 2.0) * 0.15).sin() * 0.015;

            // Idiosyncratic factor
            let idio_factor = rng.gen::<f64>() * 0.02 - 0.01;

            // Combined return
            let ret = market_factor + cluster_factor + idio_factor;
            price *= 1.0 + ret;

            let high = price * (1.0 + rng.gen::<f64>() * 0.01);
            let low = price * (1.0 - rng.gen::<f64>() * 0.01);
            let open = price * (1.0 + (rng.gen::<f64>() - 0.5) * 0.005);
            let volume = 1000.0 + rng.gen::<f64>() * 5000.0;

            klines.push(graph_pooling_trading::Kline {
                symbol: symbol.to_string(),
                open_time: (t * 3600000) as i64,
                open,
                high,
                low,
                close: price,
                volume,
                turnover: volume * price,
            });
        }

        data.insert(symbol.to_string(), klines);
    }

    data
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    println!("=================================================");
    println!("  Graph Pooling Strategy Backtest");
    println!("=================================================\n");

    // Configuration
    let symbols = vec![
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "LTCUSDT",
        "AVAXUSDT", "LINKUSDT",
    ];

    let lookback = 50;  // Candles for building graph
    let n_periods = 200; // Total simulation periods
    let rebalance_freq = 5; // Rebalance every 5 periods

    println!("Backtest Configuration:");
    println!("  Symbols: {}", symbols.len());
    println!("  Lookback: {} candles", lookback);
    println!("  Total periods: {}", n_periods);
    println!("  Rebalance frequency: every {} periods", rebalance_freq);

    // Generate simulated data
    println!("\nGenerating simulated market data...");
    let all_data = generate_simulated_data(&symbols, n_periods + lookback);

    // Initialize strategy
    let config = StrategyConfig {
        n_clusters_l1: 4,
        n_clusters_l2: 2,
        prediction_weight: 0.4,
        momentum_weight: 0.6,
        max_position: 0.15,
        ..Default::default()
    };
    let mut strategy = GraphPoolingStrategy::with_config(config);

    // Tracking variables
    let mut portfolio_returns = Vec::new();
    let mut turnovers = Vec::new();
    let mut regimes = Vec::new();
    let mut positions: HashMap<String, f64> = HashMap::new();

    println!("\nRunning backtest...");
    println!("{:-<70}", "");

    // Main backtest loop
    for t in lookback..n_periods + lookback {
        // Extract lookback window for each symbol
        let mut window_data: HashMap<String, Vec<graph_pooling_trading::Kline>> = HashMap::new();
        for symbol in &symbols {
            if let Some(klines) = all_data.get(*symbol) {
                let window: Vec<_> = klines[t - lookback..t].to_vec();
                window_data.insert(symbol.to_string(), window);
            }
        }

        // Rebalance on schedule
        if (t - lookback) % rebalance_freq == 0 {
            // Build graph from window
            let graph = match MarketGraph::from_klines(&window_data) {
                Ok(g) => g,
                Err(e) => {
                    info!("Failed to build graph at t={}: {}", t, e);
                    continue;
                }
            };

            // Generate signals
            let signals = match strategy.generate_signals(&graph) {
                Ok(s) => s,
                Err(e) => {
                    info!("Failed to generate signals at t={}: {}", t, e);
                    continue;
                }
            };

            // Calculate turnover
            let mut turnover = 0.0;
            for signal in &signals {
                let old_pos = positions.get(&signal.symbol).copied().unwrap_or(0.0);
                turnover += (signal.position - old_pos).abs();
                positions.insert(signal.symbol.clone(), signal.position);
            }
            turnovers.push(turnover);

            // Record regime
            regimes.push(strategy.current_regime());

            // Print progress every 20 rebalances
            if turnovers.len() % 20 == 0 {
                println!(
                    "  Period {}: Regime = {}, Turnover = {:.4}",
                    t - lookback,
                    strategy.current_regime(),
                    turnover
                );
            }
        }

        // Calculate portfolio return
        let mut period_return = 0.0;
        for (symbol, &pos) in &positions {
            if let Some(klines) = all_data.get(symbol) {
                if t < klines.len() && t > 0 {
                    let ret = klines[t].close / klines[t - 1].close - 1.0;
                    period_return += pos * ret;
                }
            }
        }

        // Apply transaction costs (0.1% per turnover)
        if (t - lookback) % rebalance_freq == 0 && !turnovers.is_empty() {
            let cost = turnovers.last().unwrap() * 0.001;
            period_return -= cost;
        }

        portfolio_returns.push(period_return);
    }

    println!("{:-<70}", "");

    // Calculate and display results
    let result = BacktestResult::from_returns(&portfolio_returns, &turnovers, &regimes);

    println!("\n{}", result.summary());

    // Additional analysis
    println!("\nDetailed Analysis:");
    println!("{:-<50}", "");

    // Monthly-like returns (every 20 periods)
    println!("\nPeriodic Returns (every 20 periods):");
    let mut cumulative = 1.0;
    for (i, chunk) in portfolio_returns.chunks(20).enumerate() {
        let period_ret: f64 = chunk.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;
        cumulative *= 1.0 + period_ret;
        println!(
            "  Period {:>3}-{:<3}: {:>+7.2}% (Cumulative: {:>+7.2}%)",
            i * 20,
            (i + 1) * 20,
            period_ret * 100.0,
            (cumulative - 1.0) * 100.0
        );
    }

    // Regime analysis
    println!("\nRegime Distribution:");
    let total_regimes = regimes.len();
    for (regime, count) in &result.regime_counts {
        let pct = *count as f64 / total_regimes as f64 * 100.0;
        println!("  {}: {} ({:.1}%)", regime, count, pct);
    }

    // Best and worst periods
    if !portfolio_returns.is_empty() {
        let best = portfolio_returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst = portfolio_returns.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("\nExtreme Returns:");
        println!("  Best period:  {:>+7.2}%", best * 100.0);
        println!("  Worst period: {:>+7.2}%", worst * 100.0);
    }

    println!("\n=================================================");
    println!("  Backtest completed!");
    println!("=================================================");

    Ok(())
}
