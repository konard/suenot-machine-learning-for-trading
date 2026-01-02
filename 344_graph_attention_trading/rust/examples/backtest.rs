//! Backtest GAT trading strategy
//!
//! This example demonstrates backtesting a Graph Attention Network
//! trading strategy on simulated cryptocurrency data.
//!
//! Run with: cargo run --example backtest

use anyhow::Result;
use gat_trading::api::Candle;
use gat_trading::backtest::{Backtester, PerformanceMetrics};
use gat_trading::gat::GraphAttentionNetwork;
use gat_trading::graph::{GraphBuilder, SparseGraph};
use gat_trading::trading::TradingStrategy;
use rand::Rng;

fn main() -> Result<()> {
    println!("=== GAT Trading Strategy Backtest ===\n");

    // Configuration
    let n_assets = 5;
    let n_periods = 500;
    let initial_capital = 10000.0;

    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOGEUSDT"];

    println!("Configuration:");
    println!("  Assets: {:?}", symbols);
    println!("  Periods: {}", n_periods);
    println!("  Initial capital: ${}", initial_capital);

    // Generate simulated candle data
    println!("\n1. Generating simulated market data...");
    let candles_multi = generate_simulated_candles(&symbols, n_periods);
    println!("   Generated {} candles for each asset", n_periods);

    // Build graph
    println!("\n2. Building asset graph...");
    let adj = GraphBuilder::sample_adjacency(n_assets);
    let graph = SparseGraph::from_dense(&adj);
    println!("   Graph: {} edges, density {:.1}%", graph.num_edges(), graph.density() * 100.0);

    // Create GAT
    println!("\n3. Creating GAT model...");
    let n_features = 16; // Features from FeatureExtractor
    let gat = GraphAttentionNetwork::new(n_features, 32, 2)?;
    println!("   Model: {} parameters", gat.num_parameters());

    // Create strategy
    println!("\n4. Setting up trading strategy...");
    let strategy = TradingStrategy::new("GAT_Cross_Asset");
    println!("   Strategy: {}", strategy.name);
    println!("   Max position: {:.0}%", strategy.max_position * 100.0);
    println!("   Stop loss: {:.0}%", strategy.stop_loss * 100.0);
    println!("   Take profit: {:.0}%", strategy.take_profit * 100.0);

    // Run backtest
    println!("\n5. Running backtest...");
    let backtester = Backtester::new(initial_capital).with_fee(0.001);

    let symbol_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
    let result = backtester.run(&candles_multi, &symbol_refs, &gat, &graph, &strategy);

    // Display results
    println!("\n=== Backtest Results ===\n");
    println!("{}", result.metrics.summary());

    println!("\n--- Trade Statistics ---");
    println!("  Total trades: {}", result.trade_stats.total_trades);
    println!("  Winning trades: {}", result.trade_stats.winning_trades);
    println!("  Losing trades: {}", result.trade_stats.losing_trades);
    println!("  Win rate: {:.1}%", result.trade_stats.win_rate * 100.0);
    println!("  Profit factor: {:.2}", result.trade_stats.profit_factor);

    println!("\n--- Portfolio Summary ---");
    println!("  Initial: ${:.2}", initial_capital);
    println!("  Final: ${:.2}", result.final_portfolio.total_value());
    println!("  Realized PnL: ${:.2}", result.final_portfolio.realized_pnl);

    // Benchmark comparison
    println!("\n6. Running buy-and-hold benchmark (BTC only)...");
    let btc_benchmark = backtester.benchmark_buy_hold(&candles_multi[0]);

    println!("\n--- Strategy vs Benchmark ---");
    println!(
        "  GAT Strategy: {:.2}% return, Sharpe {:.3}",
        result.metrics.total_return * 100.0,
        result.metrics.sharpe_ratio
    );
    println!(
        "  BTC Buy & Hold: {:.2}% return, Sharpe {:.3}",
        btc_benchmark.metrics.total_return * 100.0,
        btc_benchmark.metrics.sharpe_ratio
    );

    let outperformance = result.metrics.total_return - btc_benchmark.metrics.total_return;
    println!(
        "  Outperformance: {:+.2}%",
        outperformance * 100.0
    );

    // Equity curve summary
    println!("\n--- Equity Curve ---");
    let curve = &result.equity_curve;
    if curve.len() >= 10 {
        let step = curve.len() / 10;
        for i in (0..curve.len()).step_by(step) {
            println!("  Period {:4}: ${:.2}", i, curve[i]);
        }
    }

    // Test different strategies
    println!("\n7. Comparing strategy variants...");

    let strategies = vec![
        ("Conservative", TradingStrategy::conservative("GAT_Conservative")),
        ("Default", TradingStrategy::new("GAT_Default")),
        ("Aggressive", TradingStrategy::aggressive("GAT_Aggressive")),
    ];

    println!("\n  {:<15} {:>10} {:>10} {:>10}", "Strategy", "Return", "Sharpe", "MaxDD");
    println!("  {}", "-".repeat(50));

    for (name, strat) in strategies {
        let res = backtester.run(&candles_multi, &symbol_refs, &gat, &graph, &strat);
        println!(
            "  {:<15} {:>9.2}% {:>10.3} {:>9.2}%",
            name,
            res.metrics.total_return * 100.0,
            res.metrics.sharpe_ratio,
            res.metrics.max_drawdown * 100.0
        );
    }

    println!("\n=== Backtest Complete ===\n");

    Ok(())
}

/// Generate simulated candle data with realistic properties
fn generate_simulated_candles(symbols: &[&str], n_periods: usize) -> Vec<Vec<Candle>> {
    let mut rng = rand::thread_rng();

    symbols
        .iter()
        .enumerate()
        .map(|(idx, _symbol)| {
            let mut candles = Vec::with_capacity(n_periods);
            let base_price = match idx {
                0 => 50000.0, // BTC
                1 => 3000.0,  // ETH
                2 => 100.0,   // SOL
                3 => 30.0,    // AVAX
                _ => 0.1,     // DOGE
            };

            let mut price = base_price;
            let volatility = 0.02; // 2% per period

            for i in 0..n_periods {
                // Random walk with slight upward drift
                let drift = 0.0001;
                let shock: f64 = rng.gen_range(-1.0..1.0) * volatility;
                price *= 1.0 + drift + shock;

                let open = price * (1.0 + rng.gen_range(-0.005..0.005));
                let close = price;
                let high = price.max(open) * (1.0 + rng.gen_range(0.0..0.01));
                let low = price.min(open) * (1.0 - rng.gen_range(0.0..0.01));
                let volume = rng.gen_range(1000.0..10000.0) * base_price / 50000.0;

                candles.push(Candle {
                    timestamp: (i as i64) * 3600000, // Hourly
                    open,
                    high,
                    low,
                    close,
                    volume,
                    turnover: volume * close,
                });
            }

            candles
        })
        .collect()
}
