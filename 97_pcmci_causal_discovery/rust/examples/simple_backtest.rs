//! Simple Backtest Example
//!
//! Demonstrates how to use PCMCI causal discovery for backtesting
//! with synthetic market data.

use pcmci_causal_discovery::{
    BacktestConfig, Backtester, CausalTradingStrategy, PCMCIConfig, IndependenceTest,
};
use pcmci_causal_discovery::data::{generate_synthetic_data, generate_synthetic_causal_data};
use pcmci_causal_discovery::model::PCMCICausalDiscovery;

fn main() {
    env_logger::init();
    println!("=== PCMCI Causal Discovery Backtest ===\n");

    // --- Part 1: Demonstrate causal discovery on synthetic data ---
    println!("--- Part 1: Causal Discovery on Synthetic Data ---\n");

    // Generate data with known causal structure
    // X0(t-1) -> X1(t) with strength 0.6
    // X1(t-2) -> X2(t) with strength -0.4
    let true_links = vec![
        (0, 1, 1, 0.6),
        (1, 2, 2, -0.4),
    ];
    let causal_data = generate_synthetic_causal_data(3, 500, &true_links);

    println!("Generated synthetic causal data: {} samples, {} variables",
        causal_data.nrows(), causal_data.ncols());
    println!("True causal links:");
    println!("  X0(t-1) -> X1(t): strength = 0.6");
    println!("  X1(t-2) -> X2(t): strength = -0.4\n");

    // Run PCMCI
    let pcmci_config = PCMCIConfig {
        max_lag: 3,
        significance_level: 0.05,
        test_type: IndependenceTest::ParCorr,
        max_conds_dim: Some(3),
    };

    let mut pcmci = PCMCICausalDiscovery::new(pcmci_config.clone());
    let graph = pcmci.fit(&causal_data);

    // Print discovered links
    println!("Discovered causal links (p < 0.05):");
    let significant = graph.get_significant_links(0.05);
    if significant.is_empty() {
        println!("  No significant links found.");
    } else {
        for link in &significant {
            println!(
                "  X{}(t-{}) -> X{}: strength = {:.4}, p-value = {:.4}",
                link.source, link.lag, link.target, link.strength, link.p_value
            );
        }
    }

    let summary = graph.summary();
    println!("\nGraph summary:");
    println!("  Significant links: {}", summary.n_significant_links);
    println!("  Average strength: {:.4}", summary.avg_link_strength);
    println!("  Max strength: {:.4}", summary.max_link_strength);

    // --- Part 2: Trading backtest with causal strategy ---
    println!("\n--- Part 2: Causal Trading Backtest ---\n");

    // Configure the causal strategy
    let pcmci_config = PCMCIConfig {
        max_lag: 3,
        significance_level: 0.1,
        test_type: IndependenceTest::ParCorr,
        max_conds_dim: Some(3),
    };

    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.1,
        stop_loss: Some(0.02),
        take_profit: Some(0.04),
        fee_rate: 0.001,
        slippage: 0.0005,
        lookback_window: 100,
        reestimate_freq: 50,
        min_link_strength: 0.1,
        significance_level: 0.1,
    };

    // Generate synthetic market data
    println!("Generating synthetic market data...");
    let data = generate_synthetic_data(500, 0.02);
    println!("Generated {} bars of data\n", data.len());

    // Create strategy
    let mut strategy = CausalTradingStrategy::new(pcmci_config, backtest_config.clone());

    // Run backtest
    println!("Running causal backtest...");
    let backtester = Backtester::new(backtest_config);
    let result = backtester.run(&mut strategy, &data);

    // Print results
    println!("\n=== Backtest Results ===\n");
    println!("Total Return:          {:.2}%", result.total_return * 100.0);
    println!("Annual Return:         {:.2}%", result.annual_return * 100.0);
    println!("Sharpe Ratio:          {:.2}", result.sharpe_ratio);
    println!("Max Drawdown:          {:.2}%", result.max_drawdown * 100.0);
    println!("Win Rate:              {:.2}%", result.win_rate * 100.0);
    println!("Profit Factor:         {:.2}", result.profit_factor);
    println!("Total Trades:          {}", result.total_trades);
    println!("Avg Trade Return:      {:.2}%", result.avg_trade_return * 100.0);
    println!("Graph Updates:         {}", result.n_graph_updates);
    println!("Avg Significant Links: {:.1}", result.avg_significant_links);

    // Show some individual trades
    if !result.trades.is_empty() {
        println!("\n=== Sample Trades ===\n");
        for (i, trade) in result.trades.iter().take(5).enumerate() {
            println!(
                "Trade {}: {} @ {:.2} -> {:.2}, PnL: ${:.2} ({:.2}%), Links: {}, Reason: {}",
                i + 1,
                if trade.size > 0.0 { "LONG" } else { "SHORT" },
                trade.entry_price,
                trade.exit_price,
                trade.pnl,
                trade.return_pct * 100.0,
                trade.n_causal_links,
                trade.exit_reason
            );
        }
    }

    // Print causal graph summary
    println!("\n=== Final Causal Graph ===\n");
    strategy.print_graph_summary();

    // Show equity curve summary
    if !result.equity_curve.is_empty() {
        let initial = result.equity_curve.first().unwrap();
        let final_ = result.equity_curve.last().unwrap();
        let min = result.equity_curve.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = result.equity_curve.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("\n=== Equity Curve Summary ===\n");
        println!("Initial: ${:.2}", initial);
        println!("Final:   ${:.2}", final_);
        println!("Min:     ${:.2}", min);
        println!("Max:     ${:.2}", max);
    }

    println!("\n=== Backtest Complete ===");
}
