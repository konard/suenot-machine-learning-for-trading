//! Demo: Backtesting Multi-Agent Strategy
//!
//! This demo shows how to backtest a multi-agent trading strategy.

use multi_agent_trading::{
    agents::{Agent, BearAgent, BullAgent, RiskManagerAgent, TechnicalAgent},
    backtest::{buy_and_hold_benchmark, MultiAgentBacktester},
    data::create_mock_data,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(60));
    println!("Multi-Agent LLM Trading - Backtesting Demo");
    println!("{}", "=".repeat(60));

    // Create market data
    println!("\n1. Creating market data with multiple regimes...");
    let data = create_mock_data("DEMO", 504, 100.0); // ~2 years

    let start_price = data.candles[0].close;
    let end_price = data.latest_close().unwrap_or(0.0);
    let total_return = (end_price / start_price - 1.0) * 100.0;

    println!(
        "   Period: {} to {}",
        data.candles[0].timestamp.format("%Y-%m-%d"),
        data.candles.last().unwrap().timestamp.format("%Y-%m-%d")
    );
    println!("   Start price: ${:.2}", start_price);
    println!("   End price: ${:.2}", end_price);
    println!("   Total return: {:.1}%", total_return);
    println!("   Max drawdown: {:.1}%", data.max_drawdown() * 100.0);

    // Create agents
    println!("\n2. Creating agent team...");
    let tech = TechnicalAgent::new("Tech");
    let bull = BullAgent::new("Bull");
    let bear = BearAgent::new("Bear");
    let risk = RiskManagerAgent::new("Risk", 0.05, 0.15);

    let agents: Vec<&dyn Agent> = vec![&tech, &bull, &bear, &risk];
    println!("   Created {} analysis agents", agents.len());

    // Run backtest
    println!("\n3. Running backtest (this may take a moment)...");
    println!("{}", "-".repeat(60));

    let mut backtester =
        MultiAgentBacktester::new(agents, 100000.0, 0.2, 0.001);

    let result = backtester.run("DEMO", &data, 50, 5).await?;

    result.print_summary();

    // Recent trades
    if !result.trades.is_empty() {
        println!("\nRecent Trades:");
        for trade in result.trades.iter().rev().take(5) {
            let pnl_str = if trade.pnl >= 0.0 {
                format!("+${:.2}", trade.pnl)
            } else {
                format!("-${:.2}", trade.pnl.abs())
            };
            println!(
                "   {}: {} @ ${:.2} [{}]",
                trade.timestamp.format("%Y-%m-%d"),
                trade.action,
                trade.price,
                pnl_str
            );
        }
    }

    // Compare to benchmark
    println!("\n4. Comparing to Buy & Hold benchmark...");
    println!("{}", "-".repeat(60));

    let benchmark = buy_and_hold_benchmark(&data, 100000.0);

    println!("\n   Buy & Hold:");
    println!("      Total Return: {:.2}%", benchmark.total_return);
    println!("      Sharpe Ratio: {:.2}", benchmark.sharpe_ratio);
    println!("      Max Drawdown: {:.2}%", benchmark.max_drawdown);

    // Comparison
    let excess_return = result.total_return - benchmark.total_return;
    let sharpe_diff = result.sharpe_ratio - benchmark.sharpe_ratio;
    let dd_improvement = benchmark.max_drawdown - result.max_drawdown;

    println!("\n   Strategy vs Benchmark:");
    println!("      Excess Return: {:+.2}%", excess_return);
    println!("      Sharpe Difference: {:+.2}", sharpe_diff);
    println!("      Drawdown Improvement: {:+.2}%", dd_improvement);

    // Summary table
    println!("\n{}", "=".repeat(60));
    println!("COMPARISON SUMMARY");
    println!("{}", "=".repeat(60));
    println!(
        "\n{:<20} {:>12} {:>12} {:>12}",
        "Metric", "Multi-Agent", "Buy & Hold", "Difference"
    );
    println!("{}", "-".repeat(60));
    println!(
        "{:<20} {:>11.2}% {:>11.2}% {:>+11.2}%",
        "Total Return", result.total_return, benchmark.total_return, excess_return
    );
    println!(
        "{:<20} {:>12.2} {:>12.2} {:>+12.2}",
        "Sharpe Ratio", result.sharpe_ratio, benchmark.sharpe_ratio, sharpe_diff
    );
    println!(
        "{:<20} {:>11.2}% {:>11.2}% {:>+11.2}%",
        "Max Drawdown", result.max_drawdown, benchmark.max_drawdown, -dd_improvement
    );
    println!(
        "{:<20} {:>12} {:>12}",
        "Number of Trades", result.num_trades, benchmark.num_trades
    );
    println!(
        "{:<20} {:>11.2}% {:>11.2}%",
        "Win Rate", result.win_rate, benchmark.win_rate
    );

    // Conclusions
    println!("\n{}", "=".repeat(60));
    println!("CONCLUSIONS");
    println!("{}", "=".repeat(60));
    println!(
        r#"
   Key Takeaways from Backtest:

   1. MULTI-AGENT VS BUY & HOLD
      - Multi-agent strategies aim to reduce drawdowns
      - More consistent returns across market regimes
      - But may underperform in strong bull markets

   2. TRADE-OFFS
      - More trades = higher transaction costs
      - Conservative sizing reduces risk but limits upside
      - Debate mechanism adds validation layer

   IMPORTANT: This is a demonstration with simulated data.
   Real trading involves additional risks and complexities.
"#
    );

    Ok(())
}
