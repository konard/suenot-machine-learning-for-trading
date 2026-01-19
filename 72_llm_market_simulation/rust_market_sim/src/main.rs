//! LLM Market Simulation CLI
//!
//! Multi-agent market simulation with configurable agents and parameters.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use llm_market_sim::{
    agents::{ValueInvestor, MomentumTrader, MarketMaker},
    simulation::{SimulationEngine, calculate_performance_metrics, detect_bubble},
};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SimulationType {
    /// Balanced market with all agent types
    Balanced,
    /// Value-heavy market
    ValueHeavy,
    /// Momentum-heavy market (bubble-prone)
    MomentumHeavy,
    /// Market maker dominant
    LiquidityFocused,
}

#[derive(Parser)]
#[command(name = "llm-market-sim")]
#[command(about = "Multi-agent market simulation with LLM-style agents")]
struct Cli {
    /// Simulation type
    #[arg(short, long, value_enum, default_value = "balanced")]
    simulation_type: SimulationType,

    /// Number of simulation steps
    #[arg(short = 'n', long, default_value = "500")]
    steps: u64,

    /// Initial price
    #[arg(long, default_value = "100.0")]
    initial_price: f64,

    /// Fundamental value
    #[arg(long, default_value = "100.0")]
    fundamental: f64,

    /// Price volatility per step
    #[arg(long, default_value = "0.02")]
    volatility: f64,

    /// Initial cash per agent
    #[arg(long, default_value = "100000.0")]
    initial_cash: f64,

    /// Initial shares per agent
    #[arg(long, default_value = "100")]
    initial_shares: i64,

    /// Show detailed output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         LLM Market Simulation - Rust Edition             ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Create simulation engine
    let mut engine = SimulationEngine::new(
        cli.initial_price,
        cli.fundamental,
        cli.volatility,
    );

    // Add agents based on simulation type
    println!("Simulation Type: {:?}", cli.simulation_type);
    println!("Creating agents...");

    match cli.simulation_type {
        SimulationType::Balanced => {
            // 3 value investors
            for i in 0..3 {
                engine.add_agent(Box::new(
                    ValueInvestor::new(
                        format!("value_{}", i + 1),
                        cli.initial_cash,
                        cli.initial_shares,
                        cli.fundamental,
                    )
                    .with_thresholds(0.05 + i as f64 * 0.02, 0.05 + i as f64 * 0.02)
                ));
            }
            // 3 momentum traders
            for i in 0..3 {
                engine.add_agent(Box::new(
                    MomentumTrader::new(
                        format!("momentum_{}", i + 1),
                        cli.initial_cash,
                        cli.initial_shares,
                    )
                    .with_windows(5 + i * 5, 20 + i * 10)
                ));
            }
            // 2 market makers
            for i in 0..2 {
                engine.add_agent(Box::new(
                    MarketMaker::new(
                        format!("mm_{}", i + 1),
                        cli.initial_cash * 2.0,
                        cli.initial_shares * 2,
                    )
                    .with_spread(0.002 + i as f64 * 0.001)
                ));
            }
            println!("  Added: 3 Value Investors, 3 Momentum Traders, 2 Market Makers");
        }
        SimulationType::ValueHeavy => {
            for i in 0..5 {
                engine.add_agent(Box::new(
                    ValueInvestor::new(
                        format!("value_{}", i + 1),
                        cli.initial_cash,
                        cli.initial_shares,
                        cli.fundamental,
                    )
                ));
            }
            for i in 0..2 {
                engine.add_agent(Box::new(
                    MomentumTrader::new(
                        format!("momentum_{}", i + 1),
                        cli.initial_cash,
                        cli.initial_shares,
                    )
                ));
            }
            engine.add_agent(Box::new(
                MarketMaker::new("mm_1".to_string(), cli.initial_cash * 2.0, cli.initial_shares * 2)
            ));
            println!("  Added: 5 Value Investors, 2 Momentum Traders, 1 Market Maker");
        }
        SimulationType::MomentumHeavy => {
            engine.add_agent(Box::new(
                ValueInvestor::new(
                    "value_1".to_string(),
                    cli.initial_cash,
                    cli.initial_shares,
                    cli.fundamental,
                )
                .with_thresholds(0.15, 0.15) // Very patient
            ));
            for i in 0..5 {
                engine.add_agent(Box::new(
                    MomentumTrader::new(
                        format!("momentum_{}", i + 1),
                        cli.initial_cash,
                        cli.initial_shares,
                    )
                    .with_windows(3, 8)
                    .with_threshold(0.01)
                ));
            }
            for i in 0..2 {
                engine.add_agent(Box::new(
                    MarketMaker::new(
                        format!("mm_{}", i + 1),
                        cli.initial_cash * 2.0,
                        cli.initial_shares * 2,
                    )
                ));
            }
            println!("  Added: 1 Value Investor, 5 Momentum Traders, 2 Market Makers");
            println!("  (This configuration is prone to bubble formation)");
        }
        SimulationType::LiquidityFocused => {
            for i in 0..2 {
                engine.add_agent(Box::new(
                    ValueInvestor::new(
                        format!("value_{}", i + 1),
                        cli.initial_cash,
                        cli.initial_shares,
                        cli.fundamental,
                    )
                ));
            }
            for i in 0..2 {
                engine.add_agent(Box::new(
                    MomentumTrader::new(
                        format!("momentum_{}", i + 1),
                        cli.initial_cash,
                        cli.initial_shares,
                    )
                ));
            }
            for i in 0..4 {
                engine.add_agent(Box::new(
                    MarketMaker::new(
                        format!("mm_{}", i + 1),
                        cli.initial_cash * 2.0,
                        cli.initial_shares * 2,
                    )
                    .with_spread(0.001 + i as f64 * 0.0005)
                    .with_quote_size(20)
                ));
            }
            println!("  Added: 2 Value Investors, 2 Momentum Traders, 4 Market Makers");
        }
    }

    println!();
    println!("Parameters:");
    println!("  Initial Price: ${:.2}", cli.initial_price);
    println!("  Fundamental Value: ${:.2}", cli.fundamental);
    println!("  Volatility: {:.1}%", cli.volatility * 100.0);
    println!("  Steps: {}", cli.steps);
    println!();

    // Run simulation with progress bar
    println!("Running simulation...");
    let pb = ProgressBar::new(cli.steps);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let result = engine.run_with_progress(cli.steps, |step, price| {
        pb.set_position(step);
        pb.set_message(format!("${:.2}", price));
    });

    pb.finish_with_message("Complete!");
    println!();

    // Print results
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                    SIMULATION RESULTS                    ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    println!("Price Evolution:");
    println!("  Start: ${:.2}", result.price_history[0]);
    println!("  End:   ${:.2}", result.final_price);
    println!("  Min:   ${:.2}", result.price_history.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("  Max:   ${:.2}", result.price_history.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!();

    // Calculate metrics
    let metrics = calculate_performance_metrics(
        &result.price_history,
        Some(&result.fundamental_history),
        0.02,
        252.0,
    );

    println!("Performance Metrics:");
    println!("  Total Return: {:.2}%", metrics.total_return_pct);
    println!("  Volatility (annualized): {:.2}%", metrics.volatility * 100.0);
    if let Some(sharpe) = metrics.sharpe_ratio {
        println!("  Sharpe Ratio: {:.3}", sharpe);
    }
    if let Some(sortino) = metrics.sortino_ratio {
        println!("  Sortino Ratio: {:.3}", sortino);
    }
    println!("  Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    println!("  Win Rate: {:.1}%", metrics.win_rate * 100.0);
    println!("  VaR (95%): {:.2}%", metrics.var_95 * 100.0);
    if let Some(te) = metrics.tracking_error {
        println!("  Tracking Error: {:.3}", te);
    }
    if let Some(dev) = metrics.final_deviation_pct {
        println!("  Final Deviation from Fundamental: {:.2}%", dev);
    }
    if let Some(corr) = metrics.fundamental_correlation {
        println!("  Fundamental Correlation: {:.3}", corr);
    }
    println!();

    // Bubble detection
    let bubble_info = detect_bubble(&result.price_history, &result.fundamental_history, 0.30);

    println!("Bubble Analysis (threshold: 30%):");
    println!("  Bubble Detected: {}", bubble_info.bubble_detected);
    println!("  Number of Bubble Periods: {}", bubble_info.num_bubbles);
    println!("  Max Deviation: {:.1}%", bubble_info.max_deviation * 100.0);
    println!("  Time in Bubble: {:.1}%", bubble_info.time_in_bubble_pct);

    if cli.verbose && !bubble_info.bubble_periods.is_empty() {
        println!();
        println!("Bubble Periods:");
        for (i, period) in bubble_info.bubble_periods.iter().enumerate() {
            println!("  Period {}: steps {}-{}, peak at step {} ({:.1}% deviation)",
                i + 1, period.start, period.end, period.peak, period.peak_deviation * 100.0);
        }
    }
    println!();

    // Agent results
    println!("Agent Performance:");
    println!("─────────────────────────────────────────────────────────────");
    println!("{:<15} {:<15} {:>12} {:>10} {:>8}", "Agent", "Type", "Final Value", "Return", "Trades");
    println!("─────────────────────────────────────────────────────────────");

    let mut sorted_results: Vec<_> = result.agent_results.iter().collect();
    sorted_results.sort_by(|a, b| b.1.final_value.partial_cmp(&a.1.final_value).unwrap());

    for (id, agent_result) in sorted_results {
        println!(
            "{:<15} {:<15} ${:>10.0} {:>9.2}% {:>8}",
            id,
            agent_result.agent_type,
            agent_result.final_value,
            agent_result.return_pct,
            agent_result.num_trades
        );
    }
    println!("─────────────────────────────────────────────────────────────");
    println!();

    println!("Total Trades Executed: {}", result.total_trades);
    println!();
    println!("Simulation complete!");

    Ok(())
}
