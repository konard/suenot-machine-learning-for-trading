//! Basic Simulation Example
//!
//! Demonstrates a simple multi-agent market simulation.

use llm_market_sim::{
    agents::{ValueInvestor, MomentumTrader, MarketMaker},
    simulation::{SimulationEngine, calculate_performance_metrics},
};

fn main() {
    println!("LLM Market Simulation - Basic Example");
    println!("======================================");

    // Create simulation engine
    let mut engine = SimulationEngine::new(100.0, 100.0, 0.02);

    // Add value investors
    for i in 0..3 {
        engine.add_agent(Box::new(
            ValueInvestor::new(
                format!("value_{}", i + 1),
                100000.0,
                100,
                100.0,
            )
            .with_thresholds(0.05 + i as f64 * 0.02, 0.05 + i as f64 * 0.02)
        ));
    }

    // Add momentum traders
    for i in 0..3 {
        engine.add_agent(Box::new(
            MomentumTrader::new(
                format!("momentum_{}", i + 1),
                100000.0,
                100,
            )
            .with_windows(5 + i * 5, 20 + i * 10)
        ));
    }

    // Add market makers
    for i in 0..2 {
        engine.add_agent(Box::new(
            MarketMaker::new(
                format!("mm_{}", i + 1),
                200000.0,
                200,
            )
            .with_spread(0.002 + i as f64 * 0.001)
        ));
    }

    println!("\nRunning simulation for 500 steps...");

    // Run simulation
    let result = engine.run(500);

    // Print results
    println!("\nResults:");
    println!("  Final Price: ${:.2}", result.final_price);
    println!("  Total Trades: {}", result.total_trades);

    let metrics = calculate_performance_metrics(
        &result.price_history,
        Some(&result.fundamental_history),
        0.02,
        252.0,
    );

    println!("\nMetrics:");
    println!("  Total Return: {:.2}%", metrics.total_return_pct);
    println!("  Volatility: {:.2}%", metrics.volatility * 100.0);
    if let Some(sharpe) = metrics.sharpe_ratio {
        println!("  Sharpe Ratio: {:.3}", sharpe);
    }
    println!("  Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);

    println!("\nAgent Performance:");
    for (id, agent_result) in &result.agent_results {
        println!(
            "  {}: ${:.0f} ({:.2}%)",
            id, agent_result.final_value, agent_result.return_pct
        );
    }

    println!("\nDone!");
}
