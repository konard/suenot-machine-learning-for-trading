//! Simulate a multi-agent market
//!
//! Usage:
//!   cargo run --bin simulate_market -- --agents 4 --steps 1000

use anyhow::Result;
use rust_marl_trading::{
    agents::{
        Agent, MarketMakerAgent, MeanReversionAgent, NoiseTrader, TrendFollowingAgent,
    },
    environment::{MultiAgentConfig, MultiAgentEnv},
};

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse arguments
    let args: Vec<String> = std::env::args().collect();

    let n_agents: usize = args
        .iter()
        .position(|a| a == "--agents")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let max_steps: usize = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let initial_price: f64 = args
        .iter()
        .position(|a| a == "--price")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50000.0);

    println!("=== Multi-Agent Market Simulation ===");
    println!("Number of agents: {}", n_agents);
    println!("Max steps: {}", max_steps);
    println!("Initial price: ${:.2}", initial_price);
    println!();

    // Create environment
    let config = MultiAgentConfig {
        n_agents,
        initial_cash: 100_000.0,
        initial_price,
        max_steps,
        volatility: 0.02,
        mean_reversion: 0.1,
        fundamental_value: initial_price,
        transaction_cost: 0.001,
        price_impact: 0.0001,
        ..Default::default()
    };

    let mut env = MultiAgentEnv::new(config);

    // Create diverse agents
    let mut agents: Vec<Box<dyn Agent>> = Vec::new();

    // Add different agent types
    let agent_types = [
        "TrendFollowing",
        "MeanReversion",
        "NoiseTrader",
        "MarketMaker",
    ];

    for i in 0..n_agents {
        let agent_type = agent_types[i % agent_types.len()];
        let agent: Box<dyn Agent> = match agent_type {
            "TrendFollowing" => Box::new(TrendFollowingAgent::default()),
            "MeanReversion" => Box::new(MeanReversionAgent::default()),
            "NoiseTrader" => Box::new(NoiseTrader::default()),
            "MarketMaker" => Box::new(MarketMakerAgent::default()),
            _ => Box::new(NoiseTrader::default()),
        };
        agents.push(agent);
    }

    // Print agent configuration
    println!("=== Agent Configuration ===");
    for (i, agent) in agents.iter().enumerate() {
        println!(
            "Agent {}: {} ({})",
            i,
            agent.name(),
            agent.params_string()
        );
    }
    println!();

    // Run simulation
    println!("=== Running Simulation ===");
    let result = env.run_episode(&mut agents, Some(max_steps));

    // Print results
    println!();
    println!("=== Simulation Results ===");
    println!("Steps completed: {}", result.n_steps);
    println!();

    // Print agent results
    println!("{:-<70}", "");
    println!(
        "{:<10} {:<15} {:>15} {:>15} {:>10}",
        "Agent", "Type", "Final Value", "PnL", "Trades"
    );
    println!("{:-<70}", "");

    let mut rankings: Vec<_> = (0..n_agents).collect();
    rankings.sort_by(|&a, &b| {
        let pnl_a = result.total_pnl.get(&a).unwrap_or(&0.0);
        let pnl_b = result.total_pnl.get(&b).unwrap_or(&0.0);
        pnl_b.partial_cmp(pnl_a).unwrap()
    });

    for (rank, &agent_id) in rankings.iter().enumerate() {
        let final_value = result.final_values.get(&agent_id).unwrap_or(&0.0);
        let pnl = result.total_pnl.get(&agent_id).unwrap_or(&0.0);
        let trades = result.trade_counts.get(&agent_id).unwrap_or(&0);
        let agent_name = agents[agent_id].name();

        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        println!(
            "{} {:<8} {:<15} {:>15.2} {:>+15.2} {:>10}",
            medal, agent_id, agent_name, final_value, pnl, trades
        );
    }
    println!("{:-<70}", "");

    // Market statistics
    println!();
    println!("=== Market Statistics ===");
    let summary = result.summary();
    println!("Final price: ${:.2}", summary.final_price);
    println!("Market volatility: {:.4}%", summary.market_volatility * 100.0);
    println!("Total trades: {}", summary.total_trades);
    println!("Mean PnL: ${:.2}", summary.mean_pnl);
    println!("Best PnL: ${:.2}", summary.best_pnl);
    println!("Worst PnL: ${:.2}", summary.worst_pnl);

    // Price history summary
    if !result.price_history.is_empty() {
        let first_price = result.price_history.first().unwrap();
        let last_price = result.price_history.last().unwrap();
        let max_price = result
            .price_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_price = result
            .price_history
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        println!();
        println!("=== Price History ===");
        println!("Starting price: ${:.2}", first_price);
        println!("Ending price: ${:.2}", last_price);
        println!("High: ${:.2}", max_price);
        println!("Low: ${:.2}", min_price);
        println!(
            "Price change: {:+.2}%",
            (last_price - first_price) / first_price * 100.0
        );
    }

    Ok(())
}
