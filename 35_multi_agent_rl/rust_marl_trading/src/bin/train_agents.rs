//! Train agents using independent learning
//!
//! Usage:
//!   cargo run --bin train_agents -- --episodes 500 --agents 4

use anyhow::Result;
use rust_marl_trading::{
    agents::{
        Agent, MarketMakerAgent, MeanReversionAgent, NoiseTrader, RLAgent, TrendFollowingAgent,
    },
    algorithms::{IndependentLearners, TrainingConfig},
    environment::MultiAgentConfig,
};

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse arguments
    let args: Vec<String> = std::env::args().collect();

    let n_episodes: usize = args
        .iter()
        .position(|a| a == "--episodes")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

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
        .unwrap_or(500);

    println!("=== Independent Learners Training ===");
    println!("Episodes: {}", n_episodes);
    println!("Agents: {}", n_agents);
    println!("Max steps per episode: {}", max_steps);
    println!();

    // Create training config
    let training_config = TrainingConfig {
        n_episodes,
        max_steps,
        eval_frequency: 50,
        n_eval_episodes: 10,
        log_frequency: 10,
        early_stopping: true,
        patience: 20,
        ..Default::default()
    };

    // Create environment config
    let env_config = MultiAgentConfig {
        n_agents,
        initial_cash: 100_000.0,
        initial_price: 50_000.0,
        max_steps,
        volatility: 0.02,
        ..Default::default()
    };

    // Create agents: mix of RL and rule-based
    let mut agents: Vec<Box<dyn Agent>> = Vec::new();

    // Add one RL agent
    agents.push(Box::new(RLAgent::new(8, 64)));

    // Add rule-based opponents
    if n_agents > 1 {
        agents.push(Box::new(TrendFollowingAgent::default()));
    }
    if n_agents > 2 {
        agents.push(Box::new(MeanReversionAgent::default()));
    }
    if n_agents > 3 {
        agents.push(Box::new(NoiseTrader::default()));
    }

    // Fill remaining slots with noise traders
    while agents.len() < n_agents {
        agents.push(Box::new(NoiseTrader::default()));
    }

    // Print agent configuration
    println!("=== Agent Configuration ===");
    for (i, agent) in agents.iter().enumerate() {
        println!("Agent {}: {}", i, agent.name());
    }
    println!();

    // Create trainer
    let trainer = IndependentLearners::new(training_config.clone(), env_config.clone());

    // Train
    println!("=== Training Started ===");
    let result = trainer.train(&mut agents);

    // Print results
    println!();
    println!("=== Training Results ===");
    println!("Training time: {:.2} seconds", result.training_time);
    println!("Best score: {:.2} (episode {})", result.best_score, result.best_episode);
    println!("Converged: {}", result.converged);
    println!();

    // Run tournament
    println!("=== Post-Training Tournament ===");
    let tournament_stats = trainer.tournament(&mut agents, 20);

    println!("{:-<60}", "");
    println!(
        "{:<10} {:<15} {:>10} {:>10} {:>12}",
        "Agent", "Type", "Wins", "Games", "Avg PnL"
    );
    println!("{:-<60}", "");

    for (agent_id, stats) in &tournament_stats {
        let agent_name = agents[*agent_id].name();
        println!(
            "{:<10} {:<15} {:>10} {:>10} {:>12.2}",
            agent_id,
            agent_name,
            stats.wins,
            stats.games,
            stats.avg_pnl()
        );
    }
    println!("{:-<60}", "");

    // Find winner
    let winner = tournament_stats
        .iter()
        .max_by(|(_, a), (_, b)| a.avg_pnl().partial_cmp(&b.avg_pnl()).unwrap())
        .map(|(id, _)| *id);

    if let Some(winner_id) = winner {
        println!();
        println!(
            "🏆 Tournament winner: Agent {} ({})",
            winner_id,
            agents[winner_id].name()
        );
    }

    Ok(())
}
