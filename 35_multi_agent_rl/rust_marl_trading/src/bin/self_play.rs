//! Train an agent using self-play
//!
//! Usage:
//!   cargo run --bin self_play -- --episodes 300 --pool 5

use anyhow::Result;
use rust_marl_trading::{
    agents::{Agent, RLAgent},
    algorithms::{SelfPlayTrainer, TrainingConfig},
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
        .unwrap_or(300);

    let pool_size: usize = args
        .iter()
        .position(|a| a == "--pool")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let snapshot_freq: usize = args
        .iter()
        .position(|a| a == "--snapshot")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    println!("=== Self-Play Training ===");
    println!("Episodes: {}", n_episodes);
    println!("Opponent pool size: {}", pool_size);
    println!("Snapshot frequency: {}", snapshot_freq);
    println!();

    // Create training config
    let training_config = TrainingConfig {
        n_episodes,
        max_steps: 500,
        eval_frequency: 50,
        n_eval_episodes: 10,
        log_frequency: 10,
        ..Default::default()
    };

    // Create environment config (2 players for self-play)
    let env_config = MultiAgentConfig {
        n_agents: 2,
        initial_cash: 100_000.0,
        initial_price: 50_000.0,
        max_steps: 500,
        volatility: 0.02,
        ..Default::default()
    };

    // Create RL agent
    let mut agent: Box<dyn Agent> = Box::new(RLAgent::new(8, 64));

    println!("=== Agent Configuration ===");
    println!("Agent type: {}", agent.name());
    println!("Parameters: {}", agent.params_string());
    println!();

    // Create trainer
    let trainer = SelfPlayTrainer::new(training_config, env_config)
        .with_pool_size(pool_size)
        .with_snapshot_frequency(snapshot_freq);

    // Train
    println!("=== Training Started ===");
    println!("The agent will play against historical versions of itself.");
    println!("Harder opponents (lower win rate) are selected more often.");
    println!();

    let result = trainer.train(&mut agent);

    // Print results
    println!();
    println!("=== Training Results ===");
    println!("Training time: {:.2} seconds", result.training_time);
    println!("Best score: {:.2} (episode {})", result.best_score, result.best_episode);
    println!();

    // Show learning curve summary
    if !result.history.is_empty() {
        let first_10: f64 = result.history.iter().take(10).sum::<f64>() / 10.0;
        let last_10: f64 = result.history.iter().rev().take(10).sum::<f64>() / 10.0;

        println!("=== Learning Progress ===");
        println!("Average PnL (first 10 episodes): ${:.2}", first_10);
        println!("Average PnL (last 10 episodes): ${:.2}", last_10);
        println!("Improvement: {:+.2}%", (last_10 - first_10) / first_10.abs() * 100.0);
    }

    // Show evaluation history
    if !result.eval_history.is_empty() {
        println!();
        println!("=== Evaluation History ===");
        println!("{:-<60}", "");
        println!("{:>10} {:>15} {:>15} {:>15}", "Episode", "Mean Reward", "Win Rate", "Mean PnL");
        println!("{:-<60}", "");

        for eval in &result.eval_history {
            println!(
                "{:>10} {:>15.2} {:>14.1}% {:>15.2}",
                eval.episode,
                eval.mean_reward,
                eval.win_rate * 100.0,
                eval.mean_pnl
            );
        }
        println!("{:-<60}", "");
    }

    println!();
    println!("=== Summary ===");
    println!(
        "Self-play training completed. The agent learned to compete against {} historical versions of itself.",
        pool_size
    );
    println!("Final agent parameters: {}", agent.params_string());

    Ok(())
}
