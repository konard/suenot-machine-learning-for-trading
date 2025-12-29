//! Population-based training with tournament selection
//!
//! Usage:
//!   cargo run --bin population_tournament -- --population 20 --generations 50

use anyhow::Result;
use rust_marl_trading::{
    algorithms::{PopulationTrainer, TrainingConfig},
    environment::MultiAgentConfig,
};

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse arguments
    let args: Vec<String> = std::env::args().collect();

    let population_size: usize = args
        .iter()
        .position(|a| a == "--population")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let n_generations: usize = args
        .iter()
        .position(|a| a == "--generations")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let n_agents_per_game: usize = args
        .iter()
        .position(|a| a == "--agents-per-game")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    println!("=== Population-Based Training ===");
    println!("Population size: {}", population_size);
    println!("Generations: {}", n_generations);
    println!("Agents per game: {}", n_agents_per_game);
    println!();

    println!("=== Method Description ===");
    println!("1. Initialize random population with varied hyperparameters");
    println!("2. Each generation: agents compete in multi-agent games");
    println!("3. Fitness = average PnL across games");
    println!("4. Top 25% survive, bottom 75% replaced by mutated copies of top agents");
    println!("5. Repeat for {} generations", n_generations);
    println!();

    // Create training config
    let training_config = TrainingConfig {
        max_steps: 500,
        log_frequency: 5,
        ..Default::default()
    };

    // Create environment config
    let env_config = MultiAgentConfig {
        n_agents: n_agents_per_game,
        initial_cash: 100_000.0,
        initial_price: 50_000.0,
        max_steps: 500,
        volatility: 0.02,
        ..Default::default()
    };

    // Create trainer
    let trainer = PopulationTrainer::new(training_config, env_config)
        .with_population_size(population_size)
        .with_generations(n_generations);

    // Train
    println!("=== Evolution Started ===");
    let (result, best_member) = trainer.train();

    // Print results
    println!();
    println!("=== Evolution Results ===");
    println!("Training time: {:.2} seconds", result.training_time);
    println!(
        "Best fitness: {:.2} (generation {})",
        result.best_score, result.best_episode
    );
    println!();

    // Show best hyperparameters
    println!("=== Best Agent Hyperparameters ===");
    println!("Learning rate: {:.2e}", best_member.hyperparams.learning_rate);
    println!("Gamma (discount): {:.4}", best_member.hyperparams.gamma);
    println!("Epsilon: {:.4}", best_member.hyperparams.epsilon);
    println!("Hidden size: {}", best_member.hyperparams.hidden_size);
    println!("Generation: {}", best_member.generation);
    println!();

    // Show evolution curve
    if !result.history.is_empty() {
        println!("=== Evolution Curve ===");
        println!("{:-<50}", "");
        println!("{:>10} {:>15}", "Generation", "Best Fitness");
        println!("{:-<50}", "");

        let step = (result.history.len() / 10).max(1);
        for (i, &fitness) in result.history.iter().enumerate() {
            if i % step == 0 || i == result.history.len() - 1 {
                println!("{:>10} {:>15.2}", i, fitness);
            }
        }
        println!("{:-<50}", "");

        // Show improvement
        let first_fitness = result.history.first().copied().unwrap_or(0.0);
        let last_fitness = result.history.last().copied().unwrap_or(0.0);

        println!();
        println!("=== Summary ===");
        println!("Initial best fitness: {:.2}", first_fitness);
        println!("Final best fitness: {:.2}", last_fitness);

        if first_fitness != 0.0 {
            println!(
                "Improvement: {:+.1}%",
                (last_fitness - first_fitness) / first_fitness.abs() * 100.0
            );
        }
    }

    println!();
    println!("Population-based training completed successfully!");
    println!("The evolved hyperparameters can be used to train production agents.");

    Ok(())
}
