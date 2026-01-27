//! Basic Meta-RL example demonstrating core concepts.
//!
//! This example shows:
//! - Creating a Meta-RL agent
//! - Setting up trading environments
//! - Meta-training on multiple environments
//! - Adapting to a new environment via hidden state

use meta_rl_trading::{
    agent::meta_rl::MetaRLAgent,
    data::bybit::SimulatedDataGenerator,
    data::features::FeatureGenerator,
    env::trading_env::TradingEnvironment,
    trainer::ppo::PPOTrainer,
};

fn main() {
    println!("=== Basic Meta-RL for Trading Example ===\n");

    // Step 1: Create environments (different MDPs)
    println!("Step 1: Creating trading environments...");
    let feature_gen = FeatureGenerator::default_window();

    let regimes = [
        ("Bull Market", 0.015, 0.0003),
        ("Bear Market", 0.020, -0.0003),
        ("Sideways", 0.008, 0.0),
        ("High Volatility", 0.030, 0.0),
    ];

    let mut environments = Vec::new();

    for (name, vol, trend) in &regimes {
        let klines = SimulatedDataGenerator::generate_trending_klines(300, 50000.0, *vol, *trend);
        let features = feature_gen.compute_features(&klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let offset = closes.len() - features.len();
        let prices: Vec<f64> = closes[offset..].to_vec();

        let env = TradingEnvironment::new(
            prices,
            features.clone(),
            10000.0,
            0.001,
            50,
        );
        environments.push(env);
        println!("  {} environment: {} steps", name, features.len());
    }
    println!();

    // Step 2: Create Meta-RL agent
    println!("Step 2: Creating Meta-RL agent...");
    let state_dim = environments[0].state_dim();
    let action_dim = environments[0].action_dim();
    let hidden_size = 32;

    let agent = MetaRLAgent::new(state_dim, action_dim, hidden_size);
    println!("  State dim: {}", state_dim);
    println!("  Action dim: {}", action_dim);
    println!("  Hidden size: {}", hidden_size);
    println!("  Parameters: {}\n", agent.num_parameters());

    // Step 3: Create PPO trainer
    println!("Step 3: Setting up PPO meta-trainer...");
    let mut trainer = PPOTrainer::new(
        agent,
        0.001,  // learning rate
        0.99,   // gamma
        0.95,   // gae_lambda
        2,      // episodes per trial
    );
    println!("  Learning rate: 0.001");
    println!("  Episodes per trial: 2\n");

    // Step 4: Meta-training
    println!("Step 4: Meta-training on {} environments...", environments.len());
    let num_epochs = 5;

    for epoch in 0..num_epochs {
        let score = trainer.meta_train_step(&mut environments);
        println!("  Epoch {}: score = {:.6}", epoch + 1, score);
    }
    println!();

    // Step 5: Test adaptation
    println!("Step 5: Testing adaptation on new environment...");
    let new_klines = SimulatedDataGenerator::generate_klines(200, 45000.0, 0.025);
    let new_features = feature_gen.compute_features(&new_klines);
    let new_closes: Vec<f64> = new_klines.iter().map(|k| k.close).collect();
    let new_offset = new_closes.len() - new_features.len();
    let new_prices: Vec<f64> = new_closes[new_offset..].to_vec();

    let mut new_env = TradingEnvironment::new(
        new_prices,
        new_features,
        10000.0,
        0.001,
        50,
    );

    let agent = trainer.agent_mut();
    agent.reset_hidden();
    let mut prev_action = 0;
    let mut prev_reward = 0.0;
    let mut prev_done = false;

    for episode in 0..3 {
        let mut state = new_env.reset();
        let mut done = false;
        let mut total_reward = 0.0;
        let mut final_capital = 10000.0;

        while !done {
            let (action, _value) = agent.get_action(
                &state, prev_action, prev_reward, prev_done,
            );
            let result = new_env.step(action);
            total_reward += result.reward;
            final_capital = result.capital;
            prev_action = action;
            prev_reward = result.reward;
            prev_done = result.done;
            state = result.state;
            done = result.done;
        }

        println!("  Episode {}: reward = {:.4}, capital = ${:.2}",
                 episode + 1, total_reward, final_capital);
    }

    println!("\n=== Example Complete ===");
}
