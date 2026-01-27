//! Multi-asset Meta-RL training example.
//!
//! Demonstrates training across multiple simulated assets
//! with different volatilities and trends.

use meta_rl_trading::{
    agent::meta_rl::MetaRLAgent,
    data::bybit::SimulatedDataGenerator,
    data::features::FeatureGenerator,
    env::trading_env::TradingEnvironment,
    trainer::ppo::PPOTrainer,
};

fn main() {
    println!("=== Multi-Asset Meta-RL Training ===\n");

    let feature_gen = FeatureGenerator::default_window();

    // Simulate different crypto assets
    let assets = [
        ("BTC-like",  50000.0, 0.020, 0.0001),
        ("ETH-like",  3000.0,  0.025, 0.0002),
        ("SOL-like",  100.0,   0.035, 0.0003),
        ("AVAX-like", 30.0,    0.030, -0.0001),
        ("DOT-like",  7.0,     0.028, 0.0),
    ];

    println!("Creating environments for {} assets:", assets.len());
    let mut environments = Vec::new();

    for (name, price, vol, trend) in &assets {
        let klines = SimulatedDataGenerator::generate_trending_klines(400, *price, *vol, *trend);
        let features = feature_gen.compute_features(&klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let offset = closes.len() - features.len();
        let prices: Vec<f64> = closes[offset..].to_vec();

        let env = TradingEnvironment::new(prices, features.clone(), 10000.0, 0.001, 80);
        environments.push(env);
        println!("  {}: base_price=${:.2}, vol={:.3}, trend={:.4}, steps={}",
                 name, price, vol, trend, features.len());
    }
    println!();

    // Create agent and trainer
    let state_dim = environments[0].state_dim();
    let action_dim = environments[0].action_dim();
    let agent = MetaRLAgent::new(state_dim, action_dim, 32);

    println!("Agent: {} parameters\n", agent.num_parameters());

    let mut trainer = PPOTrainer::new(agent, 0.001, 0.99, 0.95, 2);

    // Meta-training
    println!("Meta-training across all assets...");
    for epoch in 0..10 {
        let score = trainer.meta_train_step(&mut environments);
        if (epoch + 1) % 2 == 0 {
            println!("  Epoch {}: score = {:.6}", epoch + 1, score);
        }
    }

    // Test on unseen asset
    println!("\nTesting on unseen asset (LINK-like, $15, high vol)...");
    let test_klines = SimulatedDataGenerator::generate_regime_changing_klines(300, 15.0);
    let test_features = feature_gen.compute_features(&test_klines);
    let test_closes: Vec<f64> = test_klines.iter().map(|k| k.close).collect();
    let test_offset = test_closes.len() - test_features.len();
    let test_prices: Vec<f64> = test_closes[test_offset..].to_vec();

    let mut test_env = TradingEnvironment::new(
        test_prices, test_features, 10000.0, 0.001, 80
    );

    let agent = trainer.agent_mut();
    agent.reset_hidden();
    let mut prev_action = 0;
    let mut prev_reward = 0.0;
    let mut prev_done = false;

    for episode in 0..3 {
        let mut state = test_env.reset();
        let mut done = false;
        let mut total_reward = 0.0;
        let mut capital = 10000.0;

        while !done {
            let (action, _) = agent.get_action(&state, prev_action, prev_reward, prev_done);
            let result = test_env.step(action);
            total_reward += result.reward;
            capital = result.capital;
            prev_action = action;
            prev_reward = result.reward;
            prev_done = result.done;
            state = result.state;
            done = result.done;
        }

        println!("  Episode {}: reward={:.4}, capital=${:.2}", episode + 1, total_reward, capital);
    }

    println!("\n=== Multi-Asset Training Complete ===");
}
