//! Trading strategy example with backtesting.
//!
//! Demonstrates end-to-end Meta-RL workflow:
//! 1. Generate training environments
//! 2. Meta-train the agent
//! 3. Backtest on new data

use meta_rl_trading::{
    agent::meta_rl::MetaRLAgent,
    data::bybit::SimulatedDataGenerator,
    data::features::FeatureGenerator,
    env::trading_env::TradingEnvironment,
    trainer::ppo::PPOTrainer,
    backtest::engine::{BacktestEngine, BacktestConfig},
};

fn main() {
    println!("=== Meta-RL Trading Strategy with Backtest ===\n");

    let feature_gen = FeatureGenerator::default_window();

    // Phase 1: Create training environments
    println!("Phase 1: Creating training environments...");
    let regimes = [
        ("Bull",       0.015, 0.0003),
        ("Bear",       0.020, -0.0003),
        ("Sideways",   0.008, 0.0),
        ("Volatile",   0.030, 0.0),
    ];

    let mut environments = Vec::new();
    for (name, vol, trend) in &regimes {
        let klines = SimulatedDataGenerator::generate_trending_klines(400, 50000.0, *vol, *trend);
        let features = feature_gen.compute_features(&klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let offset = closes.len() - features.len();
        let prices: Vec<f64> = closes[offset..].to_vec();

        environments.push(TradingEnvironment::new(
            prices, features.clone(), 10000.0, 0.001, 80
        ));
        println!("  {}: {} steps", name, features.len());
    }

    // Phase 2: Meta-train
    println!("\nPhase 2: Meta-training...");
    let state_dim = environments[0].state_dim();
    let action_dim = environments[0].action_dim();
    let agent = MetaRLAgent::new(state_dim, action_dim, 32);
    let mut trainer = PPOTrainer::new(agent, 0.001, 0.99, 0.95, 2);

    for epoch in 0..10 {
        let score = trainer.meta_train_step(&mut environments);
        if (epoch + 1) % 5 == 0 {
            println!("  Epoch {}: score = {:.6}", epoch + 1, score);
        }
    }

    // Phase 3: Backtest
    println!("\nPhase 3: Running backtest on regime-changing data...");
    let test_klines = SimulatedDataGenerator::generate_regime_changing_klines(800, 50000.0);
    println!("  Test data: {} candles", test_klines.len());

    let config = BacktestConfig {
        initial_capital: 10000.0,
        transaction_cost: 0.001,
        adaptation_episodes: 2,
        episode_length: 200,
    };

    let engine = BacktestEngine::new(config);
    let mut agent = trainer.agent_mut().clone_agent();
    let results = engine.run(&mut agent, &test_klines);

    println!("\n{}", results.summary());

    // Compare with buy-and-hold
    let first_price = test_klines.first().map(|k| k.close).unwrap_or(1.0);
    let last_price = test_klines.last().map(|k| k.close).unwrap_or(1.0);
    let buy_hold_return = (last_price / first_price) - 1.0;

    println!("\n=== Comparison ===");
    println!("Meta-RL Strategy: {:+.2}%", results.total_return * 100.0);
    println!("Buy & Hold:       {:+.2}%", buy_hold_return * 100.0);

    let outperformance = results.total_return - buy_hold_return;
    if outperformance > 0.0 {
        println!("\nMeta-RL outperformed by {:.2}%", outperformance * 100.0);
    } else {
        println!("\nBuy & Hold outperformed by {:.2}%", -outperformance * 100.0);
    }

    println!("\n=== Trading Strategy Complete ===");
}
