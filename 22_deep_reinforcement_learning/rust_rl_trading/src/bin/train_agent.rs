//! Train a DQN agent on cryptocurrency data.

use anyhow::Result;
use chrono::{Duration, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use rust_rl_trading::{
    agent::{Agent, DQNAgent},
    data::{BybitClient, Candle, Interval, MarketData},
    environment::{EnvConfig, TradingEnvironment},
    utils::AppConfig,
};
use std::env;

/// Load candles from CSV file
fn load_candles_from_csv(path: &str) -> Result<Vec<Candle>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut candles = Vec::new();

    for result in reader.records() {
        let record = result?;
        let candle = Candle::new(
            chrono::DateTime::parse_from_rfc3339(&record[0])?.with_timezone(&Utc),
            record[1].to_string(),
            record[2].parse()?,
            record[3].parse()?,
            record[4].parse()?,
            record[5].parse()?,
            record[6].parse()?,
            record[7].parse()?,
        );
        candles.push(candle);
    }

    Ok(candles)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    // Load configuration
    let config = AppConfig::from_env();
    let args: Vec<String> = env::args().collect();

    // Try to load from CSV first, otherwise fetch from API
    let candles = if let Some(csv_path) = args.get(1) {
        println!("Loading data from {}...", csv_path);
        load_candles_from_csv(csv_path)?
    } else {
        println!("Fetching data from Bybit...");
        let client = BybitClient::new();
        let end = Utc::now();
        let start = end - Duration::days(365);

        client
            .get_historical_klines(&config.bybit.symbol, Interval::Hour1, start, end)
            .await?
    };

    println!("Loaded {} candles", candles.len());

    if candles.len() < 500 {
        println!("Not enough data for training. Need at least 500 candles.");
        return Ok(());
    }

    // Create market data with indicators
    let market_data = MarketData::from_candles(candles);

    // Create environment
    let env_config = EnvConfig {
        episode_length: config.environment.episode_length,
        trading_cost_bps: config.environment.trading_cost_bps,
        time_cost_bps: config.environment.time_cost_bps,
        initial_capital: config.environment.initial_capital,
        max_drawdown: config.environment.max_drawdown,
        reward_scale: config.environment.reward_scale,
    };

    let mut env = TradingEnvironment::new(market_data, env_config);

    // Create DQN agent
    let dqn_config = rust_rl_trading::agent::dqn_agent::DQNConfig {
        learning_rate: config.training.learning_rate,
        gamma: config.training.gamma,
        epsilon_start: config.training.epsilon_start,
        epsilon_end: config.training.epsilon_end,
        epsilon_decay: config.training.epsilon_decay,
        buffer_size: config.training.buffer_size,
        batch_size: config.training.batch_size,
        target_update_freq: config.training.target_update_freq,
        hidden_layers: config.training.hidden_layers.clone(),
        double_dqn: config.training.double_dqn,
        ..Default::default()
    };

    let mut agent = DQNAgent::new(env.state_size(), env.action_size(), dqn_config);

    // Training loop
    println!("\nStarting training...");
    println!("Episodes: {}", config.training.num_episodes);
    println!("Agent: {}", agent.name());
    println!("State size: {}, Action size: {}", env.state_size(), env.action_size());
    println!();

    let pb = ProgressBar::new(config.training.num_episodes as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut best_return = f64::NEG_INFINITY;
    let mut episode_returns = Vec::new();
    let mut episode_sharpes = Vec::new();

    for episode in 0..config.training.num_episodes {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut done = false;

        while !done {
            // Select action
            let epsilon = agent.get_epsilon();
            let action = agent.select_action(&state, epsilon);

            // Take step
            let result = env.step(action);

            // Store experience
            agent.remember_transition(
                state.clone(),
                action,
                result.reward,
                result.state.clone(),
                result.done,
            );

            // Train
            if agent.can_train() {
                agent.train_step();
            }

            total_reward += result.reward;
            state = result.state;
            done = result.done;
        }

        // Decay epsilon
        agent.decay_epsilon();

        // Get episode stats
        let stats = env.get_episode_stats();
        episode_returns.push(stats.total_return);
        episode_sharpes.push(stats.sharpe_ratio);

        // Update progress bar
        pb.set_position(episode as u64 + 1);
        pb.set_message(format!(
            "Return: {:>6.2}% | Sharpe: {:>5.2} | ε: {:.3}",
            stats.total_return * 100.0,
            stats.sharpe_ratio,
            agent.get_epsilon()
        ));

        // Log periodically
        if (episode + 1) % config.training.log_freq == 0 {
            let recent_returns: Vec<f64> = episode_returns
                .iter()
                .rev()
                .take(config.training.log_freq)
                .copied()
                .collect();
            let avg_return =
                recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;

            log::info!(
                "Episode {} | Avg Return: {:.2}% | Epsilon: {:.3} | Buffer: {}",
                episode + 1,
                avg_return * 100.0,
                agent.get_epsilon(),
                agent.buffer_size()
            );
        }

        // Save best model
        if stats.total_return > best_return {
            best_return = stats.total_return;
            std::fs::create_dir_all("models")?;
            agent.save_full("models/best_agent.json")?;
        }

        // Save periodically
        if (episode + 1) % config.training.save_freq == 0 {
            std::fs::create_dir_all("models")?;
            agent.save_full(&format!("models/agent_ep{}.json", episode + 1))?;
        }
    }

    pb.finish_with_message("Training complete!");

    // Print summary
    println!("\n=== Training Summary ===");
    println!("Total episodes: {}", config.training.num_episodes);
    println!("Best return: {:.2}%", best_return * 100.0);

    let avg_return = episode_returns.iter().sum::<f64>() / episode_returns.len() as f64;
    let avg_sharpe = episode_sharpes.iter().sum::<f64>() / episode_sharpes.len() as f64;

    println!("Average return: {:.2}%", avg_return * 100.0);
    println!("Average Sharpe: {:.2}", avg_sharpe);

    // Save final model
    agent.save_full("models/final_agent.json")?;
    println!("\nFinal model saved to models/final_agent.json");
    println!("Best model saved to models/best_agent.json");

    Ok(())
}
