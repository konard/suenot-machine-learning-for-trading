//! Обучение RL агента
//!
//! Использование:
//! ```bash
//! cargo run --release --bin train_agent
//! cargo run --release --bin train_agent -- --data data/BTCUSDT_60_180d.csv
//! ```

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rust_optimal_execution::{
    agent::{Agent, DQNAgent, DQNConfig},
    baselines::TWAPExecutor,
    environment::{EnvConfig, ExecutionAction, ExecutionEnv},
    utils::{load_candles_csv, PerformanceStats},
};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train RL agent for optimal execution")]
struct Args {
    /// Path to data file (CSV)
    #[arg(long)]
    data: Option<PathBuf>,

    /// Number of episodes
    #[arg(long, default_value = "1000")]
    episodes: usize,

    /// Batch size
    #[arg(long, default_value = "64")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    learning_rate: f64,

    /// Output path for model
    #[arg(long, default_value = "models/dqn_agent.json")]
    output: PathBuf,

    /// Evaluate every N episodes
    #[arg(long, default_value = "100")]
    eval_freq: usize,

    /// Use synthetic data if no data file provided
    #[arg(long)]
    synthetic: bool,
}

fn run_episode(env: &mut ExecutionEnv, agent: &dyn Agent, epsilon: f64) -> (f64, f64) {
    let mut state = env.reset();
    let mut total_reward = 0.0;
    let mut done = false;

    while !done {
        let action = agent.select_action(&state, epsilon);
        let result = env.step(action);

        total_reward += result.reward;
        state = result.state;
        done = result.done;
    }

    let shortfall = env.cumulative_shortfall();
    (total_reward, shortfall)
}

fn run_training_episode(
    env: &mut ExecutionEnv,
    agent: &mut DQNAgent,
    epsilon: f64,
) -> (f64, f64, f64) {
    let mut state = env.reset();
    let mut total_reward = 0.0;
    let mut total_loss = 0.0;
    let mut train_steps = 0;
    let mut done = false;

    while !done {
        let action = agent.select_action(&state, epsilon);
        let result = env.step(action);

        agent.remember(
            state.clone(),
            action,
            result.reward,
            result.state.clone(),
            result.done,
        );

        if agent.can_train() {
            let loss = agent.train_step();
            total_loss += loss;
            train_steps += 1;
        }

        total_reward += result.reward;
        state = result.state;
        done = result.done;
    }

    let avg_loss = if train_steps > 0 {
        total_loss / train_steps as f64
    } else {
        0.0
    };

    let shortfall = env.cumulative_shortfall();
    (total_reward, shortfall, avg_loss)
}

fn run_twap_baseline(env: &mut ExecutionEnv) -> f64 {
    let config = env.config();
    let twap = TWAPExecutor::new(config.total_quantity, config.max_steps);
    let schedule = twap.generate_schedule();

    let mut state = env.reset();
    let mut done = false;
    let mut step = 0;

    while !done && step < schedule.num_steps() {
        let fraction = schedule.fraction_at(step);
        let action = ExecutionAction::Continuous(fraction);
        let result = env.step(action);
        state = result.state;
        done = result.done;
        step += 1;
    }

    env.cumulative_shortfall()
}

#[tokio::main]
async fn main() -> Result<()> {
    // Инициализация логирования
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    info!("Starting training with {} episodes", args.episodes);

    // Создаём среду
    let env_config = EnvConfig {
        total_quantity: 1000.0,
        max_steps: 60,
        num_actions: 11,
        ..Default::default()
    };

    let mut env = if let Some(data_path) = &args.data {
        info!("Loading data from {:?}", data_path);
        let candles = load_candles_csv(data_path)?;
        info!("Loaded {} candles", candles.len());
        ExecutionEnv::from_candles(candles, env_config.clone())
    } else if args.synthetic {
        info!("Using synthetic data");
        ExecutionEnv::synthetic(50000.0, 0.02, env_config.clone())
    } else {
        info!("No data provided, using synthetic data");
        ExecutionEnv::synthetic(50000.0, 0.02, env_config.clone())
    };

    // Создаём агента
    let dqn_config = DQNConfig {
        state_dim: env.state_dim(),
        num_actions: env.action_dim(),
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        ..Default::default()
    };

    let mut agent = DQNAgent::new(dqn_config);

    // Progress bar
    let pb = ProgressBar::new(args.episodes as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    // Сбор метрик
    let mut rewards = Vec::new();
    let mut shortfalls = Vec::new();
    let mut baseline_shortfalls = Vec::new();
    let mut losses = Vec::new();

    // Обучение
    for episode in 0..args.episodes {
        let (reward, shortfall, loss) = run_training_episode(&mut env, &mut agent, agent.get_epsilon());

        rewards.push(reward);
        shortfalls.push(shortfall);
        losses.push(loss);

        // Baseline для сравнения
        let twap_shortfall = run_twap_baseline(&mut env);
        baseline_shortfalls.push(twap_shortfall);

        agent.decay_epsilon();
        pb.inc(1);

        // Периодическая оценка
        if (episode + 1) % args.eval_freq == 0 {
            let recent_rewards: Vec<_> = rewards.iter().rev().take(args.eval_freq).copied().collect();
            let recent_shortfalls: Vec<_> = shortfalls.iter().rev().take(args.eval_freq).copied().collect();
            let recent_baseline: Vec<_> = baseline_shortfalls.iter().rev().take(args.eval_freq).copied().collect();

            let stats = PerformanceStats::from_shortfalls(
                &recent_shortfalls,
                &recent_rewards,
                &recent_baseline,
            );

            pb.println(format!(
                "Episode {}: Avg Reward: {:.4}, Avg IS: {:.6} bps, Win Rate: {:.1}%, Epsilon: {:.4}",
                episode + 1,
                stats.mean_reward,
                stats.mean_shortfall * 10000.0,
                stats.win_rate * 100.0,
                agent.get_epsilon(),
            ));
        }
    }

    pb.finish_with_message("Training complete!");

    // Финальная статистика
    let final_stats = PerformanceStats::from_shortfalls(&shortfalls, &rewards, &baseline_shortfalls);
    info!("\n{}", final_stats.report());

    // Сохраняем модель
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    agent.save(args.output.to_str().unwrap())?;
    info!("Model saved to {:?}", args.output);

    Ok(())
}
