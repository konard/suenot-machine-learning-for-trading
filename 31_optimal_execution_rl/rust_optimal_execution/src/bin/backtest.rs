//! Бэктестинг агента на исторических данных
//!
//! Использование:
//! ```bash
//! cargo run --release --bin backtest -- --model models/dqn_agent.json --data data/BTCUSDT_60_180d.csv
//! ```

use anyhow::Result;
use clap::Parser;
use rust_optimal_execution::{
    agent::{Agent, DQNAgent, DQNConfig},
    baselines::TWAPExecutor,
    environment::{EnvConfig, ExecutionAction, ExecutionEnv},
    utils::load_candles_csv,
};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(author, version, about = "Backtest trained agent on historical data")]
struct Args {
    /// Path to trained model
    #[arg(long)]
    model: PathBuf,

    /// Path to data file (CSV)
    #[arg(long)]
    data: PathBuf,

    /// Total quantity to execute
    #[arg(long, default_value = "1000")]
    quantity: f64,

    /// Maximum steps per execution
    #[arg(long, default_value = "60")]
    max_steps: usize,

    /// Number of backtest runs
    #[arg(long, default_value = "50")]
    runs: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    info!("Loading data from {:?}", args.data);
    let candles = load_candles_csv(&args.data)?;
    info!("Loaded {} candles", candles.len());

    let env_config = EnvConfig {
        total_quantity: args.quantity,
        max_steps: args.max_steps,
        num_actions: 11,
        ..Default::default()
    };

    let mut env = ExecutionEnv::from_candles(candles, env_config.clone());

    // Загружаем агента
    info!("Loading model from {:?}", args.model);
    let dqn_config = DQNConfig {
        state_dim: env.state_dim(),
        num_actions: env.action_dim(),
        ..Default::default()
    };
    let mut agent = DQNAgent::new(dqn_config);
    agent.load(args.model.to_str().unwrap())?;

    // Бэктест
    let mut dqn_shortfalls = Vec::new();
    let mut twap_shortfalls = Vec::new();
    let mut dqn_rewards = Vec::new();

    info!("Running {} backtest iterations...", args.runs);

    for run in 0..args.runs {
        // DQN Agent
        let mut state = env.reset();
        let mut done = false;
        let mut episode_reward = 0.0;

        while !done {
            let action = agent.select_action(&state, 0.0);
            let result = env.step(action);
            episode_reward += result.reward;
            state = result.state;
            done = result.done;
        }

        let dqn_shortfall = env.cumulative_shortfall();
        dqn_shortfalls.push(dqn_shortfall);
        dqn_rewards.push(episode_reward);

        // TWAP Baseline
        let twap = TWAPExecutor::new(env_config.total_quantity, env_config.max_steps);
        let schedule = twap.generate_schedule();

        let _state = env.reset();
        done = false;
        let mut step = 0;

        while !done && step < schedule.num_steps() {
            let fraction = schedule.fraction_at(step);
            let action = ExecutionAction::Continuous(fraction);
            let result = env.step(action);
            done = result.done;
            step += 1;
        }

        let twap_shortfall = env.cumulative_shortfall();
        twap_shortfalls.push(twap_shortfall);

        if (run + 1) % 10 == 0 {
            info!("Completed {} runs", run + 1);
        }
    }

    // Результаты
    println!("\n=== Backtest Results ===\n");

    let dqn_mean = dqn_shortfalls.iter().sum::<f64>() / args.runs as f64;
    let twap_mean = twap_shortfalls.iter().sum::<f64>() / args.runs as f64;

    let dqn_std = (dqn_shortfalls.iter()
        .map(|s| (s - dqn_mean).powi(2))
        .sum::<f64>() / args.runs as f64).sqrt();

    let twap_std = (twap_shortfalls.iter()
        .map(|s| (s - twap_mean).powi(2))
        .sum::<f64>() / args.runs as f64).sqrt();

    let wins = dqn_shortfalls.iter()
        .zip(twap_shortfalls.iter())
        .filter(|(d, t)| d < t)
        .count();

    let improvement = (twap_mean - dqn_mean) / twap_mean.abs() * 100.0;

    println!("DQN Agent:");
    println!("  Mean Shortfall: ${:.4} ({:.4} bps)", dqn_mean, dqn_mean / (args.quantity * 50000.0) * 10000.0);
    println!("  Std Shortfall:  ${:.4}", dqn_std);
    println!("  Mean Reward:    {:.4}", dqn_rewards.iter().sum::<f64>() / args.runs as f64);

    println!("\nTWAP Baseline:");
    println!("  Mean Shortfall: ${:.4} ({:.4} bps)", twap_mean, twap_mean / (args.quantity * 50000.0) * 10000.0);
    println!("  Std Shortfall:  ${:.4}", twap_std);

    println!("\nComparison:");
    println!("  Win Rate:    {:.1}% ({}/{})", wins as f64 / args.runs as f64 * 100.0, wins, args.runs);
    println!("  Improvement: {:.2}%", improvement);

    if improvement > 0.0 {
        println!("\n✓ DQN Agent outperforms TWAP by {:.2}%", improvement);
    } else {
        println!("\n✗ TWAP outperforms DQN Agent by {:.2}%", -improvement);
    }

    Ok(())
}
