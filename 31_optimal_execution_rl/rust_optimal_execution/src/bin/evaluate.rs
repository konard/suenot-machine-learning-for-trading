//! Оценка и сравнение стратегий
//!
//! Использование:
//! ```bash
//! cargo run --release --bin evaluate
//! cargo run --release --bin evaluate -- --model models/dqn_agent.json
//! ```

use anyhow::Result;
use clap::Parser;
use rust_optimal_execution::{
    agent::{Agent, DQNAgent, DQNConfig},
    baselines::{AlmgrenChrissExecutor, TWAPExecutor, VWAPExecutor},
    environment::{EnvConfig, ExecutionAction, ExecutionEnv},
    impact::ImpactParams,
    utils::{load_candles_csv, PerformanceStats, StrategyComparison},
};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(author, version, about = "Evaluate and compare execution strategies")]
struct Args {
    /// Path to data file (CSV)
    #[arg(long)]
    data: Option<PathBuf>,

    /// Path to trained model
    #[arg(long)]
    model: Option<PathBuf>,

    /// Number of evaluation episodes
    #[arg(long, default_value = "100")]
    episodes: usize,
}

fn evaluate_dqn(
    env: &mut ExecutionEnv,
    agent: &DQNAgent,
    episodes: usize,
) -> Vec<f64> {
    let mut shortfalls = Vec::with_capacity(episodes);

    for _ in 0..episodes {
        let mut state = env.reset();
        let mut done = false;

        while !done {
            let action = agent.select_action(&state, 0.0); // Greedy
            let result = env.step(action);
            state = result.state;
            done = result.done;
        }

        shortfalls.push(env.cumulative_shortfall());
    }

    shortfalls
}

fn evaluate_twap(env: &mut ExecutionEnv, episodes: usize) -> Vec<f64> {
    let config = env.config();
    let mut shortfalls = Vec::with_capacity(episodes);

    for _ in 0..episodes {
        let twap = TWAPExecutor::new(config.total_quantity, config.max_steps);
        let schedule = twap.generate_schedule();

        let _state = env.reset();
        let mut done = false;
        let mut step = 0;

        while !done && step < schedule.num_steps() {
            let fraction = schedule.fraction_at(step);
            let action = ExecutionAction::Continuous(fraction);
            let result = env.step(action);
            done = result.done;
            step += 1;
        }

        shortfalls.push(env.cumulative_shortfall());
    }

    shortfalls
}

fn evaluate_vwap(env: &mut ExecutionEnv, episodes: usize) -> Vec<f64> {
    let config = env.config();
    let mut shortfalls = Vec::with_capacity(episodes);

    for _ in 0..episodes {
        let vwap = VWAPExecutor::with_u_shape(config.total_quantity, config.max_steps);
        let schedule = vwap.generate_schedule();

        let _state = env.reset();
        let mut done = false;
        let mut step = 0;

        while !done && step < schedule.num_steps() {
            let fraction = schedule.fraction_at(step);
            let action = ExecutionAction::Continuous(fraction);
            let result = env.step(action);
            done = result.done;
            step += 1;
        }

        shortfalls.push(env.cumulative_shortfall());
    }

    shortfalls
}

fn evaluate_almgren_chriss(env: &mut ExecutionEnv, episodes: usize) -> Vec<f64> {
    let config = env.config();
    let params = ImpactParams::crypto_default();
    let mut shortfalls = Vec::with_capacity(episodes);

    for _ in 0..episodes {
        let ac = AlmgrenChrissExecutor::from_params(
            config.total_quantity,
            config.max_steps,
            config.risk_aversion,
            &params,
        );
        let schedule = ac.generate_schedule();

        let _state = env.reset();
        let mut done = false;
        let mut step = 0;

        while !done && step < schedule.num_steps() {
            let fraction = schedule.fraction_at(step);
            let action = ExecutionAction::Continuous(fraction);
            let result = env.step(action);
            done = result.done;
            step += 1;
        }

        shortfalls.push(env.cumulative_shortfall());
    }

    shortfalls
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    info!("Evaluating strategies with {} episodes", args.episodes);

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
        ExecutionEnv::from_candles(candles, env_config.clone())
    } else {
        info!("Using synthetic data");
        ExecutionEnv::synthetic(50000.0, 0.02, env_config.clone())
    };

    // Оценка TWAP
    info!("Evaluating TWAP...");
    let twap_shortfalls = evaluate_twap(&mut env, args.episodes);
    let twap_stats = PerformanceStats::from_shortfalls(&twap_shortfalls, &[], &[]);

    // Оценка VWAP
    info!("Evaluating VWAP...");
    let vwap_shortfalls = evaluate_vwap(&mut env, args.episodes);
    let vwap_stats = PerformanceStats::from_shortfalls(&vwap_shortfalls, &[], &twap_shortfalls);

    // Оценка Almgren-Chriss
    info!("Evaluating Almgren-Chriss...");
    let ac_shortfalls = evaluate_almgren_chriss(&mut env, args.episodes);
    let ac_stats = PerformanceStats::from_shortfalls(&ac_shortfalls, &[], &twap_shortfalls);

    // Оценка DQN (если модель есть)
    let dqn_stats = if let Some(model_path) = &args.model {
        info!("Loading DQN model from {:?}", model_path);
        let dqn_config = DQNConfig {
            state_dim: env.state_dim(),
            num_actions: env.action_dim(),
            ..Default::default()
        };
        let mut agent = DQNAgent::new(dqn_config);
        agent.load(model_path.to_str().unwrap())?;

        info!("Evaluating DQN...");
        let dqn_shortfalls = evaluate_dqn(&mut env, &agent, args.episodes);
        Some(PerformanceStats::from_shortfalls(
            &dqn_shortfalls,
            &[],
            &twap_shortfalls,
        ))
    } else {
        None
    };

    // Сравнение
    let mut comparison = StrategyComparison::default();
    comparison.add("TWAP", twap_stats);
    comparison.add("VWAP (U-shape)", vwap_stats);
    comparison.add("Almgren-Chriss", ac_stats);

    if let Some(stats) = dqn_stats {
        comparison.add("DQN Agent", stats);
    }

    println!("\n{}", comparison.report());

    if let Some(best) = comparison.best_strategy() {
        println!("\nBest strategy: {}", best);
    }

    Ok(())
}
