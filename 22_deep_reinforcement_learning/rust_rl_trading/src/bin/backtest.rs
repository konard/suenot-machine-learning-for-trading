//! Backtest a trained agent on historical data.

use anyhow::Result;
use chrono::Utc;
use rust_rl_trading::{
    agent::{Agent, DQNAgent},
    data::{Candle, MarketData},
    environment::{EnvConfig, TradingEnvironment, TradingAction},
    utils::{AppConfig, PerformanceMetrics},
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

fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!("Usage: backtest <model_path> <data_path>");
        println!("Example: backtest models/best_agent.json data/BTCUSDT_60_365d.csv");
        return Ok(());
    }

    let model_path = &args[1];
    let data_path = &args[2];

    println!("Loading model from {}...", model_path);
    let agent = DQNAgent::load_full(model_path)?;

    println!("Loading data from {}...", data_path);
    let candles = load_candles_from_csv(data_path)?;
    println!("Loaded {} candles", candles.len());

    // Use last portion of data for testing (not seen during training)
    let test_size = candles.len().min(1000);
    let test_candles: Vec<Candle> = candles.into_iter().rev().take(test_size).rev().collect();

    let market_data = MarketData::from_candles(test_candles);

    // Create environment with same config
    let config = AppConfig::default();
    let env_config = EnvConfig {
        episode_length: market_data.len() - 100, // Use all available data
        trading_cost_bps: config.environment.trading_cost_bps,
        time_cost_bps: config.environment.time_cost_bps,
        initial_capital: config.environment.initial_capital,
        max_drawdown: 1.0, // Don't stop on drawdown during backtest
        reward_scale: config.environment.reward_scale,
    };

    let mut env = TradingEnvironment::new(market_data, env_config);

    println!("\nRunning backtest...");
    println!("Agent: {}", agent.name());

    // Run single episode without exploration
    let mut state = env.reset();
    let mut done = false;
    let mut actions_count = [0usize; 3];
    let mut nav_history = vec![config.environment.initial_capital];
    let mut returns = Vec::new();

    while !done {
        // Always use greedy policy (epsilon = 0)
        let action = agent.select_action(&state, 0.0);
        actions_count[action.to_index()] += 1;

        let result = env.step(action);

        nav_history.push(result.info.nav);
        returns.push(result.info.strategy_return);

        state = result.state;
        done = result.done;
    }

    // Calculate metrics
    let stats = env.get_episode_stats();
    let metrics = PerformanceMetrics::from_data(&returns, &nav_history, 252.0 * 24.0); // Hourly data

    println!("\n=== Backtest Results ===\n");
    println!("{}", stats);
    println!();
    println!("{}", metrics);

    println!("\nAction Distribution:");
    println!(
        "  SHORT: {:>6} ({:.1}%)",
        actions_count[0],
        actions_count[0] as f64 / returns.len() as f64 * 100.0
    );
    println!(
        "  HOLD:  {:>6} ({:.1}%)",
        actions_count[1],
        actions_count[1] as f64 / returns.len() as f64 * 100.0
    );
    println!(
        "  LONG:  {:>6} ({:.1}%)",
        actions_count[2],
        actions_count[2] as f64 / returns.len() as f64 * 100.0
    );

    // Compare to buy and hold
    println!("\n=== Comparison to Buy & Hold ===");
    println!("Strategy Return: {:>10.2}%", stats.total_return * 100.0);
    println!("Benchmark Return: {:>9.2}%", stats.benchmark_return * 100.0);
    let alpha = stats.total_return - stats.benchmark_return;
    println!(
        "Alpha: {:>18.2}% {}",
        alpha * 100.0,
        if alpha > 0.0 { "✓" } else { "✗" }
    );

    // Save results to CSV
    let results_path = "backtest_results.csv";
    let mut writer = csv::Writer::from_path(results_path)?;

    writer.write_record(&["step", "nav", "benchmark_nav", "return"])?;

    for (i, (nav, ret)) in nav_history.iter().zip(returns.iter()).enumerate() {
        writer.write_record(&[
            i.to_string(),
            nav.to_string(),
            (config.environment.initial_capital * (1.0 + stats.benchmark_return * (i as f64 / returns.len() as f64))).to_string(),
            ret.to_string(),
        ])?;
    }

    writer.flush()?;
    println!("\nDetailed results saved to {}", results_path);

    Ok(())
}
