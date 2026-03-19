//! # Curiosity-Driven Trading Example
//!
//! Demonstrates how curiosity-driven exploration improves a trading agent's
//! ability to discover novel market regimes and rare opportunities.
//!
//! Usage:
//!   cargo run --example trading_example
//!   cargo run --example trading_example -- --live   # Fetch live data from Bybit

use curiosity_driven_trading::*;

fn run_episode(
    agent: &mut CuriosityDrivenAgent,
    env: &mut TradingEnvironment,
    train: bool,
) -> (f64, f64, usize, Vec<f64>) {
    let mut state = env.reset().expect("Failed to reset environment");
    let mut total_reward = 0.0;
    let mut total_intrinsic = 0.0;
    let mut steps = 0;
    let mut intrinsic_history = Vec::new();

    while !env.is_done() {
        let action = agent.select_action(&state);
        let (next_state_opt, extrinsic_reward, done) = env.step(action);

        if let Some(ref next_state) = next_state_opt {
            if train {
                let (_total, intrinsic, _extrinsic) =
                    agent.train_step(&state, action, extrinsic_reward, next_state);
                total_intrinsic += intrinsic;
                intrinsic_history.push(intrinsic);
            }
            total_reward += extrinsic_reward;
            state = next_state.clone();
        }

        steps += 1;
        if done {
            break;
        }
    }

    (total_reward, total_intrinsic, steps, intrinsic_history)
}

fn print_intrinsic_reward_chart(history: &[f64], width: usize) {
    if history.is_empty() {
        return;
    }

    // Downsample to fit width
    let step = (history.len() as f64 / width as f64).max(1.0) as usize;
    let samples: Vec<f64> = history
        .chunks(step)
        .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
        .collect();

    let max_val = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_val = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = (max_val - min_val).max(1e-10);
    let height = 10;

    println!("\n  Intrinsic Reward Over Time:");
    println!("  {:.4} |", max_val);
    for row in 0..height {
        let threshold = max_val - (row as f64 + 0.5) * range / height as f64;
        let line: String = samples
            .iter()
            .map(|&v| if v >= threshold { '#' } else { ' ' })
            .collect();
        if row == height / 2 {
            println!("  {:.4} |{}", (max_val + min_val) / 2.0, line);
        } else {
            println!("         |{}", line);
        }
    }
    println!("  {:.4} |{}", min_val, "-".repeat(samples.len()));
    println!(
        "         {}{}",
        "0",
        " ".repeat(samples.len().saturating_sub(4).max(1))
            .to_string()
            + &format!("{}", history.len())
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_live_data = args.iter().any(|a| a == "--live");

    println!("=== Curiosity-Driven Trading ===\n");

    // ------------------------------------------------------------------
    // Step 1: Get market data
    // ------------------------------------------------------------------
    let candles = if use_live_data {
        println!("Fetching live BTCUSDT data from Bybit...");
        let client = BybitClient::new();
        match client.fetch_klines("BTCUSDT", "5", 200) {
            Ok(c) => {
                println!("Fetched {} candles from Bybit", c.len());
                println!(
                    "  Price range: {:.2} - {:.2}",
                    c.iter().map(|x| x.low).fold(f64::INFINITY, f64::min),
                    c.iter().map(|x| x.high).fold(f64::NEG_INFINITY, f64::max)
                );
                c
            }
            Err(e) => {
                println!("Failed to fetch from Bybit: {}. Using synthetic data.", e);
                generate_synthetic_candles(200, 3)
            }
        }
    } else {
        println!("Using synthetic market data with regime changes...");
        generate_synthetic_candles(200, 3)
    };

    println!("Total candles: {}\n", candles.len());

    // ------------------------------------------------------------------
    // Step 2: Train curiosity-driven agent
    // ------------------------------------------------------------------
    println!("--- Training Curiosity-Driven Agent (ICM + RND) ---");

    let curious_config = AgentConfig {
        state_dim: 8,
        feature_dim: 16,
        hidden_dim: 32,
        num_actions: 3,
        beta: 0.5,
        beta_decay: 0.995,
        learning_rate: 0.001,
        icm_eta: 0.1,
        icm_alpha: 0.5,
        use_icm: true,
        use_rnd: true,
        rnd_output_dim: 16,
        epsilon: 0.15,
    };

    let mut curious_agent = CuriosityDrivenAgent::new(curious_config);
    let mut env = TradingEnvironment::new(candles.clone());

    let num_episodes = 20;
    let mut curious_rewards = Vec::new();
    let mut last_intrinsic_history = Vec::new();

    for episode in 0..num_episodes {
        let (reward, intrinsic, steps, intrinsic_hist) =
            run_episode(&mut curious_agent, &mut env, true);
        curious_rewards.push(reward);

        if episode == num_episodes - 1 {
            last_intrinsic_history = intrinsic_hist;
        }

        if episode % 5 == 0 || episode == num_episodes - 1 {
            println!(
                "  Episode {:3}: PnL={:+.4}, Intrinsic={:.4}, Steps={}, Beta={:.4}",
                episode,
                reward,
                intrinsic,
                steps,
                curious_agent.current_beta()
            );
        }
    }

    // Print intrinsic reward chart for the last episode
    print_intrinsic_reward_chart(&last_intrinsic_history, 60);

    // ------------------------------------------------------------------
    // Step 3: Train non-curious baseline agent
    // ------------------------------------------------------------------
    println!("\n--- Training Non-Curious Baseline Agent ---");

    let baseline_config = AgentConfig {
        state_dim: 8,
        feature_dim: 16,
        hidden_dim: 32,
        num_actions: 3,
        beta: 0.0,  // No intrinsic reward
        beta_decay: 1.0,
        learning_rate: 0.001,
        icm_eta: 0.0,
        icm_alpha: 0.5,
        use_icm: false,
        use_rnd: false,
        rnd_output_dim: 16,
        epsilon: 0.15,
    };

    let mut baseline_agent = CuriosityDrivenAgent::new(baseline_config);
    let mut baseline_rewards = Vec::new();

    for episode in 0..num_episodes {
        let (reward, _intrinsic, steps, _) =
            run_episode(&mut baseline_agent, &mut env, true);
        baseline_rewards.push(reward);

        if episode % 5 == 0 || episode == num_episodes - 1 {
            println!(
                "  Episode {:3}: PnL={:+.4}, Steps={}",
                episode, reward, steps
            );
        }
    }

    // ------------------------------------------------------------------
    // Step 4: Compare results
    // ------------------------------------------------------------------
    println!("\n=== Comparison ===");

    let curious_avg: f64 = curious_rewards.iter().sum::<f64>() / curious_rewards.len() as f64;
    let baseline_avg: f64 = baseline_rewards.iter().sum::<f64>() / baseline_rewards.len() as f64;

    let curious_last5: f64 =
        curious_rewards.iter().rev().take(5).sum::<f64>() / 5.0;
    let baseline_last5: f64 =
        baseline_rewards.iter().rev().take(5).sum::<f64>() / 5.0;

    println!("Curious Agent:");
    println!("  Average PnL:  {:+.6}", curious_avg);
    println!("  Last 5 avg:   {:+.6}", curious_last5);
    println!(
        "  Total intrinsic reward: {:.4}",
        curious_agent.cumulative_intrinsic_reward
    );
    println!("  Total steps:  {}", curious_agent.total_steps);

    println!("Baseline Agent:");
    println!("  Average PnL:  {:+.6}", baseline_avg);
    println!("  Last 5 avg:   {:+.6}", baseline_last5);
    println!("  Total steps:  {}", baseline_agent.total_steps);

    // ------------------------------------------------------------------
    // Step 5: Analyze exploration patterns
    // ------------------------------------------------------------------
    println!("\n=== Exploration Analysis ===");

    // Use the curious agent to scan all states and find the most novel ones
    let mut novelty_scores: Vec<(usize, f64)> = Vec::new();
    for i in 5..candles.len() {
        if let Some(state) = candles_to_state(&candles, i) {
            let novelty = curious_agent.compute_intrinsic_reward(
                &state,
                TradingAction::Hold.to_index(),
                &state, // Self-transition for pure state novelty
            );
            novelty_scores.push((i, novelty));
        }
    }

    novelty_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 5 most novel market states (potential regime changes):");
    for (rank, (idx, score)) in novelty_scores.iter().take(5).enumerate() {
        let c = &candles[*idx];
        println!(
            "  {}. Index {:3}: close={:.2}, volume={:.1}, novelty={:.6}",
            rank + 1,
            idx,
            c.close,
            c.volume,
            score
        );
    }

    println!("\nTop 5 most familiar market states:");
    for (rank, (idx, score)) in novelty_scores.iter().rev().take(5).enumerate() {
        let c = &candles[*idx];
        println!(
            "  {}. Index {:3}: close={:.2}, volume={:.1}, novelty={:.6}",
            rank + 1,
            idx,
            c.close,
            c.volume,
            score
        );
    }

    println!("\n=== Done ===");
}
