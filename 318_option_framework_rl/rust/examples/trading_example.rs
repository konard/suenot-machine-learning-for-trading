//! # Option Framework RL — Trading Example
//!
//! Demonstrates hierarchical RL with the Options Framework applied to trading:
//! 1. Fetches BTCUSDT kline data from Bybit (falls back to synthetic data)
//! 2. Defines trading options for different market regimes
//! 3. Trains an option-critic agent
//! 4. Shows option selection patterns over time

use option_framework_rl::*;
use rand::Rng;
use rand::SeedableRng;

fn main() -> anyhow::Result<()> {
    println!("=== Option Framework RL for Trading ===\n");

    // -----------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // -----------------------------------------------------------------
    println!("[1] Fetching BTCUSDT data from Bybit...");
    let (prices, volumes) = match fetch_bybit_klines("BTCUSDT", "60", 500) {
        Ok(klines) if !klines.is_empty() => {
            println!("    Fetched {} klines from Bybit", klines.len());
            klines_to_series(&klines)
        }
        Ok(_) | Err(_) => {
            println!("    Using synthetic data (Bybit unavailable)");
            generate_synthetic_prices(500, 42)
        }
    };
    println!(
        "    Price range: {:.2} - {:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!("    Data points: {}\n", prices.len());

    // -----------------------------------------------------------------
    // Step 2: Define trading options for different market regimes
    // -----------------------------------------------------------------
    println!("[2] Defining trading options (temporal abstractions)...");
    let num_states = MarketState::TOTAL_STATES;
    let options = vec![
        TradingOption::trend_follow(num_states),
        TradingOption::mean_revert(num_states),
        TradingOption::hold(num_states),
    ];

    for opt in &options {
        let active_count = opt.initiation_set.iter().filter(|&&v| v).count();
        println!(
            "    {:?}: active in {}/{} states, avg termination prob = {:.3}",
            opt.kind,
            active_count,
            num_states,
            opt.termination_prob.iter().sum::<f64>() / num_states as f64
        );
    }
    println!();

    // -----------------------------------------------------------------
    // Step 3: Train option-critic agent
    // -----------------------------------------------------------------
    println!("[3] Training Option-Critic agent...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut critic = OptionCritic::new(
        num_states,
        0.05,  // alpha_critic
        0.05,  // alpha_intra
        0.005, // alpha_term
        0.99,  // gamma
        0.2,   // epsilon
    );
    let mut stats = TrainingStats::new();

    let num_episodes = 200;
    let report_interval = 50;

    for episode in 0..num_episodes {
        let mut env = SemiMdpEnv::new(prices.clone(), volumes.clone(), 20);
        let reward = critic.train_episode(&mut env, &mut rng);
        stats.record_reward(reward);

        if (episode + 1) % report_interval == 0 {
            let avg = stats.mean_reward(report_interval);
            println!(
                "    Episode {}/{}: avg reward (last {}) = {:.6}",
                episode + 1,
                num_episodes,
                report_interval,
                avg
            );
        }
    }
    println!();

    // -----------------------------------------------------------------
    // Step 4: Show option selection over time (evaluation run)
    // -----------------------------------------------------------------
    println!("[4] Evaluation run — option selection over time...");
    let mut eval_env = SemiMdpEnv::new(prices.clone(), volumes.clone(), 20);
    let state = eval_env.reset();
    let mut state_idx = state.to_index();
    let mut current_option = critic.select_option(state_idx, &mut rng);
    stats.record_option(TradingOptionKind::from_index(current_option));

    let mut option_trace: Vec<(usize, TradingOptionKind, f64)> = Vec::new();
    let mut step_rewards: Vec<f64> = Vec::new();
    let mut equity_curve: Vec<f64> = vec![1.0];
    let mut equity = 1.0;

    let mut step_count = 0;
    loop {
        let action = critic.select_action(state_idx, current_option, &mut rng);
        let (next_state, reward, done) = eval_env.step(action);
        let next_state_idx = next_state.to_index();

        step_rewards.push(reward);
        equity += reward;
        equity_curve.push(equity);

        // Record option trace every 20 steps
        if step_count % 20 == 0 {
            option_trace.push((
                step_count,
                TradingOptionKind::from_index(current_option),
                eval_env.prices[eval_env.time.min(eval_env.prices.len() - 1)],
            ));
        }

        if done {
            break;
        }

        // Check termination
        let should_terminate =
            rng.gen::<f64>() < critic.termination_prob(next_state_idx, current_option);
        if should_terminate {
            current_option = critic.select_option(next_state_idx, &mut rng);
            stats.record_option(TradingOptionKind::from_index(current_option));
        }

        state_idx = next_state_idx;
        step_count += 1;
    }

    println!("\n    Option trace (sampled every 20 steps):");
    println!("    {:>6}  {:>15}  {:>12}", "Step", "Option", "Price");
    println!("    {}", "-".repeat(38));
    for (step, opt, price) in &option_trace {
        println!("    {:>6}  {:>15?}  {:>12.2}", step, opt, price);
    }

    // -----------------------------------------------------------------
    // Step 5: Performance summary
    // -----------------------------------------------------------------
    println!("\n[5] Performance Summary:");
    let sr = sharpe_ratio(&step_rewards, 252.0 * 24.0); // hourly data
    let dd = max_drawdown(&equity_curve);
    println!("    Total PnL:        {:.6}", eval_env.pnl);
    println!("    Sharpe Ratio:     {:.4}", sr);
    println!("    Max Drawdown:     {:.4}", dd);
    println!("    Total steps:      {}", step_count);
    println!("    Final equity:     {:.6}", equity);

    println!("\n    Option usage distribution:");
    for (key, count) in &stats.option_selections {
        println!("      {}: {} times", key, count);
    }

    // Show learned Q-values for a few representative states
    println!("\n    Learned Q_omega values (sample states):");
    let sample_states = vec![
        MarketState { momentum: 0, volatility: 1, volume: 1 }, // down
        MarketState { momentum: 2, volatility: 1, volume: 1 }, // neutral
        MarketState { momentum: 4, volatility: 1, volume: 1 }, // up
    ];
    for s in &sample_states {
        let idx = s.to_index();
        let q = &critic.q_omega[idx];
        println!(
            "    State(m={}, v={}, vol={}): TF={:.4}, MR={:.4}, Hold={:.4}",
            s.momentum, s.volatility, s.volume, q[0], q[1], q[2]
        );
    }

    // Show learned termination probabilities
    println!("\n    Learned termination probabilities (sample states):");
    for s in &sample_states {
        let idx = s.to_index();
        println!(
            "    State(m={}, v={}, vol={}): TF={:.3}, MR={:.3}, Hold={:.3}",
            s.momentum,
            s.volatility,
            s.volume,
            critic.termination_prob(idx, 0),
            critic.termination_prob(idx, 1),
            critic.termination_prob(idx, 2),
        );
    }

    println!("\n=== Done ===");
    Ok(())
}
