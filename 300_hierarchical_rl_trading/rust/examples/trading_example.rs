//! Hierarchical RL Trading Example
//!
//! Demonstrates:
//! 1. Fetching BTCUSDT data from Bybit
//! 2. Training a hierarchical (feudal) agent
//! 3. Showing manager goals and worker execution
//! 4. Comparing with a flat RL agent

use hierarchical_rl_trading::*;
use ndarray::Array1;

fn generate_synthetic_data(n: usize) -> (Vec<Array1<f64>>, Vec<f64>) {
    let mut prices = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    // Generate trending + mean-reverting price series
    let mut rng_state: u64 = 42;
    for i in 0..n {
        // Simple pseudo-random
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 200.0;

        // Add trend component
        let trend = if i < n / 3 {
            50.0 // Bull phase
        } else if i < 2 * n / 3 {
            -30.0 // Bear phase
        } else {
            5.0 // Sideways phase
        };

        price += trend + noise;
        price = price.max(10000.0);
        prices.push(price);
    }

    // Extract features from prices
    let lookback = 20;
    let mut features = Vec::new();
    let mut feat_prices = Vec::new();

    for i in lookback..prices.len() {
        let ret = (prices[i] / prices[i - 1]).ln();

        let returns: Vec<f64> = (i - lookback..i)
            .map(|j| (prices[j] / prices[j.saturating_sub(1).max(0)].max(1.0)).ln())
            .collect();
        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt();

        let sma5: f64 = prices[i - 5..i].iter().sum::<f64>() / 5.0;
        let sma20: f64 = prices[i - lookback..i].iter().sum::<f64>() / lookback as f64;
        let momentum = (sma5 - sma20) / sma20;

        // Synthetic volume ratio
        let vol_ratio = 1.0 + ret.abs() * 5.0;

        features.push(Array1::from_vec(vec![ret, volatility, momentum, vol_ratio]));
        feat_prices.push(prices[i]);
    }

    (features, feat_prices)
}

fn main() {
    println!("=== Chapter 300: Hierarchical RL Trading ===\n");

    // --------------------------------------------------
    // Step 1: Try to fetch data from Bybit, fall back to synthetic
    // --------------------------------------------------
    println!("Step 1: Loading market data...");

    let (features, prices) = match std::panic::catch_unwind(|| {
        let client = BybitClient::new();
        client.fetch_klines("BTCUSDT", "60", 200)
    }) {
        Ok(Ok(candles)) if candles.len() > 30 => {
            println!(
                "  Fetched {} candles from Bybit (BTCUSDT, 1h)",
                candles.len()
            );
            println!(
                "  Price range: {:.2} - {:.2}",
                candles.iter().map(|c| c.close).fold(f64::MAX, f64::min),
                candles.iter().map(|c| c.close).fold(f64::MIN, f64::max)
            );
            extract_features(&candles)
        }
        _ => {
            println!("  Bybit API unavailable, using synthetic data");
            generate_synthetic_data(200)
        }
    };

    println!("  Feature vectors: {}", features.len());
    println!("  Price points: {}\n", prices.len());

    // --------------------------------------------------
    // Step 2: Train Hierarchical Agent
    // --------------------------------------------------
    println!("Step 2: Training Hierarchical Agent (Feudal Network)...");
    println!("  Architecture: Manager (high-level) + Worker (low-level)");
    println!("  Manager decision interval: every 10 steps");
    println!("  Goal dimension: 4 (direction, magnitude, vol_tolerance, horizon)\n");

    let state_dim = 4;
    let hidden_dim = 32;
    let goal_dim = 4;
    let decision_interval = 10;
    let num_episodes = 20;

    let mut agent = HierarchicalAgent::new(state_dim, hidden_dim, goal_dim, decision_interval);

    let mut episode_rewards = Vec::new();
    for ep in 0..num_episodes {
        let (reward, log) = agent.run_episode(&features, &prices);
        let (manager_loss, worker_loss) = agent.train();
        episode_rewards.push(reward);

        if ep % 5 == 0 || ep == num_episodes - 1 {
            // Count regimes in this episode
            let bull_count = log.iter().filter(|(_, r, _, _)| *r == MarketRegime::Bull).count();
            let bear_count = log.iter().filter(|(_, r, _, _)| *r == MarketRegime::Bear).count();
            let side_count = log
                .iter()
                .filter(|(_, r, _, _)| *r == MarketRegime::Sideways)
                .count();

            println!(
                "  Episode {:3}: reward={:+.6}, mgr_loss={:.6}, wkr_loss={:.6}",
                ep + 1,
                reward,
                manager_loss,
                worker_loss
            );
            println!(
                "              regimes: Bull={}, Bear={}, Sideways={}",
                bull_count, bear_count, side_count
            );
        }
    }

    // --------------------------------------------------
    // Step 3: Show Manager Goals and Worker Execution
    // --------------------------------------------------
    println!("\nStep 3: Detailed Manager-Worker Interaction (last episode)...\n");

    agent.network.reset();
    let mut detailed_log = Vec::new();
    let mut position = 0.0;

    for i in 0..features.len().saturating_sub(1).min(60) {
        let state = &features[i];
        let was_at_boundary =
            agent.network.step_count % agent.network.manager.decision_interval == 0
                || agent.network.current_goal.is_none();

        let (action, regime) = agent.act(state);
        let goal = agent.network.current_goal.clone().unwrap();

        if was_at_boundary {
            println!(
                "  [Step {:3}] MANAGER DECISION: regime={:?}, goal=[{:.3}, {:.3}, {:.3}, {:.3}]",
                i, regime, goal[0], goal[1], goal[2], goal[3]
            );
        }

        let price_return = if prices[i] > 0.0 {
            (prices[i + 1] - prices[i]) / prices[i]
        } else {
            0.0
        };
        let step_pnl = position * price_return;
        position = action;

        detailed_log.push((i, regime, action, prices[i], step_pnl));

        if i < 30 || was_at_boundary {
            println!(
                "    Worker: action={:+.4} (position), price={:.2}, step_pnl={:+.6}",
                action, prices[i], step_pnl
            );
        }
    }

    // --------------------------------------------------
    // Step 4: Compare with Flat RL Agent
    // --------------------------------------------------
    println!("\nStep 4: Comparison with Flat RL Agent...\n");

    let flat_agent = FlatAgent::new(state_dim, hidden_dim);

    let mut flat_rewards = Vec::new();
    for _ in 0..num_episodes {
        let reward = flat_agent.run_episode(&features, &prices);
        flat_rewards.push(reward);
    }

    let hier_avg = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
    let hier_std = (episode_rewards
        .iter()
        .map(|r| (r - hier_avg).powi(2))
        .sum::<f64>()
        / episode_rewards.len() as f64)
        .sqrt();
    let hier_sharpe = if hier_std > 0.0 {
        hier_avg / hier_std
    } else {
        0.0
    };

    let flat_avg = flat_rewards.iter().sum::<f64>() / flat_rewards.len() as f64;
    let flat_std = (flat_rewards
        .iter()
        .map(|r| (r - flat_avg).powi(2))
        .sum::<f64>()
        / flat_rewards.len() as f64)
        .sqrt();
    let flat_sharpe = if flat_std > 0.0 {
        flat_avg / flat_std
    } else {
        0.0
    };

    println!("  Hierarchical Agent:");
    println!("    Avg Reward:  {:+.6}", hier_avg);
    println!("    Std Reward:  {:.6}", hier_std);
    println!("    Sharpe:      {:.4}", hier_sharpe);
    println!();
    println!("  Flat Agent:");
    println!("    Avg Reward:  {:+.6}", flat_avg);
    println!("    Std Reward:  {:.6}", flat_std);
    println!("    Sharpe:      {:.4}", flat_sharpe);
    println!();

    if hier_sharpe > flat_sharpe {
        println!("  Result: Hierarchical agent achieved better risk-adjusted returns.");
    } else {
        println!("  Result: Flat agent performed comparably (expected with limited training).");
        println!("  Note: Hierarchical advantage grows with more training episodes and");
        println!("        multi-regime market data where temporal abstraction matters.");
    }

    // --------------------------------------------------
    // Summary
    // --------------------------------------------------
    println!("\n=== Summary ===");
    println!("  - Feudal network: Manager sets regime + goal every {} steps", decision_interval);
    println!("  - Worker executes trades to achieve manager's goal at each step");
    println!("  - Intrinsic reward (cosine similarity) guides worker learning");
    println!("  - Total experiences collected: {}", agent.replay_buffer.len());
    println!("  - Episodes trained: {}", agent.episode_count);
    println!("\nDone!");
}
