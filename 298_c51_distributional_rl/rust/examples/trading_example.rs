use anyhow::Result;
use c51_distributional_rl::*;
use ndarray::Array1;

/// Generate synthetic price data for demonstration
fn generate_synthetic_prices(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut price = 50000.0; // Starting BTC-like price

    for i in 0..n {
        // Add trend component
        let trend = 0.0001 * (i as f64 / n as f64 - 0.5);
        // Add volatility
        let noise: f64 = rand::Rng::gen_range(&mut rng, -0.02..0.02);
        // Add occasional jumps (regime changes)
        let jump = if rand::Rng::gen::<f64>(&mut rng) < 0.02 {
            rand::Rng::gen_range(&mut rng, -0.05..0.05)
        } else {
            0.0
        };

        price *= 1.0 + trend + noise + jump;
        prices.push(price);
    }
    prices
}

/// Print a text-based histogram of a distribution
fn print_distribution(dist: &Array1<f64>, support: &Array1<f64>, label: &str) {
    println!("\n  Distribution for action: {}", label);
    println!("  {:-<60}", "");

    let max_prob = dist.iter().cloned().fold(0.0f64, f64::max);
    let bar_width = 40;

    // Sample every few atoms to keep output manageable
    let step = 5;
    for i in (0..dist.len()).step_by(step) {
        let bar_len = if max_prob > 0.0 {
            (dist[i] / max_prob * bar_width as f64) as usize
        } else {
            0
        };
        let bar: String = "#".repeat(bar_len);
        println!(
            "  {:>7.2} | {:<40} {:.4}",
            support[i], bar, dist[i]
        );
    }
    println!("  {:-<60}", "");
}

fn main() -> Result<()> {
    println!("=================================================================");
    println!("  C51 Distributional RL for Trading");
    println!("  Chapter 298 - Trading Example");
    println!("=================================================================");

    // =========================================================================
    // Step 1: Try to fetch data from Bybit, fall back to synthetic
    // =========================================================================
    println!("\n[1] Fetching market data...");

    let prices = match fetch_bybit_data() {
        Ok(p) if p.len() >= 100 => {
            println!("    Loaded {} candles from Bybit BTCUSDT", p.len());
            p
        }
        _ => {
            println!("    Using synthetic price data (1000 candles)");
            generate_synthetic_prices(1000)
        }
    };

    println!("    Price range: {:.2} - {:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // =========================================================================
    // Step 2: Initialize C51 Agent and Environment
    // =========================================================================
    println!("\n[2] Initializing C51 agent...");

    let window_size = 20;
    let state_size = 8;
    let mut env = TradingEnvironment::new(prices.clone(), window_size);
    let mut agent = C51Agent::new(state_size);

    println!("    Network: {} -> 128 -> 128 -> {} x {} atoms",
        state_size, NUM_ACTIONS, NUM_ATOMS);
    println!("    Support range: [{}, {}], dz = {:.4}", V_MIN, V_MAX, DZ);
    println!("    Gamma: {}, Epsilon: {} -> {}", agent.gamma, agent.epsilon, agent.epsilon_min);

    // =========================================================================
    // Step 3: Train C51 Agent
    // =========================================================================
    println!("\n[3] Training C51 agent...");

    let num_episodes = 20;
    let mut episode_rewards = Vec::new();

    for episode in 0..num_episodes {
        env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;

        while env.current_step < env.prices.len() - 1 {
            let state = env.get_state();
            let action = agent.select_action(&state);
            let (reward, done) = env.step(action);

            let next_state = env.get_state();
            agent.store_transition(Transition {
                state,
                action,
                reward,
                next_state,
                done,
            });

            // Train
            if let Some(loss) = agent.train_step() {
                if steps % 200 == 0 && episode % 5 == 0 {
                    println!("    Episode {}, Step {}: loss = {:.4}, epsilon = {:.4}",
                        episode, steps, loss, agent.epsilon);
                }
            }

            total_reward += reward;
            steps += 1;

            if done {
                break;
            }
        }

        episode_rewards.push(total_reward);

        if episode % 5 == 0 {
            let avg_reward = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
            println!("    Episode {}/{}: total_reward = {:.2}, avg = {:.2}, epsilon = {:.4}",
                episode + 1, num_episodes, total_reward, avg_reward, agent.epsilon);
        }
    }

    // =========================================================================
    // Step 4: Visualize Return Distributions
    // =========================================================================
    println!("\n[4] Visualizing return distributions for sample states...");

    let support = generate_support();
    env.reset();

    // Advance to a specific point
    for _ in 0..50 {
        env.step(2); // Hold
    }

    let state = env.get_state();
    let action_names = ["Buy", "Sell", "Hold"];

    for (action, name) in action_names.iter().enumerate() {
        let dist = agent.get_distribution(&state, action);
        print_distribution(&dist, &support, name);

        // Compute Q-value
        let q = dist.dot(&support);
        println!("    Q-value: {:.4}", q);

        // Compute risk metrics
        let var_5 = agent.compute_var(&state, action, 0.05);
        let cvar_5 = agent.compute_cvar(&state, action, 0.05);
        println!("    VaR(5%): {:.4}, CVaR(5%): {:.4}", var_5, cvar_5);
    }

    // =========================================================================
    // Step 5: Compare with Standard DQN
    // =========================================================================
    println!("\n[5] Comparing C51 vs Standard DQN...");

    let dqn = DQNNetwork::new(state_size, NUM_ACTIONS);

    println!("\n    Standard DQN outputs (point estimates):");
    let dqn_q = dqn.forward(&state);
    for (action, name) in action_names.iter().enumerate() {
        println!("      {}: Q = {:.4}", name, dqn_q[action]);
    }

    println!("\n    C51 outputs (full distributions):");
    let c51_q = agent.network.q_values(&state);
    for (action, name) in action_names.iter().enumerate() {
        let dist = agent.get_distribution(&state, action);
        let std_dev: f64 = dist
            .iter()
            .zip(support.iter())
            .map(|(p, z)| p * (z - c51_q[action]).powi(2))
            .sum::<f64>()
            .sqrt();
        println!("      {}: Q = {:.4}, StdDev = {:.4} (risk insight!)",
            name, c51_q[action], std_dev);
    }

    // =========================================================================
    // Step 6: Risk-Aware Action Selection
    // =========================================================================
    println!("\n[6] Risk-aware action selection...");

    println!("\n    Greedy (highest Q-value):");
    let greedy_action = agent.network.best_action(&state);
    println!("      Selected: {} (Q = {:.4})", action_names[greedy_action], c51_q[greedy_action]);

    println!("\n    Risk-averse (best CVaR at 5%):");
    let mut best_cvar_action = 0;
    let mut best_cvar = f64::NEG_INFINITY;
    for action in 0..NUM_ACTIONS {
        let cvar = agent.compute_cvar(&state, action, 0.05);
        if cvar > best_cvar {
            best_cvar = cvar;
            best_cvar_action = action;
        }
        println!("      {}: CVaR(5%) = {:.4}", action_names[action], cvar);
    }
    println!("      Selected: {} (CVaR = {:.4})", action_names[best_cvar_action], best_cvar);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=================================================================");
    println!("  Summary");
    println!("=================================================================");
    println!("  - Trained C51 agent for {} episodes", num_episodes);
    println!("  - Final epsilon: {:.4}", agent.epsilon);
    println!("  - Replay buffer size: {}", agent.replay_buffer.len());
    let final_avg = episode_rewards.iter().rev().take(5).sum::<f64>() / 5.0;
    println!("  - Avg reward (last 5 episodes): {:.2}", final_avg);
    println!("  - C51 advantage: full return distributions enable");
    println!("    risk-aware decisions via VaR/CVaR metrics");
    println!("=================================================================");

    Ok(())
}

/// Attempt to fetch live data from Bybit
fn fetch_bybit_data() -> Result<Vec<f64>> {
    let client = BybitClient::new();
    let klines = client.fetch_klines_blocking("BTCUSDT", "60", 1000)?;
    Ok(extract_close_prices(&klines))
}
