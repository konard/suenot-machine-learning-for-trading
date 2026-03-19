use anyhow::Result;
use lob_deep_learning::*;
use ndarray::Array1;

/// Generate synthetic LOB snapshots with realistic dynamics
fn generate_synthetic_lob(n: usize) -> Vec<LOBSnapshot> {
    let mut rng = rand::thread_rng();
    let mut snapshots = Vec::with_capacity(n);
    let mut mid_price = 50000.0_f64; // Starting BTC-like price

    for i in 0..n {
        // Price dynamics: trend + noise + occasional jumps
        let trend = 0.0001 * (i as f64 / n as f64 - 0.5);
        let noise: f64 = rand::Rng::gen_range(&mut rng, -0.002..0.002);
        let jump = if rand::Rng::gen::<f64>(&mut rng) < 0.02 {
            rand::Rng::gen_range(&mut rng, -0.01..0.01)
        } else {
            0.0
        };
        mid_price *= 1.0 + trend + noise + jump;

        let base_spread = mid_price * 0.0002; // 2 bps spread
        let tick = mid_price * 0.00005;

        let mut bid_prices = Vec::with_capacity(NUM_LEVELS);
        let mut bid_volumes = Vec::with_capacity(NUM_LEVELS);
        let mut ask_prices = Vec::with_capacity(NUM_LEVELS);
        let mut ask_volumes = Vec::with_capacity(NUM_LEVELS);

        // Create volume imbalance that correlates with future price direction
        let imbalance_factor: f64 = rand::Rng::gen_range(&mut rng, -0.5..0.5);

        for level in 0..NUM_LEVELS {
            let offset = base_spread / 2.0 + tick * level as f64;
            bid_prices.push(mid_price - offset);
            ask_prices.push(mid_price + offset);

            // Volume with decay and imbalance
            let base_vol = 100.0 / (1.0 + level as f64 * 0.4);
            let bid_factor = 1.0 + imbalance_factor * 0.3;
            let ask_factor = 1.0 - imbalance_factor * 0.3;

            let bid_noise: f64 = rand::Rng::gen_range(&mut rng, 0.7..1.3);
            let ask_noise: f64 = rand::Rng::gen_range(&mut rng, 0.7..1.3);

            bid_volumes.push((base_vol * bid_factor * bid_noise).max(0.1));
            ask_volumes.push((base_vol * ask_factor * ask_noise).max(0.1));
        }

        snapshots.push(LOBSnapshot {
            timestamp: i as u64,
            bid_prices,
            bid_volumes,
            ask_prices,
            ask_volumes,
        });
    }

    snapshots
}

/// Print a text-based bar chart for class probabilities
fn print_prediction(probs: &Array1<f64>, label: &str) {
    let class_names = ["Down", "Stationary", "Up"];
    println!("  Prediction for: {}", label);
    for (i, name) in class_names.iter().enumerate() {
        let bar_len = (probs[i] * 40.0) as usize;
        let bar: String = "#".repeat(bar_len);
        println!("    {:>10} | {:<40} {:.4}", name, bar, probs[i]);
    }
}

fn main() -> Result<()> {
    println!("=================================================================");
    println!("  LOB Deep Learning for Trading");
    println!("  Chapter 261 - Trading Example");
    println!("=================================================================");

    // =========================================================================
    // Step 1: Load data (Bybit or synthetic)
    // =========================================================================
    println!("\n[1] Loading market data...");

    let snapshots = match fetch_bybit_lob() {
        Ok(snaps) if snaps.len() >= 100 => {
            println!("    Loaded {} LOB snapshots from Bybit data", snaps.len());
            snaps
        }
        _ => {
            println!("    Using synthetic LOB data (1000 snapshots)");
            generate_synthetic_lob(1000)
        }
    };

    let first_mid = snapshots[0].mid_price();
    let last_mid = snapshots[snapshots.len() - 1].mid_price();
    println!("    Snapshots: {}", snapshots.len());
    println!(
        "    Mid-price range: {:.2} - {:.2}",
        snapshots
            .iter()
            .map(|s| s.mid_price())
            .fold(f64::INFINITY, f64::min),
        snapshots
            .iter()
            .map(|s| s.mid_price())
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "    Price change: {:.2}%",
        (last_mid / first_mid - 1.0) * 100.0
    );

    // =========================================================================
    // Step 2: Generate labels
    // =========================================================================
    println!("\n[2] Generating labels...");

    let horizon = 5;
    let threshold = 0.0005; // 5 bps
    let labels = generate_labels(&snapshots, horizon, threshold);

    let up_count = labels.iter().filter(|(l, _)| *l == 2).count();
    let down_count = labels.iter().filter(|(l, _)| *l == 0).count();
    let stat_count = labels.iter().filter(|(l, _)| *l == 1).count();

    println!("    Horizon: {} steps, Threshold: {:.2} bps", horizon, threshold * 10000.0);
    println!(
        "    Label distribution: Up={} ({:.1}%), Down={} ({:.1}%), Stationary={} ({:.1}%)",
        up_count,
        up_count as f64 / labels.len() as f64 * 100.0,
        down_count,
        down_count as f64 / labels.len() as f64 * 100.0,
        stat_count,
        stat_count as f64 / labels.len() as f64 * 100.0,
    );

    // =========================================================================
    // Step 3: Analyze LOB features
    // =========================================================================
    println!("\n[3] Analyzing LOB features...");

    // Compute average volume imbalance
    let avg_imbalance: f64 = snapshots
        .iter()
        .map(|s| {
            let imb = s.volume_imbalances();
            if imb.is_empty() {
                0.0
            } else {
                imb[0] // Best level imbalance
            }
        })
        .sum::<f64>()
        / snapshots.len() as f64;

    let avg_spread: f64 = snapshots
        .iter()
        .map(|s| {
            let mid = s.mid_price();
            if mid > 0.0 {
                s.spread() / mid * 10000.0 // in bps
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / snapshots.len() as f64;

    println!("    Avg best-level volume imbalance: {:.4}", avg_imbalance);
    println!("    Avg spread: {:.2} bps", avg_spread);

    // OFI statistics
    let ofis: Vec<f64> = (1..snapshots.len())
        .map(|i| compute_ofi(&snapshots[i - 1], &snapshots[i]))
        .collect();
    let avg_ofi = ofis.iter().sum::<f64>() / ofis.len() as f64;
    println!("    Avg Order Flow Imbalance: {:.4}", avg_ofi);

    // =========================================================================
    // Step 4: Initialize and train LOB agent
    // =========================================================================
    println!("\n[4] Training LOB agent...");

    let mut env = LOBEnvironment::new(snapshots.clone());
    let state_size = env.state_size();
    let mut agent = LOBAgent::new(state_size);

    println!(
        "    Network: {} -> 256 -> 128 -> 64 -> {} classes",
        state_size, NUM_CLASSES
    );
    println!(
        "    Gamma: {}, Epsilon: {} -> {}",
        agent.gamma, agent.epsilon, agent.epsilon_min
    );

    let num_episodes = 20;
    let mut episode_rewards = Vec::new();
    let mut episode_trades = Vec::new();

    for episode in 0..num_episodes {
        env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;

        while env.current_step < env.snapshots.len() - 1 {
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

            if let Some(loss) = agent.train_step() {
                if steps % 200 == 0 && episode % 5 == 0 {
                    println!(
                        "    Episode {}, Step {}: loss = {:.4}, epsilon = {:.4}",
                        episode, steps, loss, agent.epsilon
                    );
                }
            }

            total_reward += reward;
            steps += 1;

            if done {
                break;
            }
        }

        episode_rewards.push(total_reward);
        episode_trades.push(env.num_trades);

        if episode % 5 == 0 {
            let avg_reward = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
            println!(
                "    Episode {}/{}: reward = {:.2}, avg = {:.2}, trades = {}, epsilon = {:.4}",
                episode + 1,
                num_episodes,
                total_reward,
                avg_reward,
                env.num_trades,
                agent.epsilon
            );
        }
    }

    // =========================================================================
    // Step 5: Evaluate predictions
    // =========================================================================
    println!("\n[5] Evaluating prediction quality...");

    let eval_start = snapshots.len() / 2;
    let mut predictions = Vec::new();
    let mut true_labels = Vec::new();

    for i in eval_start..snapshots.len().saturating_sub(1) {
        let curr = &snapshots[i];
        let prev = if i > 0 {
            Some(&snapshots[i - 1])
        } else {
            None
        };
        let state = curr.to_state(prev);
        let pred = agent.network.predict_direction(&state);
        predictions.push(pred);
        true_labels.push(labels[i].0);
    }

    let accuracy = compute_accuracy(&predictions, &true_labels);
    let f1_up = compute_f1(&predictions, &true_labels, 2);
    let f1_down = compute_f1(&predictions, &true_labels, 0);
    let f1_stat = compute_f1(&predictions, &true_labels, 1);

    println!("    Accuracy: {:.2}%", accuracy * 100.0);
    println!(
        "    F1 scores: Up={:.4}, Down={:.4}, Stationary={:.4}",
        f1_up, f1_down, f1_stat
    );

    // =========================================================================
    // Step 6: Visualize sample predictions
    // =========================================================================
    println!("\n[6] Sample predictions...");

    let sample_indices = [
        eval_start,
        eval_start + snapshots.len() / 8,
        eval_start + snapshots.len() / 4,
    ];

    let class_names = ["Down", "Stationary", "Up"];
    for &idx in &sample_indices {
        if idx >= snapshots.len() - 1 {
            continue;
        }
        let curr = &snapshots[idx];
        let prev = if idx > 0 {
            Some(&snapshots[idx - 1])
        } else {
            None
        };
        let state = curr.to_state(prev);
        let probs = agent.predict(&state);

        let label_str = format!(
            "t={}, mid={:.2}, true={}",
            idx,
            curr.mid_price(),
            class_names[labels[idx].0]
        );
        print_prediction(&probs, &label_str);

        let signal = agent.imbalance_signal(&state);
        println!("    Imbalance signal: {:.4} (>0 = buy, <0 = sell)\n", signal);
    }

    // =========================================================================
    // Step 7: Trading performance metrics
    // =========================================================================
    println!("[7] Trading performance metrics...");

    // Run a final evaluation episode
    env.reset();
    let mut trade_returns = Vec::new();
    let mut portfolio_value = 1.0;
    let mut values = vec![1.0];

    while env.current_step < env.snapshots.len() - 1 {
        let state = env.get_state();
        let action = agent.network.best_action(&state);
        let (reward, done) = env.step(action);

        let ret = reward / 100.0; // Un-scale
        trade_returns.push(ret);
        portfolio_value *= 1.0 + ret;
        values.push(portfolio_value);

        if done {
            break;
        }
    }

    let sharpe = compute_sharpe(&trade_returns);
    let max_dd = compute_max_drawdown(&trade_returns);

    println!("    Final portfolio value: {:.4}", portfolio_value);
    println!("    Total return: {:.2}%", (portfolio_value - 1.0) * 100.0);
    println!("    Sharpe ratio (annualized): {:.4}", sharpe);
    println!("    Maximum drawdown: {:.2}%", max_dd * 100.0);
    println!("    Total trades: {}", env.num_trades);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=================================================================");
    println!("  Summary");
    println!("=================================================================");
    println!("  - Trained LOB deep learning agent for {} episodes", num_episodes);
    println!("  - State size: {} features ({} raw LOB + {} derived)",
        STATE_SIZE, LOB_FEATURE_SIZE, DERIVED_FEATURES);
    println!("  - Final epsilon: {:.4}", agent.epsilon);
    println!("  - Replay buffer size: {}", agent.replay_buffer.len());
    let final_avg = episode_rewards.iter().rev().take(5).sum::<f64>() / 5.0;
    println!("  - Avg reward (last 5 episodes): {:.2}", final_avg);
    println!("  - Prediction accuracy: {:.2}%", accuracy * 100.0);
    println!("  - LOB deep learning captures order book microstructure");
    println!("    for informed trading via volume imbalance and OFI signals");
    println!("=================================================================");

    Ok(())
}

/// Attempt to fetch live data from Bybit and synthesize LOB snapshots
fn fetch_bybit_lob() -> Result<Vec<LOBSnapshot>> {
    let client = BybitClient::new();
    let klines = client.fetch_klines_blocking("BTCUSDT", "1", 1000)?;
    Ok(synthesize_lob_from_klines(&klines))
}
