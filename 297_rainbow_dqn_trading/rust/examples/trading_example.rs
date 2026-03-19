//! Rainbow DQN Trading Example
//!
//! Demonstrates:
//! - Fetching BTCUSDT data from Bybit
//! - Training a Rainbow DQN agent
//! - Ablation study: contribution of each component
//! - Comparison with vanilla DQN

use rainbow_dqn_trading::*;

fn run_training(
    config: RainbowConfig,
    candles: &[Candle],
    label: &str,
    episodes: usize,
) -> Vec<f64> {
    let mut agent = RainbowAgent::new(config);
    let mut episode_rewards = Vec::new();

    for ep in 0..episodes {
        let mut env = TradingEnvironment::new(candles.to_vec(), 0.001);
        let (reward, steps) = train_episode(&mut agent, &mut env);
        episode_rewards.push(reward);

        if (ep + 1) % 5 == 0 || ep == 0 {
            let avg_reward: f64 = episode_rewards
                .iter()
                .rev()
                .take(5)
                .sum::<f64>()
                / 5.0_f64.min(episode_rewards.len() as f64);
            println!(
                "[{}] Episode {}/{}: reward={:.6}, avg_reward={:.6}, steps={}",
                label,
                ep + 1,
                episodes,
                reward,
                avg_reward,
                steps
            );
        }
    }

    episode_rewards
}

fn main() {
    println!("=== Rainbow DQN for Trading ===\n");

    // Try to fetch real data from Bybit, fall back to synthetic
    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) if c.len() >= 50 => {
            println!(
                "Fetched {} candles from Bybit (BTCUSDT, 1h)",
                c.len()
            );
            c
        }
        Ok(c) => {
            println!(
                "Bybit returned only {} candles, using synthetic data",
                c.len()
            );
            generate_synthetic_candles(200, 42)
        }
        Err(e) => {
            println!("Could not fetch Bybit data ({}), using synthetic data", e);
            generate_synthetic_candles(200, 42)
        }
    };

    println!(
        "Price range: {:.2} - {:.2}\n",
        candles.iter().map(|c| c.low).fold(f64::MAX, f64::min),
        candles.iter().map(|c| c.high).fold(f64::MIN, f64::max)
    );

    let episodes = 20;

    // ─── Full Rainbow DQN ───
    println!("--- Training Full Rainbow DQN ---");
    let rainbow_config = RainbowConfig {
        state_dim: NUM_FEATURES,
        n_actions: 3,
        hidden_dim: 32,
        n_atoms: 21,
        v_min: -5.0,
        v_max: 5.0,
        gamma: 0.99,
        n_step: 3,
        learning_rate: 0.001,
        buffer_capacity: 5000,
        batch_size: 16,
        target_update_freq: 50,
        per_alpha: 0.6,
        per_beta_start: 0.4,
        use_double: true,
        use_dueling: true,
        use_per: true,
        use_nstep: true,
        use_distributional: true,
        use_noisy: true,
    };
    let rainbow_rewards = run_training(rainbow_config, &candles, "Rainbow", episodes);

    // ─── Vanilla DQN (all improvements disabled) ───
    println!("\n--- Training Vanilla DQN ---");
    let vanilla_config = RainbowConfig {
        state_dim: NUM_FEATURES,
        n_actions: 3,
        hidden_dim: 32,
        n_atoms: 21,
        v_min: -5.0,
        v_max: 5.0,
        gamma: 0.99,
        n_step: 1,
        learning_rate: 0.001,
        buffer_capacity: 5000,
        batch_size: 16,
        target_update_freq: 50,
        per_alpha: 0.6,
        per_beta_start: 0.4,
        use_double: false,
        use_dueling: false,
        use_per: false,
        use_nstep: false,
        use_distributional: false,
        use_noisy: false,
    };
    let vanilla_rewards = run_training(vanilla_config, &candles, "Vanilla", episodes);

    // ─── Ablation Studies ───
    println!("\n=== Ablation Study ===");
    println!("Disabling each component one at a time:\n");

    let ablation_configs = vec![
        ("No Double DQN", RainbowConfig {
            use_double: false,
            ..rainbow_config_base()
        }),
        ("No Dueling", RainbowConfig {
            use_dueling: false,
            ..rainbow_config_base()
        }),
        ("No PER", RainbowConfig {
            use_per: false,
            ..rainbow_config_base()
        }),
        ("No N-step", RainbowConfig {
            use_nstep: false,
            n_step: 1,
            ..rainbow_config_base()
        }),
        ("No C51", RainbowConfig {
            use_distributional: false,
            ..rainbow_config_base()
        }),
        ("No Noisy Nets", RainbowConfig {
            use_noisy: false,
            ..rainbow_config_base()
        }),
    ];

    let mut ablation_results: Vec<(&str, f64)> = Vec::new();

    for (label, config) in &ablation_configs {
        let rewards = run_training(config.clone(), &candles, label, episodes);
        let avg = rewards.iter().sum::<f64>() / rewards.len() as f64;
        ablation_results.push((label, avg));
    }

    // ─── Summary ───
    println!("\n=== Results Summary ===\n");

    let rainbow_avg = rainbow_rewards.iter().sum::<f64>() / rainbow_rewards.len() as f64;
    let vanilla_avg = vanilla_rewards.iter().sum::<f64>() / vanilla_rewards.len() as f64;

    println!("{:<20} Avg Reward", "Configuration");
    println!("{:-<40}", "");
    println!("{:<20} {:.6}", "Rainbow (full)", rainbow_avg);
    println!("{:<20} {:.6}", "Vanilla DQN", vanilla_avg);

    for (label, avg) in &ablation_results {
        let diff = rainbow_avg - avg;
        println!(
            "{:<20} {:.6} (delta: {:.6})",
            label, avg, diff
        );
    }

    println!("\n--- Component Importance (by performance drop when removed) ---\n");

    let mut importance: Vec<(&str, f64)> = ablation_results
        .iter()
        .map(|(label, avg)| (*label, rainbow_avg - avg))
        .collect();
    importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (label, drop)) in importance.iter().enumerate() {
        println!("  {}. {} (drop: {:.6})", i + 1, label, drop);
    }

    println!("\n=== Done ===");
}

fn rainbow_config_base() -> RainbowConfig {
    RainbowConfig {
        state_dim: NUM_FEATURES,
        n_actions: 3,
        hidden_dim: 32,
        n_atoms: 21,
        v_min: -5.0,
        v_max: 5.0,
        gamma: 0.99,
        n_step: 3,
        learning_rate: 0.001,
        buffer_capacity: 5000,
        batch_size: 16,
        target_update_freq: 50,
        per_alpha: 0.6,
        per_beta_start: 0.4,
        use_double: true,
        use_dueling: true,
        use_per: true,
        use_nstep: true,
        use_distributional: true,
        use_noisy: true,
    }
}
