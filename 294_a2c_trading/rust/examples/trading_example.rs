use a2c_trading::*;

/// Generate synthetic price data mimicking a trending market with noise.
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64; // starting BTC-like price
    let mut candles = Vec::with_capacity(n);
    for i in 0..n {
        let trend = ((i as f64) * 0.02).sin() * 100.0;
        let noise = rng.gen_range(-200.0..200.0);
        price = (price + trend + noise).max(1000.0);
        let high = price + rng.gen_range(50.0..300.0);
        let low = price - rng.gen_range(50.0..300.0);
        let open = price + rng.gen_range(-100.0..100.0);
        candles.push(Candle {
            timestamp: 1_700_000_000 + (i as u64) * 60,
            open,
            high,
            low,
            close: price,
            volume: rng.gen_range(10.0..1000.0),
        });
    }
    candles
}

fn main() {
    println!("=== A2C Trading Agent Example ===\n");

    // ── Step 1: Try to fetch data from Bybit, fall back to synthetic ──
    println!("[1] Fetching market data...");
    let candles = match fetch_bybit_klines("BTCUSDT", "5", 200) {
        Ok(c) if c.len() > 50 => {
            println!("    Fetched {} candles from Bybit (BTCUSDT, 5m)", c.len());
            c
        }
        _ => {
            println!("    Bybit API unavailable, using synthetic data (200 candles)");
            generate_synthetic_candles(200)
        }
    };

    // ── Step 2: Train A2C agent ──
    println!("\n[2] Training A2C agent...");
    let mut env = TradingEnv::new(&candles, 0.001);
    let config = A2CConfig {
        gamma: 0.99,
        lambda: 0.95,
        learning_rate: 1e-3,
        entropy_coef: 0.02,
        value_coef: 0.5,
        max_grad_norm: 0.5,
        hidden_dim: 64,
        rollout_steps: 32,
    };
    let mut agent = A2CAgent::new(env.state_dim(), config);

    let num_episodes = 20;
    let mut episode_rewards = Vec::new();
    let mut episode_entropies = Vec::new();

    for ep in 0..num_episodes {
        env.reset();
        let mut ep_entropy = Vec::new();
        while !env.is_done() {
            let rollout = agent.collect_rollout(&mut env);
            for &e in &rollout.entropies {
                ep_entropy.push(e);
            }
            agent.update(&rollout);
        }
        let mean_ent = if ep_entropy.is_empty() {
            0.0
        } else {
            ep_entropy.iter().sum::<f64>() / ep_entropy.len() as f64
        };
        episode_rewards.push(env.total_reward);
        episode_entropies.push(mean_ent);

        if (ep + 1) % 5 == 0 || ep == 0 {
            println!(
                "    Episode {:3}/{}: reward = {:+.4}, mean entropy = {:.4}",
                ep + 1,
                num_episodes,
                env.total_reward,
                mean_ent
            );
        }
    }

    // ── Step 3: Show policy entropy trend ──
    println!("\n[3] Policy entropy over training:");
    println!("    (Higher = more exploration, Lower = more confident)");
    let chunk_size = num_episodes / 4;
    for (i, chunk) in episode_entropies.chunks(chunk_size.max(1)).enumerate() {
        let avg: f64 = chunk.iter().sum::<f64>() / chunk.len() as f64;
        let bar_len = (avg * 20.0).round() as usize;
        let bar: String = "#".repeat(bar_len.min(40));
        println!(
            "    Episodes {:2}-{:2}: entropy={:.4} |{}|",
            i * chunk_size + 1,
            (i + 1) * chunk_size,
            avg,
            bar
        );
    }

    // ── Step 4: Compare with REINFORCE (no baseline) ──
    println!("\n[4] Comparing A2C vs REINFORCE (no baseline)...");
    let mut reinforce_env = TradingEnv::new(&candles, 0.001);
    let mut reinforce_agent = ReinforceAgent::new(reinforce_env.state_dim(), 64);
    let mut reinforce_rewards = Vec::new();
    for _ in 0..num_episodes {
        let r = reinforce_agent.run_episode(&mut reinforce_env);
        reinforce_rewards.push(r);
    }

    let a2c_mean: f64 = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
    let reinforce_mean: f64 =
        reinforce_rewards.iter().sum::<f64>() / reinforce_rewards.len() as f64;
    let a2c_std = std_dev(&episode_rewards, a2c_mean);
    let reinforce_std = std_dev(&reinforce_rewards, reinforce_mean);

    println!("    A2C       : mean reward = {:+.4}, std = {:.4}", a2c_mean, a2c_std);
    println!(
        "    REINFORCE : mean reward = {:+.4}, std = {:.4}",
        reinforce_mean, reinforce_std
    );
    println!();
    if a2c_std < reinforce_std {
        println!("    -> A2C shows lower variance (std), demonstrating the benefit of the critic baseline.");
    } else {
        println!("    -> Results vary by run. In expectation, A2C typically has lower variance than REINFORCE.");
    }

    // ── Step 5: Final evaluation ──
    println!("\n[5] Final evaluation (single episode with trained A2C):");
    env.reset();
    let mut actions_count = [0usize; NUM_ACTIONS];
    while !env.is_done() {
        let state = env.get_state();
        let (action, probs, value) = agent.act(&state);
        let _ = env.step_action(action);
        actions_count[action] += 1;
        if env.step <= 5 || env.is_done() {
            println!(
                "    Step {:3}: action={} probs=[{:.2},{:.2},{:.2}] V={:+.4}",
                env.step,
                ["HOLD", "BUY ", "SELL"][action],
                probs[0],
                probs[1],
                probs[2],
                value
            );
        }
    }
    println!("    ---");
    println!(
        "    Final reward: {:+.4} | Actions: Hold={}, Buy={}, Sell={}",
        env.total_reward, actions_count[0], actions_count[1], actions_count[2]
    );
    println!("\n=== Done ===");
}

fn std_dev(data: &[f64], mean: f64) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}
