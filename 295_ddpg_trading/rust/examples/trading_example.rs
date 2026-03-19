use ddpg_trading::*;

/// Generate synthetic price data resembling BTC for offline testing.
fn generate_synthetic_prices(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    for i in 0..n {
        // Trending + mean-reverting + noise
        let trend = 0.0001 * (i as f64 / n as f64 - 0.5);
        let cycle = (i as f64 * 0.05).sin() * 0.002;
        let noise = (rand::Rng::gen_range(&mut rng, -1.0_f64..1.0)) * 0.003;
        price *= 1.0 + trend + cycle + noise;
        prices.push(price);
        volumes.push(rand::Rng::gen_range(&mut rng, 500.0_f64..2000.0));
    }
    (prices, volumes)
}

/// Simple discrete DQN-style baseline: buy(+1), sell(-1), hold(0) based on recent return.
fn discrete_baseline(prices: &[f64]) -> f64 {
    let mut pnl: f64 = 0.0;
    let mut position: f64 = 0.0;
    for i in 11..prices.len() - 1 {
        let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
        // Simple momentum: if positive return, go long; negative, go short
        let new_pos: f64 = if ret > 0.001 {
            1.0
        } else if ret < -0.001 {
            -1.0
        } else {
            0.0
        };
        let cost = (new_pos - position).abs() * 0.0004;
        let forward_ret = (prices[i + 1] - prices[i]) / prices[i];
        pnl += position * forward_ret - cost;
        position = new_pos;
    }
    pnl
}

fn main() {
    println!("=== DDPG for Continuous Position Sizing ===\n");

    // --- Attempt to fetch live data from Bybit ---
    println!("Attempting to fetch BTCUSDT data from Bybit...");
    let (prices, volumes) = match BybitClient::new().fetch_klines("BTCUSDT", "60", 200) {
        Ok(candles) if candles.len() > 20 => {
            println!("Fetched {} candles from Bybit API.\n", candles.len());
            let p: Vec<f64> = candles.iter().map(|c| c.close).collect();
            let v: Vec<f64> = candles.iter().map(|c| c.volume).collect();
            (p, v)
        }
        Ok(_) => {
            println!("Not enough candles from API, using synthetic data.\n");
            generate_synthetic_prices(500)
        }
        Err(e) => {
            println!("Could not fetch from Bybit ({}), using synthetic data.\n", e);
            generate_synthetic_prices(500)
        }
    };

    println!("Data points: {}", prices.len());
    println!(
        "Price range: {:.2} - {:.2}\n",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // --- Train DDPG Agent ---
    let state_dim = 8;
    let action_dim = 1;
    let mut agent = DDPGAgent::new(state_dim, action_dim);
    let num_episodes: usize = 50;

    println!("Training DDPG for {} episodes...\n", num_episodes);

    let mut episode_rewards = Vec::new();

    for episode in 0..num_episodes {
        let mut env = TradingEnv::new(prices.clone(), volumes.clone());
        let mut state = env.reset();
        agent.noise.reset();

        let mut total_reward = 0.0;
        let mut steps = 0;

        loop {
            let action = agent.select_action(&state, true);
            let action_val = action[0];
            let (next_state, reward, done) = env.step(action_val);

            agent.store_transition(Transition {
                state: state.clone(),
                action: action.clone(),
                reward,
                next_state: next_state.clone(),
                done,
            });

            agent.train_step();

            total_reward += reward;
            steps += 1;
            state = next_state;

            if done {
                break;
            }
        }

        episode_rewards.push(total_reward);

        if (episode + 1) % 10 == 0 {
            let avg: f64 = episode_rewards[episode.saturating_sub(9)..=episode]
                .iter()
                .sum::<f64>()
                / episode_rewards[episode.saturating_sub(9)..=episode].len() as f64;
            println!(
                "Episode {}/{}: reward = {:.6}, avg(10) = {:.6}, PnL = {:.6}, steps = {}",
                episode + 1,
                num_episodes,
                total_reward,
                avg,
                env.pnl,
                steps
            );
        }
    }

    // --- Evaluate trained agent (no exploration noise) ---
    println!("\n=== Evaluation (no noise) ===\n");

    let mut env = TradingEnv::new(prices.clone(), volumes.clone());
    let mut state = env.reset();
    let mut positions = Vec::new();
    let mut pnl_curve = Vec::new();
    let mut step_rewards = Vec::new();

    loop {
        let action = agent.select_action(&state, false);
        let action_val = action[0];
        positions.push(action_val);

        let (next_state, reward, done) = env.step(action_val);
        step_rewards.push(reward);
        pnl_curve.push(env.pnl);
        state = next_state;

        if done {
            break;
        }
    }

    // --- Performance metrics ---
    let final_pnl = *pnl_curve.last().unwrap_or(&0.0);
    let sharpe = compute_sharpe(&step_rewards);
    let max_dd = compute_max_drawdown(&pnl_curve);

    println!("DDPG Results:");
    println!("  Final PnL:      {:.6}", final_pnl);
    println!("  Sharpe Ratio:   {:.4}", sharpe);
    println!("  Max Drawdown:   {:.6}", max_dd);
    println!("  Total Steps:    {}", positions.len());

    // --- Position size evolution ---
    println!("\nPosition Size Evolution (first 30 steps):");
    for (i, pos) in positions.iter().take(30).enumerate() {
        let bar_len = ((pos.abs()) * 20.0) as usize;
        let bar: String = if *pos >= 0.0 {
            format!("{}{}", " ".repeat(20), "+".repeat(bar_len))
        } else {
            let start = 20 - bar_len;
            format!("{}{}{}", " ".repeat(start), "-".repeat(bar_len), " ".repeat(0))
        };
        println!("  Step {:3}: [{:+.3}] |{}|", i + 1, pos, bar);
    }

    // --- Compare with discrete baseline ---
    println!("\n=== Comparison with Discrete Baseline ===\n");
    let baseline_pnl = discrete_baseline(&prices);
    println!("  Discrete Baseline PnL: {:.6}", baseline_pnl);
    println!("  DDPG Continuous PnL:   {:.6}", final_pnl);
    println!(
        "  Improvement:           {:.4}%",
        if baseline_pnl.abs() > 1e-10 {
            (final_pnl - baseline_pnl) / baseline_pnl.abs() * 100.0
        } else {
            0.0
        }
    );

    // --- Position statistics ---
    let avg_pos: f64 = positions.iter().sum::<f64>() / positions.len() as f64;
    let avg_abs_pos: f64 = positions.iter().map(|p| p.abs()).sum::<f64>() / positions.len() as f64;
    let long_frac = positions.iter().filter(|&&p| p > 0.05).count() as f64 / positions.len() as f64;
    let short_frac = positions.iter().filter(|&&p| p < -0.05).count() as f64 / positions.len() as f64;
    let flat_frac = 1.0 - long_frac - short_frac;

    println!("\nPosition Statistics:");
    println!("  Average Position:     {:+.4}", avg_pos);
    println!("  Average |Position|:   {:.4}", avg_abs_pos);
    println!("  Long  (>+0.05):      {:.1}%", long_frac * 100.0);
    println!("  Short (<-0.05):      {:.1}%", short_frac * 100.0);
    println!("  Flat  (|p|<=0.05):   {:.1}%", flat_frac * 100.0);

    println!("\n=== DDPG Trading Example Complete ===");
}
