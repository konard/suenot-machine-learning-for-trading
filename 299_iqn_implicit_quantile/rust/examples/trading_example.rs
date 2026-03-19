use iqn_implicit_quantile::*;
use ndarray::Array1;

fn main() {
    println!("=== IQN (Implicit Quantile Networks) for Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Try to fetch BTCUSDT data from Bybit, fall back to synthetic
    // -----------------------------------------------------------------------
    println!("Step 1: Loading market data...");
    let prices = match load_market_data() {
        Ok(p) => {
            println!("  Loaded {} candles from Bybit (BTCUSDT 15m)\n", p.len());
            p
        }
        Err(e) => {
            println!("  Bybit fetch failed ({}), using synthetic data\n", e);
            generate_synthetic_prices(200, 0.05, 0.3, 50000.0)
        }
    };

    // -----------------------------------------------------------------------
    // Step 2: Train IQN agent
    // -----------------------------------------------------------------------
    println!("Step 2: Training IQN agent...");
    let state_dim = 8;
    let action_dim = 3; // sell, hold, buy
    let hidden_dim = 32;
    let n_cosines = 64;
    let buffer_capacity = 5000;
    let n_episodes = 20;
    let batch_size = 16;

    let mut agent = IQNAgent::new(
        state_dim,
        action_dim,
        hidden_dim,
        n_cosines,
        buffer_capacity,
        RiskMeasure::Neutral,
    );
    agent.epsilon = 0.3; // higher exploration during training

    let mut episode_rewards = Vec::new();

    for episode in 0..n_episodes {
        let mut env = TradingEnvironment::new(prices.clone(), 0.001, 10);
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;

        loop {
            let action = agent.select_action(&state);
            let (next_state, reward, done) = env.step(action);

            agent.store_experience(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });

            if agent.replay_buffer.len() >= batch_size {
                let _loss = agent.train_step(batch_size);
            }

            total_reward += reward;
            state = next_state;
            steps += 1;

            if done {
                break;
            }
        }

        episode_rewards.push(total_reward);

        if (episode + 1) % 5 == 0 {
            let avg: f64 =
                episode_rewards.iter().rev().take(5).sum::<f64>() / 5.0f64.min(episode_rewards.len() as f64);
            println!(
                "  Episode {}/{}: reward = {:.4}, avg(5) = {:.4}, steps = {}",
                episode + 1,
                n_episodes,
                total_reward,
                avg,
                steps
            );
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Show return distribution for different actions
    // -----------------------------------------------------------------------
    println!("\nStep 3: Return distribution analysis...");
    let _env = TradingEnvironment::new(prices.clone(), 0.001, 10);
    let sample_state = Array1::from_vec(vec![0.01, -0.005, 0.003, -0.001, 0.002, 0.015, 0.0, 0.0]);

    let action_names = ["Sell/Short", "Hold/Flat", "Buy/Long"];
    let n_quantiles = 10;

    for (action, name) in action_names.iter().enumerate() {
        let dist = agent.get_return_distribution(&sample_state, action, n_quantiles);
        println!("\n  Action: {}", name);
        println!("  {:>8} | {:>10}", "Quantile", "Value");
        println!("  {}", "-".repeat(23));
        for (tau, value) in &dist {
            println!("  {:>8.1}% | {:>10.6}", tau * 100.0, value);
        }

        // Compute summary statistics
        let values: Vec<f64> = dist.iter().map(|(_, v)| *v).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var_5 = dist
            .iter()
            .find(|(t, _)| *t >= 0.05)
            .map(|(_, v)| *v)
            .unwrap_or(values[0]);
        let cvar_10: f64 = dist
            .iter()
            .filter(|(t, _)| *t <= 0.1)
            .map(|(_, v)| *v)
            .sum::<f64>()
            / dist.iter().filter(|(t, _)| *t <= 0.1).count().max(1) as f64;

        println!("  Mean: {:.6}, VaR(5%): {:.6}, CVaR(10%): {:.6}", mean, var_5, cvar_10);
    }

    // -----------------------------------------------------------------------
    // Step 4: Compare risk-neutral vs CVaR policy
    // -----------------------------------------------------------------------
    println!("\n\nStep 4: Comparing risk-neutral vs CVaR policies...\n");

    let policies: Vec<(&str, RiskMeasure)> = vec![
        ("Risk-Neutral", RiskMeasure::Neutral),
        ("CVaR α=0.25 (conservative)", RiskMeasure::CVaR { alpha: 0.25 }),
        ("CVaR α=0.10 (very conservative)", RiskMeasure::CVaR { alpha: 0.10 }),
        ("Wang η=-0.5 (risk-averse)", RiskMeasure::Wang { eta: -0.5 }),
    ];

    for (name, risk_measure) in &policies {
        let taus = risk_measure.sample_taus(32);
        let action = select_action_risk_sensitive(&agent.network, &sample_state, risk_measure, 32);

        // Compute risk-adjusted values for each action
        let q_values = agent.network.forward_batch(&sample_state, &taus);
        let mean_q: Vec<f64> = (0..action_dim)
            .map(|a| {
                (0..taus.len())
                    .map(|i| q_values[[i, a]])
                    .sum::<f64>()
                    / taus.len() as f64
            })
            .collect();

        println!("  Policy: {}", name);
        println!(
            "    Selected action: {} ({})",
            action, action_names[action]
        );
        for (a, (q, aname)) in mean_q.iter().zip(action_names.iter()).enumerate() {
            let marker = if a == action { " <--" } else { "" };
            println!("    Q({}): {:.6}{}", aname, q, marker);
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Step 5: Backtest comparison
    // -----------------------------------------------------------------------
    println!("Step 5: Backtesting policies on test data...\n");

    let test_prices = if prices.len() > 100 {
        prices[prices.len() - 100..].to_vec()
    } else {
        generate_synthetic_prices(100, 0.05, 0.3, prices.last().copied().unwrap_or(50000.0))
    };

    let backtest_policies: Vec<(&str, RiskMeasure)> = vec![
        ("Risk-Neutral", RiskMeasure::Neutral),
        ("CVaR α=0.25", RiskMeasure::CVaR { alpha: 0.25 }),
    ];

    for (name, risk_measure) in &backtest_policies {
        let mut env = TradingEnvironment::new(test_prices.clone(), 0.001, 10);
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut max_drawdown = 0.0;
        let mut peak_reward = 0.0;
        let mut n_trades = 0;
        let mut prev_action = 1usize; // start flat

        loop {
            let action = select_action_risk_sensitive(&agent.network, &state, risk_measure, 16);
            let (next_state, reward, done) = env.step(action);

            if action != prev_action {
                n_trades += 1;
            }
            prev_action = action;

            total_reward += reward;
            if total_reward > peak_reward {
                peak_reward = total_reward;
            }
            let dd = peak_reward - total_reward;
            if dd > max_drawdown {
                max_drawdown = dd;
            }

            state = next_state;
            if done {
                break;
            }
        }

        println!("  {}", name);
        println!("    Total return:  {:.4}%", total_reward * 100.0);
        println!("    Max drawdown:  {:.4}%", max_drawdown * 100.0);
        println!("    Num trades:    {}", n_trades);
        let sharpe = if max_drawdown > 0.0 {
            total_reward / max_drawdown
        } else {
            0.0
        };
        println!("    Return/DD:     {:.4}", sharpe);
        println!();
    }

    println!("=== Done ===");
}

/// Attempt to load real market data from Bybit
fn load_market_data() -> anyhow::Result<Vec<f64>> {
    let client = BybitClient::new();
    let prices = client.get_close_prices("BTCUSDT", "15", 200)?;
    if prices.is_empty() {
        anyhow::bail!("No data returned");
    }
    Ok(prices)
}
