use goal_conditioned_rl::*;

fn main() {
    println!("=== Goal-Conditioned RL Trading ===\n");

    // -----------------------------------------------------------------------
    // 1. Attempt to fetch real data from Bybit, fall back to synthetic
    // -----------------------------------------------------------------------
    let candles = match fetch_bybit_data() {
        Some(c) => c,
        None => {
            println!("Using synthetic market data for demonstration.\n");
            generate_synthetic_candles(200, 50000.0)
        }
    };

    println!(
        "Loaded {} candles. Price range: {:.2} - {:.2}\n",
        candles.len(),
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max),
    );

    // -----------------------------------------------------------------------
    // 2. Set up training parameters
    // -----------------------------------------------------------------------
    let window_size = 5;
    let state_dim = window_size * 5; // 5 features per candle
    let goal_dim = 3;
    let hidden_dim = 64;
    let lr = 0.001;
    let buffer_capacity = 50000;
    let batch_size = 32;
    let num_episodes = 50;

    // -----------------------------------------------------------------------
    // 3. Train with target return goal
    // -----------------------------------------------------------------------
    println!("--- Training with Target Return Goal (5%) ---");
    let goal = TradingGoal::TargetReturn(0.05);
    let mut agent = GCRLAgent::new(
        state_dim,
        goal_dim,
        hidden_dim,
        lr,
        buffer_capacity,
        batch_size,
        HERStrategy::Future,
    );

    let mut env = TradingEnvironment::new(candles.clone(), goal, window_size);

    for episode in 0..num_episodes {
        let reward = agent.train_episode(&mut env);
        if (episode + 1) % 10 == 0 {
            let achieved = env.get_achieved_goal();
            println!(
                "  Episode {:3}: reward={:8.3}, return={:+.4}, drawdown={:+.4}, sharpe={:+.4}, epsilon={:.3}",
                episode + 1,
                reward,
                achieved[0],
                achieved[1],
                achieved[2],
                agent.policy.epsilon,
            );
        }
    }

    // -----------------------------------------------------------------------
    // 4. Evaluate with different goal targets
    // -----------------------------------------------------------------------
    println!("\n--- Evaluating Agent with Different Goals ---\n");

    let test_goals = vec![
        ("Conservative (2% return)", TradingGoal::TargetReturn(0.02)),
        ("Moderate (5% return)", TradingGoal::TargetReturn(0.05)),
        ("Aggressive (10% return)", TradingGoal::TargetReturn(0.10)),
        ("Drawdown limit (-5%)", TradingGoal::MaxDrawdown(-0.05)),
        ("Sharpe target (1.5)", TradingGoal::TargetSharpe(1.5)),
    ];

    for (name, goal) in &test_goals {
        let mut eval_env = TradingEnvironment::new(candles.clone(), goal.clone(), window_size);
        let (total_reward, achieved) = agent.evaluate(&mut eval_env, goal.clone());
        let final_value = eval_env.portfolio_value();
        let pnl = (final_value - 10000.0) / 10000.0 * 100.0;

        println!("  Goal: {}", name);
        println!("    Total reward:    {:+.3}", total_reward);
        println!("    Achieved return: {:+.4} ({:+.2}%)", achieved[0], pnl);
        println!("    Max drawdown:    {:+.4}", achieved[1]);
        println!("    Sharpe ratio:    {:+.4}", achieved[2]);
        println!("    Final value:     ${:.2}", final_value);
        println!();
    }

    // -----------------------------------------------------------------------
    // 5. Demonstrate HER effect
    // -----------------------------------------------------------------------
    println!("--- HER Replay Buffer Statistics ---");
    println!(
        "  Total transitions stored: {} (includes HER-relabeled)",
        agent.replay_buffer.len()
    );
    println!(
        "  Training episodes completed: {}",
        agent.train_steps
    );
    println!(
        "  Final epsilon: {:.4}",
        agent.policy.epsilon
    );

    println!("\n=== Goal-Conditioned RL Trading Complete ===");
}

/// Try to fetch BTCUSDT klines from Bybit; return None on failure
fn fetch_bybit_data() -> Option<Vec<MarketCandle>> {
    println!("Fetching BTCUSDT data from Bybit...");
    let client = BybitClient::new();
    match client.fetch_klines("BTCUSDT", "60", 200) {
        Ok(candles) if !candles.is_empty() => {
            println!(
                "Fetched {} candles from Bybit (latest close: {:.2})\n",
                candles.len(),
                candles.last().unwrap().close
            );
            Some(candles)
        }
        Ok(_) => {
            println!("Bybit returned no data, falling back to synthetic data.");
            None
        }
        Err(e) => {
            println!("Could not fetch from Bybit ({}), using synthetic data.", e);
            None
        }
    }
}
