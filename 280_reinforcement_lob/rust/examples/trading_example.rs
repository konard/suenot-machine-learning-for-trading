//! Trading Example: RL Market Making on LOB
//!
//! This example:
//! 1. Fetches BTCUSDT orderbook from Bybit
//! 2. Trains an RL market making agent
//! 3. Compares RL agent vs fixed-spread baseline
//! 4. Shows PnL and inventory over time

use reinforcement_lob::*;

fn main() {
    println!("=== Reinforcement Learning for LOB Trading ===\n");

    // --------------------------------------------------------
    // Step 1: Fetch Bybit orderbook (best-effort, continue on failure)
    // --------------------------------------------------------
    println!("--- Step 1: Fetching BTCUSDT orderbook from Bybit ---");
    match fetch_bybit_orderbook_blocking("BTCUSDT", 25) {
        Ok(ob) => {
            if let (Some(bid), Some(ask)) = (ob.best_bid(), ob.best_ask()) {
                println!("  Best bid: {:.2} (qty: {:.6})", bid.price, bid.quantity);
                println!("  Best ask: {:.2} (qty: {:.6})", ask.price, ask.quantity);
                println!(
                    "  Mid price: {:.2}",
                    ob.mid_price().unwrap_or(0.0)
                );
                println!(
                    "  Spread: {:.2}",
                    ob.spread().unwrap_or(0.0)
                );
            }
            println!("  Top 5 bids:");
            for level in ob.top_bids(5) {
                println!("    {:.2} x {:.6}", level.price, level.quantity);
            }
            println!("  Top 5 asks:");
            for level in ob.top_asks(5) {
                println!("    {:.2} x {:.6}", level.price, level.quantity);
            }
        }
        Err(e) => {
            println!("  Could not fetch orderbook (network may be unavailable): {}", e);
            println!("  Continuing with simulated data...");
        }
    }
    println!();

    // --------------------------------------------------------
    // Step 2: Train RL market making agent
    // --------------------------------------------------------
    println!("--- Step 2: Training RL Market Making Agent ---");

    let initial_mid = 50000.0; // BTC-like price
    let tick_size = 0.5;
    let volatility = 2.0;
    let time_horizon = 200;
    let quote_size = 0.01; // 0.01 BTC
    let risk_aversion = 0.0001;
    let num_episodes = 500;

    let mut env = TradingEnvironment::new(
        initial_mid,
        tick_size,
        volatility,
        time_horizon,
        quote_size,
        risk_aversion,
    );
    let mut agent = DQNMarketMaker::new(env.num_actions());

    let mut rewards_history: Vec<f64> = Vec::new();
    let mut pnl_history: Vec<f64> = Vec::new();

    for episode in 0..num_episodes {
        let result = run_episode(&mut env, &mut agent, true);
        rewards_history.push(result.total_reward);
        pnl_history.push(result.total_pnl);

        if (episode + 1) % 100 == 0 {
            let recent_reward: f64 =
                rewards_history[rewards_history.len().saturating_sub(100)..].iter().sum::<f64>()
                    / 100.0_f64.min(rewards_history.len() as f64);
            let recent_pnl: f64 =
                pnl_history[pnl_history.len().saturating_sub(100)..].iter().sum::<f64>()
                    / 100.0_f64.min(pnl_history.len() as f64);
            println!(
                "  Episode {}: avg_reward={:.4}, avg_pnl={:.4}, epsilon={:.4}, q_states={}",
                episode + 1,
                recent_reward,
                recent_pnl,
                agent.epsilon,
                agent.q_table.len(),
            );
        }
    }
    println!();

    // --------------------------------------------------------
    // Step 3: Evaluate RL agent vs fixed-spread baseline
    // --------------------------------------------------------
    println!("--- Step 3: Evaluation (RL Agent vs Fixed-Spread Baseline) ---");

    let eval_episodes = 100;

    // Evaluate RL agent (greedy, no exploration).
    let mut rl_rewards = Vec::new();
    let mut rl_pnls = Vec::new();
    let mut rl_inventories = Vec::new();

    for _ in 0..eval_episodes {
        let result = run_episode(&mut env, &mut agent, false);
        rl_rewards.push(result.total_reward);
        rl_pnls.push(result.total_pnl);
        rl_inventories.push(result.final_inventory.abs());
    }

    // Evaluate fixed-spread baseline.
    let baseline = FixedSpreadAgent::new(env.spread_options.len());
    let mut bl_rewards = Vec::new();
    let mut bl_pnls = Vec::new();
    let mut bl_inventories = Vec::new();

    for _ in 0..eval_episodes {
        let result = run_baseline_episode(&mut env, &baseline);
        bl_rewards.push(result.total_reward);
        bl_pnls.push(result.total_pnl);
        bl_inventories.push(result.final_inventory.abs());
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

    println!("  RL Agent:");
    println!("    Avg reward:    {:.4}", mean(&rl_rewards));
    println!("    Avg PnL:       {:.4}", mean(&rl_pnls));
    println!("    Avg |inventory|: {:.4}", mean(&rl_inventories));
    println!();
    println!("  Fixed-Spread Baseline:");
    println!("    Avg reward:    {:.4}", mean(&bl_rewards));
    println!("    Avg PnL:       {:.4}", mean(&bl_pnls));
    println!("    Avg |inventory|: {:.4}", mean(&bl_inventories));
    println!();

    // --------------------------------------------------------
    // Step 4: Show per-step PnL and inventory for a single episode
    // --------------------------------------------------------
    println!("--- Step 4: Single Episode Trace (RL Agent) ---");

    let state = env.reset();
    let mut state_key = state.to_key();
    let mut step_pnls = Vec::new();
    let mut step_inventories = Vec::new();

    for step in 0..time_horizon {
        let action = agent.best_action(&state_key);
        let (next_state, reward, done) = env.step(action);
        step_pnls.push(env.total_pnl());
        step_inventories.push(env.agent_inventory);

        if step % 40 == 0 || done {
            println!(
                "  Step {:>3}: PnL={:>10.4}, Inventory={:>8.4}, Reward={:>8.4}",
                step, env.total_pnl(), env.agent_inventory, reward
            );
        }

        if done {
            break;
        }
        state_key = next_state.to_key();
    }

    println!();
    println!("  Final PnL: {:.4}", env.total_pnl());
    println!("  Final Inventory: {:.4}", env.agent_inventory);
    println!();
    println!("=== Done ===");
}
