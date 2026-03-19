//! # Planning RL Trading Example
//!
//! Demonstrates model-based RL with planning for BTCUSDT trading.
//! - Fetches data from Bybit (or uses synthetic data as fallback)
//! - Learns a world model from historical transitions
//! - Plans optimal actions using MPC (random shooting + CEM)
//! - Compares model-based planning vs model-free baseline

use planning_rl_trading::*;

fn main() -> anyhow::Result<()> {
    println!("=== Chapter 312: Planning RL Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch data from Bybit (with synthetic fallback)
    // -----------------------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT data from Bybit...");
    let klines = match BybitClient::new().fetch_klines("BTCUSDT", "60", 200) {
        Ok(k) if k.len() >= 50 => {
            println!("  Fetched {} klines from Bybit", k.len());
            k
        }
        _ => {
            println!("  Bybit unavailable, using synthetic data");
            generate_synthetic_klines(200, 50000.0)
        }
    };

    println!(
        "  Price range: {:.2} - {:.2}",
        klines.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
        klines.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max),
    );

    // -----------------------------------------------------------------------
    // Step 2: Build transitions from historical data
    // -----------------------------------------------------------------------
    println!("\nStep 2: Building state representations and transitions...");
    let lookback = 10;
    let transitions = build_transitions(&klines, lookback);
    println!("  Generated {} transitions", transitions.len());

    let split = transitions.len() * 3 / 4;
    let (train_data, test_data) = transitions.split_at(split);
    println!(
        "  Train: {} transitions, Test: {} transitions",
        train_data.len(),
        test_data.len()
    );

    // -----------------------------------------------------------------------
    // Step 3: Train world model
    // -----------------------------------------------------------------------
    println!("\nStep 3: Training world model...");
    let state_dim = 5;
    let mut world_model = WorldModel::new(state_dim, 0.001);

    let (state_err_before, reward_err_before) = world_model.evaluate(test_data);
    println!(
        "  Before training -- State MSE: {:.6}, Reward MSE: {:.6}",
        state_err_before, reward_err_before
    );

    world_model.train(train_data, 50);

    let (state_err_after, reward_err_after) = world_model.evaluate(test_data);
    println!(
        "  After training  -- State MSE: {:.6}, Reward MSE: {:.6}",
        state_err_after, reward_err_after
    );
    println!(
        "  Improvement: state {:.1}%, reward {:.1}%",
        if state_err_before > 0.0 {
            (1.0 - state_err_after / state_err_before) * 100.0
        } else {
            0.0
        },
        if reward_err_before > 0.0 {
            (1.0 - reward_err_after / reward_err_before) * 100.0
        } else {
            0.0
        },
    );

    // -----------------------------------------------------------------------
    // Step 4: Train model ensemble for uncertainty estimation
    // -----------------------------------------------------------------------
    println!("\nStep 4: Training model ensemble (5 models)...");
    let mut ensemble = ModelEnsemble::new(5, state_dim, 0.001);
    ensemble.train(train_data, 30);

    // Show uncertainty for a sample state
    if let Some(sample_state) = build_state(&klines, lookback + 1, lookback) {
        for &action in Action::all() {
            let unc = ensemble.uncertainty(&sample_state, action);
            println!("  Uncertainty for {:?}: {:.6}", action, unc);
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: MPC Planning
    // -----------------------------------------------------------------------
    println!("\nStep 5: MPC Planning (Random Shooting)...");
    let ensemble_rs = ModelEnsemble::new(3, state_dim, 0.001);
    let mut planner_rs = MPCPlanner::new(
        ensemble_rs,
        10,   // horizon
        100,  // candidates
        0.99, // gamma
        PlanningMethod::RandomShooting,
        1e6,  // uncertainty threshold
    );
    planner_rs.train(train_data, 30);

    println!("\n  MPC Planning (CEM)...");
    let ensemble_cem = ModelEnsemble::new(3, state_dim, 0.001);
    let mut planner_cem = MPCPlanner::new(
        ensemble_cem,
        10,
        100,
        0.99,
        PlanningMethod::CEM {
            elite_frac: 0.2,
            iterations: 5,
        },
        1e6,
    );
    planner_cem.train(train_data, 30);

    // Evaluate both planners on test period
    let test_states: Vec<Vec<f64>> = (lookback + split / 3..klines.len() - 1)
        .filter_map(|i| build_state(&klines, i, lookback))
        .take(20)
        .collect();

    println!("\n  Planning results on {} test states:", test_states.len());
    println!("  {:>5} | {:>12} {:>12} | {:>12} {:>12}", "Step", "RS Action", "RS Reward", "CEM Action", "CEM Reward");
    println!("  {}", "-".repeat(65));

    let mut rs_total = 0.0;
    let mut cem_total = 0.0;
    for (i, state) in test_states.iter().enumerate() {
        let (rs_action, rs_reward) = planner_rs.plan(state);
        let (cem_action, cem_reward) = planner_cem.plan(state);
        rs_total += rs_reward;
        cem_total += cem_reward;
        if i < 10 {
            println!(
                "  {:>5} | {:>12?} {:>12.6} | {:>12?} {:>12.6}",
                i, rs_action, rs_reward, cem_action, cem_reward
            );
        }
    }
    println!("  ...");
    println!(
        "  Total planned reward -- RS: {:.6}, CEM: {:.6}",
        rs_total, cem_total
    );

    // -----------------------------------------------------------------------
    // Step 6: Dyna-Q Agent comparison
    // -----------------------------------------------------------------------
    println!("\nStep 6: Dyna-Q agent vs pure Q-learning baseline...");

    // Dyna-Q (model-based)
    let mut dyna_agent = DynaQAgent::new(state_dim, 0.1, 0.99, 0.1, 10);
    let mut dyna_reward = 0.0;

    // Pure Q-learning (model-free: planning_steps = 0)
    let mut qlearn_agent = DynaQAgent::new(state_dim, 0.1, 0.99, 0.1, 0);
    let mut qlearn_reward = 0.0;

    // Feed transitions through both
    for t in train_data.iter().take(200) {
        dyna_agent.step(t.clone());
        dyna_reward += t.reward;

        qlearn_agent.step(t.clone());
        qlearn_reward += t.reward;
    }

    println!(
        "  Dyna-Q    -- Q-table size: {}, Buffer: {}, Total reward: {:.4}",
        dyna_agent.q_table_size(),
        dyna_agent.buffer_size(),
        dyna_reward,
    );
    println!(
        "  Q-learning -- Q-table size: {}, Buffer: {}, Total reward: {:.4}",
        qlearn_agent.q_table_size(),
        qlearn_agent.buffer_size(),
        qlearn_reward,
    );

    // Evaluate on test data: count how many times each agent picks the best action
    let mut dyna_correct = 0;
    let mut qlearn_correct = 0;
    let mut total_test = 0;

    for t in test_data.iter().take(100) {
        let dyna_action = dyna_agent.select_action(&t.state);
        let qlearn_action = qlearn_agent.select_action(&t.state);

        // The "correct" action is the one with highest actual reward
        // (we know from the transition data)
        let best_action = t.action; // ground truth from data
        if dyna_action == best_action {
            dyna_correct += 1;
        }
        if qlearn_action == best_action {
            qlearn_correct += 1;
        }
        total_test += 1;
    }

    println!(
        "\n  Action agreement with data -- Dyna-Q: {}/{} ({:.1}%), Q-learning: {}/{} ({:.1}%)",
        dyna_correct,
        total_test,
        100.0 * dyna_correct as f64 / total_test as f64,
        qlearn_correct,
        total_test,
        100.0 * qlearn_correct as f64 / total_test as f64,
    );

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== Summary ===");
    println!("  World model successfully trained on {} transitions", train_data.len());
    println!("  Ensemble uncertainty estimation operational with 5 models");
    println!("  MPC planning compared Random Shooting vs CEM on {} test states", test_states.len());
    println!("  Dyna-Q (model-based) vs Q-learning (model-free) compared");
    println!("\n  Key insight: Model-based planning allows the agent to reason about");
    println!("  multi-step consequences before committing capital, improving sample");
    println!("  efficiency and enabling sophisticated execution strategies.");

    Ok(())
}
