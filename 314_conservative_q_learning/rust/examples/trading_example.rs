//! # CQL Trading Example
//!
//! Demonstrates Conservative Q-Learning for offline trading on BTCUSDT data from Bybit.
//! Compares CQL with standard Q-learning to show the benefits of conservative regularization.

use conservative_q_learning::*;

/// Generate synthetic candle data as fallback when API is unavailable.
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0f64;
    let mut rng_state: u64 = 42;

    for i in 0..n {
        // Simple LCG for deterministic pseudo-random numbers
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.02;

        // Add a slight trend and mean reversion
        let trend = (i as f64 * 0.01).sin() * 0.005;
        let ret = trend + noise;
        price *= 1.0 + ret;

        let high = price * (1.0 + noise.abs() * 0.5);
        let low = price * (1.0 - noise.abs() * 0.5);
        let open = price * (1.0 - ret * 0.3);
        let close = price;

        candles.push(Candle {
            timestamp: (1700000000 + i * 900) as u64, // 15-min intervals
            open,
            high,
            low,
            close,
            volume: 100.0 + (i as f64 * 0.5).sin().abs() * 200.0,
        });
    }

    candles
}

fn main() {
    println!("=== Conservative Q-Learning for Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch or generate market data
    // -----------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT data from Bybit...");
    let client = BybitClient::new();
    let candles = match client.fetch_klines("BTCUSDT", "15", 200) {
        Ok(c) if c.len() >= 50 => {
            println!("    Fetched {} candles from Bybit API", c.len());
            c
        }
        Ok(_) | Err(_) => {
            println!("    API unavailable, using synthetic data (200 candles)");
            generate_synthetic_candles(200)
        }
    };

    println!(
        "    Price range: {:.2} - {:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
    );

    // -----------------------------------------------------------------------
    // Step 2: Build offline dataset
    // -----------------------------------------------------------------------
    println!("\n[2] Building offline dataset...");
    let transaction_cost = 0.0004; // 4 bps
    let buffer = build_offline_dataset(&candles, transaction_cost);
    println!("    Dataset size: {} transitions", buffer.len());

    // Show dataset statistics
    let all = buffer.all_transitions();
    let mean_reward: f64 = all.iter().map(|t| t.reward).sum::<f64>() / all.len() as f64;
    let total_reward: f64 = all.iter().map(|t| t.reward).sum();
    println!("    Mean reward per step: {:.6}", mean_reward);
    println!("    Total behavior policy reward: {:.6}", total_reward);

    // Action distribution in dataset
    let mut action_counts = [0usize; 5];
    for t in &all {
        action_counts[t.action] += 1;
    }
    println!("    Action distribution: StrongSell={}, Sell={}, Hold={}, Buy={}, StrongBuy={}",
        action_counts[0], action_counts[1], action_counts[2],
        action_counts[3], action_counts[4]);

    // -----------------------------------------------------------------------
    // Step 3: Train CQL agent
    // -----------------------------------------------------------------------
    println!("\n[3] Training CQL agent (offline)...");
    let cql_config = CQLConfig {
        state_dim: 7,
        hidden1: 64,
        hidden2: 32,
        gamma: 0.99,
        alpha: 1.0,  // conservative regularization weight
        learning_rate: 5e-4,
        tau: 0.005,
        epsilon: 0.0, // no exploration for evaluation
        num_ood_samples: 10,
    };
    let mut cql_agent = CQLAgent::new(cql_config);

    let epochs = 20;
    let batch_size = 32;
    let cql_losses = cql_agent.train_offline(&buffer, epochs, batch_size);

    println!("    Training complete.");
    println!("    Initial loss: {:.6}", cql_losses.first().unwrap_or(&0.0));
    println!("    Final loss:   {:.6}", cql_losses.last().unwrap_or(&0.0));

    // -----------------------------------------------------------------------
    // Step 4: Train standard Q-learning agent (for comparison)
    // -----------------------------------------------------------------------
    println!("\n[4] Training standard Q-learning agent (offline, no CQL)...");
    let mut std_agent = StandardQLAgent::new(7, 64, 32);
    std_agent.learning_rate = 5e-4;
    let std_losses = std_agent.train_offline(&buffer, epochs, batch_size);

    println!("    Training complete.");
    println!("    Initial loss: {:.6}", std_losses.first().unwrap_or(&0.0));
    println!("    Final loss:   {:.6}", std_losses.last().unwrap_or(&0.0));

    // -----------------------------------------------------------------------
    // Step 5: Compare Q-value estimates
    // -----------------------------------------------------------------------
    println!("\n[5] Comparing Q-value estimates...");

    // Sample some states and compare Q-values
    let sample_states = buffer.sample(10);
    let mut cql_max_qs = Vec::new();
    let mut std_max_qs = Vec::new();
    let mut cql_mean_qs = Vec::new();
    let mut std_mean_qs = Vec::new();

    for t in &sample_states {
        let cql_q = cql_agent.q_network.forward(&t.state);
        let std_q = std_agent.q_network.forward(&t.state);

        let cql_max = cql_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let std_max = std_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let cql_mean = cql_q.mean().unwrap_or(0.0);
        let std_mean = std_q.mean().unwrap_or(0.0);

        cql_max_qs.push(cql_max);
        std_max_qs.push(std_max);
        cql_mean_qs.push(cql_mean);
        std_mean_qs.push(std_mean);
    }

    let cql_avg_max: f64 = cql_max_qs.iter().sum::<f64>() / cql_max_qs.len() as f64;
    let std_avg_max: f64 = std_max_qs.iter().sum::<f64>() / std_max_qs.len() as f64;
    let cql_avg_mean: f64 = cql_mean_qs.iter().sum::<f64>() / cql_mean_qs.len() as f64;
    let std_avg_mean: f64 = std_mean_qs.iter().sum::<f64>() / std_mean_qs.len() as f64;

    println!("    CQL  - Avg max Q: {:.6}, Avg mean Q: {:.6}", cql_avg_max, cql_avg_mean);
    println!("    Std  - Avg max Q: {:.6}, Avg mean Q: {:.6}", std_avg_max, std_avg_mean);

    if cql_avg_max < std_avg_max {
        println!("    --> CQL produces LOWER Q-values (more conservative) as expected!");
    } else {
        println!("    --> Note: CQL may need more training to show conservatism clearly.");
    }

    // -----------------------------------------------------------------------
    // Step 6: Evaluate policies
    // -----------------------------------------------------------------------
    println!("\n[6] Evaluating learned policies on the dataset...");

    // Evaluate CQL policy
    let mut cql_actions = [0usize; 5];
    let mut std_actions = [0usize; 5];

    for t in &all {
        let cql_a = cql_agent.select_action(&t.state, false);
        let std_a = std_agent.select_action(&t.state, false);
        cql_actions[cql_a] += 1;
        std_actions[std_a] += 1;
    }

    println!("    CQL policy actions:  StrongSell={}, Sell={}, Hold={}, Buy={}, StrongBuy={}",
        cql_actions[0], cql_actions[1], cql_actions[2], cql_actions[3], cql_actions[4]);
    println!("    Std policy actions:  StrongSell={}, Sell={}, Hold={}, Buy={}, StrongBuy={}",
        std_actions[0], std_actions[1], std_actions[2], std_actions[3], std_actions[4]);

    // -----------------------------------------------------------------------
    // Step 7: CQL penalty analysis
    // -----------------------------------------------------------------------
    println!("\n[7] CQL penalty analysis...");
    let eval_batch = buffer.sample(50);
    let (total_loss, td_loss, cql_penalty) = cql_agent.compute_cql_loss(&eval_batch);
    println!("    Total loss:   {:.6}", total_loss);
    println!("    TD loss:      {:.6}", td_loss);
    println!("    CQL penalty:  {:.6}", cql_penalty);
    println!("    Alpha * CQL:  {:.6}", cql_agent.config.alpha * cql_penalty);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== Summary ===");
    println!("CQL learns conservative Q-values from offline data, avoiding");
    println!("overestimation of out-of-distribution actions. This leads to");
    println!("safer trading policies that stick to well-tested strategies.");
    println!("\nKey benefits for trading:");
    println!("  - Zero market risk during training");
    println!("  - Naturally conservative position sizing");
    println!("  - Robust to distribution shift between historical and live data");
    println!("  - Lower Q-values = less likely to take extreme untested positions");
}
