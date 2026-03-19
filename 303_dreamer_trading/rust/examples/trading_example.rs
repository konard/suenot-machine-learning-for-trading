//! # Dreamer Trading Example
//!
//! This example demonstrates the full Dreamer trading workflow:
//! 1. Fetch BTCUSDT data from Bybit (or use synthetic data)
//! 2. Train a world model on real market data
//! 3. Learn a trading policy entirely in imagined trajectories
//! 4. Evaluate the policy on held-out real market data

use dreamer_trading::{
    BybitClient, DreamerAgent, DreamerConfig, generate_synthetic_klines, Kline,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Dreamer Trading: World Model RL for Algorithmic Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // -----------------------------------------------------------------------
    println!("[Step 1] Fetching BTCUSDT market data...");

    let klines: Vec<Kline> = match fetch_bybit_data().await {
        Ok(data) => {
            println!("  Fetched {} klines from Bybit API", data.len());
            data
        }
        Err(e) => {
            println!("  Bybit API unavailable ({}), using synthetic data", e);
            let data = generate_synthetic_klines(500, 42);
            println!("  Generated {} synthetic klines", data.len());
            data
        }
    };

    println!(
        "  Price range: {:.2} - {:.2}",
        klines.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
        klines.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max)
    );
    println!();

    // Split into train/test
    let split = klines.len() * 80 / 100;
    let train_data = &klines[..split];
    let test_data = &klines[split..];
    println!(
        "  Train: {} candles, Test: {} candles\n",
        train_data.len(),
        test_data.len()
    );

    // -----------------------------------------------------------------------
    // Step 2: Configure and create the Dreamer agent
    // -----------------------------------------------------------------------
    println!("[Step 2] Configuring Dreamer agent...");

    let config = DreamerConfig {
        deterministic_size: 64,
        stochastic_size: 8,
        num_categories: 8,
        hidden_size: 64,
        imagination_horizon: 10,
        kl_balance_alpha: 0.8,
        free_bits: 1.0,
        discount: 0.99,
        lambda_gae: 0.95,
        learning_rate: 3e-4,
        obs_dim: 5,
        num_actions: 3,
    };

    let mut agent = DreamerAgent::new(config.clone());
    println!("  Deterministic state size: {}", config.deterministic_size);
    println!("  Stochastic state: {} x {} categories", config.stochastic_size, config.num_categories);
    println!("  Imagination horizon: {} steps", config.imagination_horizon);
    println!("  KL balance alpha: {}", config.kl_balance_alpha);
    println!("  Free bits: {}", config.free_bits);
    println!();

    // -----------------------------------------------------------------------
    // Step 3: Train the world model on real market data
    // -----------------------------------------------------------------------
    println!("[Step 3] Training world model on market data...");

    let train_klines: Vec<Kline> = train_data.to_vec();
    let losses = agent.train_world_model(&train_klines, 30);

    if let (Some(first), Some(last)) = (losses.first(), losses.last()) {
        println!("  Initial loss: {:.6}", first);
        println!("  Final loss: {:.6}", last);
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 4: Train policy entirely in imagination
    // -----------------------------------------------------------------------
    println!("[Step 4] Training trading policy in imagination...");
    println!("  (No real market interaction during policy learning!)");

    let avg_return = agent.train_in_imagination(500);
    println!("  Average imagined return: {:.6}\n", avg_return);

    // -----------------------------------------------------------------------
    // Step 5: Evaluate on held-out real market data
    // -----------------------------------------------------------------------
    println!("[Step 5] Evaluating on test data...");

    let test_klines: Vec<Kline> = test_data.to_vec();
    let metrics = agent.evaluate(&test_klines);
    println!("{}\n", metrics);

    // -----------------------------------------------------------------------
    // Step 6: Compare with random baseline
    // -----------------------------------------------------------------------
    println!("[Step 6] Comparing with random baseline...");

    let random_config = DreamerConfig {
        deterministic_size: 16,
        stochastic_size: 2,
        num_categories: 4,
        hidden_size: 16,
        ..config.clone()
    };
    let random_agent = DreamerAgent::new(random_config);
    let random_metrics = random_agent.evaluate(&test_klines);
    println!("Random baseline:\n{}\n", random_metrics);

    // -----------------------------------------------------------------------
    // Step 7: Demonstrate world model imagination
    // -----------------------------------------------------------------------
    println!("[Step 7] Demonstrating imagination rollouts...");

    let mut rng = rand::thread_rng();
    let mut state = dreamer_trading::LatentState::zeros(&config);

    // Observe a few real data points to seed the world model
    println!("  Seeding world model with 10 real observations...");
    for i in 0..10.min(test_klines.len()) {
        let obs = test_klines[i].to_features();
        state = agent.world_model.observe_step(&state, 1, &obs, &mut rng);
    }

    // Now imagine forward without any real data
    println!("  Imagining {} steps into the future...", config.imagination_horizon);
    let mut imagined_rewards = Vec::new();
    for step in 0..config.imagination_horizon {
        let action = agent.actor.sample_action(&state, &mut rng);
        state = agent.world_model.imagine_step(&state, action, &mut rng);
        let reward = agent.world_model.predict_reward(&state);
        imagined_rewards.push(reward);
        let action_name = match action {
            0 => "SELL",
            1 => "HOLD",
            2 => "BUY",
            _ => "?",
        };
        println!(
            "    Step {}: action={}, predicted_reward={:.6}",
            step + 1,
            action_name,
            reward
        );
    }

    let total_imagined: f64 = imagined_rewards.iter().sum();
    println!(
        "  Total imagined reward over {} steps: {:.6}\n",
        config.imagination_horizon, total_imagined
    );

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("=== Summary ===");
    println!("The Dreamer agent learned a world model of BTCUSDT market dynamics,");
    println!("then trained a trading policy entirely in imagination.");
    println!("Key advantage: policy learning required ZERO real market interaction.");
    println!("The world model enables risk-free exploration of trading strategies.");

    Ok(())
}

/// Attempt to fetch real BTCUSDT data from Bybit.
async fn fetch_bybit_data() -> anyhow::Result<Vec<Kline>> {
    let client = BybitClient::new();
    let klines = client.fetch_klines("BTCUSDT", "5", 500).await?;
    if klines.is_empty() {
        anyhow::bail!("No data returned");
    }
    Ok(klines)
}
