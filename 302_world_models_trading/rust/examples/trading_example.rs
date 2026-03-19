use anyhow::Result;
use ndarray::Array1;
use world_models_trading::*;

fn main() -> Result<()> {
    println!("=== World Models for Trading ===\n");

    // Step 1: Fetch BTCUSDT data from Bybit
    println!("Step 1: Fetching BTCUSDT kline data from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("  Fetched {} candles", c.len());
            if let Some(last) = c.last() {
                println!("  Latest close: ${:.2}, volume: {:.2}", last.close, last.volume);
            }
            c
        }
        Err(e) => {
            println!("  Warning: Could not fetch from Bybit ({}). Using synthetic data.", e);
            generate_synthetic_candles(200)
        }
    };

    // Step 2: Convert to features and normalize
    println!("\nStep 2: Preparing features...");
    let features = candles_to_features(&candles);
    let (normalized, mean, std) = normalize_features(&features);
    println!("  Generated {} feature vectors, dim={}", normalized.len(), normalized[0].len());
    println!("  Feature means: {:?}", mean.to_vec());
    println!("  Feature stds:  {:?}", std.to_vec());

    let obs_dim = 4;
    let latent_dim = 8;
    let hidden_dim = 32;
    let action_dim = 1;
    let num_mixtures = 5;

    // Step 3: Train VAE on market features
    println!("\nStep 3: Training VAE...");
    let mut model = WorldModel::new(obs_dim, latent_dim, hidden_dim, action_dim, num_mixtures);

    let vae_epochs = 50;
    for epoch in 0..vae_epochs {
        let loss = model.vae.train_step(&normalized, 0.01);
        if epoch % 10 == 0 || epoch == vae_epochs - 1 {
            println!("  VAE Epoch {}: loss = {:.4}", epoch, loss);
        }
    }

    // Encode all observations to latent space
    println!("\nStep 4: Encoding observations to latent space...");
    let latent_states: Vec<Array1<f64>> = normalized
        .iter()
        .map(|x| model.encode_observation(x))
        .collect();
    println!("  Encoded {} observations to {}-dim latent vectors", latent_states.len(), latent_dim);

    // Show latent statistics
    let latent_means: Vec<f64> = latent_states.iter().map(|z| z.mean().unwrap_or(0.0)).collect();
    let z_mean = latent_means.iter().sum::<f64>() / latent_means.len() as f64;
    let z_std = (latent_means.iter().map(|v| (v - z_mean).powi(2)).sum::<f64>() / latent_means.len() as f64).sqrt();
    println!("  Latent mean of means: {:.4}, std: {:.4}", z_mean, z_std);

    // Step 5: Test MDN-RNN predictions
    println!("\nStep 5: Testing MDN-RNN predictions...");
    let mut rnn_state = LSTMState::zeros(hidden_dim);
    let action = Array1::from_vec(vec![0.0]);

    for i in 0..5.min(latent_states.len()) {
        let (params, new_state) = model.mdnrnn.forward(&latent_states[i], &action, &rnn_state);
        let next_z = model.mdnrnn.sample(&params);
        println!(
            "  Step {}: pi_max={:.3}, predicted z_mean={:.4}",
            i,
            params.pi.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            next_z.mean().unwrap_or(0.0)
        );
        rnn_state = new_state;
    }

    // Step 6: Dream rollout
    println!("\nStep 6: Running dream rollouts...");
    let num_dreams = 10;
    let dream_steps = 50;
    let mut dream_rewards = Vec::new();

    for i in 0..num_dreams {
        let z0 = &latent_states[i % latent_states.len()];
        let reward = model.dream_rollout(z0, dream_steps);
        dream_rewards.push(reward);
    }

    let avg_reward = dream_rewards.iter().sum::<f64>() / num_dreams as f64;
    let max_reward = dream_rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_reward = dream_rewards.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("  {} dreams, {} steps each", num_dreams, dream_steps);
    println!("  Reward: avg={:.6}, min={:.6}, max={:.6}", avg_reward, min_reward, max_reward);

    // Step 7: Optimize controller with CMA-ES
    println!("\nStep 7: Optimizing controller with CMA-ES in dream environment...");
    let initial_states: Vec<Array1<f64>> = latent_states.iter().take(5).cloned().collect();
    let generations = 20;
    let population_size = 16;

    let fitness_history = model.optimize_controller(
        &initial_states,
        dream_steps,
        generations,
        population_size,
    );

    for (gen, fitness) in fitness_history.iter().enumerate() {
        if gen % 5 == 0 || gen == fitness_history.len() - 1 {
            println!("  Generation {}: best fitness = {:.6}", gen, fitness);
        }
    }

    // Step 8: Evaluate optimized controller on held-out data
    println!("\nStep 8: Evaluating optimized controller...");
    let test_start = latent_states.len() / 2;
    let test_states: Vec<&Array1<f64>> = latent_states[test_start..].iter().collect();

    let mut rnn_state = LSTMState::zeros(hidden_dim);
    let mut total_pnl = 0.0;
    let mut positions = Vec::new();

    for i in 0..test_states.len().saturating_sub(1) {
        let z = test_states[i];
        let action = model.controller.act(z, &rnn_state.h);
        let position = action[0];
        positions.push(position);

        // Simulate PnL using latent space changes
        let z_next = test_states[i + 1];
        let price_change = z_next.mean().unwrap_or(0.0) - z.mean().unwrap_or(0.0);
        total_pnl += position * price_change;

        // Update RNN state
        let (_, new_state) = model.mdnrnn.forward(z, &action, &rnn_state);
        rnn_state = new_state;
    }

    let avg_position = positions.iter().sum::<f64>() / positions.len().max(1) as f64;
    let position_std = (positions.iter().map(|p| (p - avg_position).powi(2)).sum::<f64>()
        / positions.len().max(1) as f64).sqrt();

    println!("  Test period: {} steps", test_states.len());
    println!("  Total PnL (latent): {:.6}", total_pnl);
    println!("  Avg position: {:.4}, Std: {:.4}", avg_position, position_std);
    println!("  Position range: [{:.4}, {:.4}]",
        positions.iter().cloned().fold(f64::INFINITY, f64::min),
        positions.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    println!("\n=== World Models Trading Complete ===");
    Ok(())
}

/// Generate synthetic candle data for testing when API is unavailable
fn generate_synthetic_candles(count: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut candles = Vec::with_capacity(count);

    for i in 0..count {
        let ret = (rng.gen::<f64>() - 0.5) * 0.02;
        let new_price = price * (1.0 + ret);
        let high = new_price * (1.0 + rng.gen::<f64>() * 0.005);
        let low = new_price * (1.0 - rng.gen::<f64>() * 0.005);
        let volume = 100.0 + rng.gen::<f64>() * 900.0;

        candles.push(Candle {
            timestamp: (1700000000 + i * 3600) as u64,
            open: price,
            high,
            low,
            close: new_price,
            volume,
        });
        price = new_price;
    }
    candles
}
