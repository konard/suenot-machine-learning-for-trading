use rand::Rng;
use trajectory_transformer::*;

fn main() {
    println!("=== Trajectory Transformer for Trading ===\n");

    // --- Step 1: Generate synthetic market data (simulating Bybit BTCUSDT) ---
    // In production, use: BybitClient::new().fetch_klines("BTCUSDT", "60", 200)
    println!("Step 1: Generating synthetic market data...");
    let candles = generate_synthetic_candles(200);
    println!("  Generated {} candles", candles.len());

    // --- Step 2: Extract features and build trajectory dataset ---
    println!("\nStep 2: Building trajectory dataset...");
    let lookback = 10;
    let states = extract_trading_states(&candles, lookback);
    let actions = generate_actions(&candles, lookback);
    let rewards = compute_rewards(&candles, &actions, lookback);

    let n = states.len().min(actions.len()).min(rewards.len());
    println!("  States: {}, Actions: {}, Rewards: {}", states.len(), actions.len(), rewards.len());
    println!("  State dimensions: {}", states[0].len());
    println!("  Sample state: {:?}", &states[0]);
    println!("  Sample action: {:?}", &actions[0]);
    println!("  Sample reward: {:.4}", rewards[0]);

    // --- Step 3: Fit tokenizer ---
    println!("\nStep 3: Fitting trajectory tokenizer...");
    let vocab_size = 50;
    let tokenizer = TrajectoryTokenizer::fit(
        &states[..n],
        &actions[..n],
        &rewards[..n],
        vocab_size,
    );
    println!("  Vocab size: {}", tokenizer.vocab_size);
    println!("  Tokens per step: {}", tokenizer.tokens_per_step());

    // Tokenize a sample trajectory
    let sample_tokens = tokenizer.tokenize_trajectory(
        &states[..5],
        &actions[..5],
        &rewards[..5],
    );
    println!("  Sample trajectory tokens (5 steps): {:?}", sample_tokens);

    // --- Step 4: Create and train Trajectory Transformer ---
    println!("\nStep 4: Training Trajectory Transformer...");
    let config = TransformerConfig {
        vocab_size,
        embed_dim: 32,
        num_heads: 4,
        num_layers: 1,
        max_seq_len: 256,
        ff_dim: 128,
    };
    let mut model = TrajectoryTransformer::new(config);

    // Create training trajectories (sliding windows)
    let window_size = 10;
    let mut training_sequences: Vec<Vec<usize>> = Vec::new();
    for start in 0..n.saturating_sub(window_size) {
        let end = (start + window_size).min(n);
        let traj = tokenizer.tokenize_trajectory(
            &states[start..end],
            &actions[start..end],
            &rewards[start..end],
        );
        training_sequences.push(traj);
    }
    println!("  Training sequences: {}", training_sequences.len());

    // Train for a few epochs
    let epochs = 5;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut count = 0;
        for seq in &training_sequences {
            if seq.len() > 2 {
                let loss = model.train_step(seq, 0.01);
                total_loss += loss;
                count += 1;
            }
        }
        println!("  Epoch {}/{}: avg loss = {:.4}", epoch + 1, epochs, total_loss / count.max(1) as f64);
    }

    // --- Step 5: Plan with beam search ---
    println!("\nStep 5: Planning with beam search...");
    let beam = BeamSearch::new(4, 10);
    let current_state_tokens = tokenizer.tokenize_state(&states[n - 1]);
    println!("  Current state tokens: {:?}", current_state_tokens);

    let results = beam.search(&model, &current_state_tokens);
    println!("  Beam search results: {} candidates", results.len());
    for (i, candidate) in results.iter().enumerate() {
        println!(
            "    Beam {}: {} tokens, log_prob = {:.4}",
            i + 1,
            candidate.tokens.len(),
            candidate.log_prob
        );
    }

    // --- Step 6: Return-conditioned planning ---
    println!("\nStep 6: Return-conditioned planning...");
    let planner = ReturnConditionedPlanner::new(
        model,
        tokenizer,
        4,    // beam width
        14,   // plan horizon (2 timesteps * 7 tokens/step)
    );

    let target_return = 2.0; // Target a positive return
    let current_state = &states[n - 1];
    println!("  Current state: {:?}", current_state);
    println!("  Target return: {:.2}", target_return);

    let planned_actions = planner.plan(current_state, target_return);
    println!("  Planned actions ({} steps):", planned_actions.len());
    for (i, (action, reward)) in planned_actions.iter().enumerate() {
        let action_label = match action[0] as i32 {
            a if a >= 2 => "STRONG BUY",
            a if a >= 1 => "BUY",
            a if a <= -2 => "STRONG SELL",
            a if a <= -1 => "SELL",
            _ => "HOLD",
        };
        println!(
            "    Step {}: action = {:.2} ({}), expected reward = {:.4}",
            i + 1,
            action[0],
            action_label,
            reward
        );
    }

    // --- Step 7: Show Bybit integration info ---
    println!("\n--- Bybit Integration ---");
    println!("To use with real data, replace synthetic candles with:");
    println!("  let client = BybitClient::new();");
    println!("  let candles = client.fetch_klines(\"BTCUSDT\", \"60\", 200).unwrap();");
    println!("\nDone!");
}

/// Generate synthetic candle data resembling BTC price action
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    for i in 0..n {
        let trend = (i as f64 * 0.02).sin() * 0.002;
        let noise: f64 = rng.gen_range(-0.015..0.015);
        let ret = trend + noise;

        let open = price;
        let close = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.001..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.001..0.01));
        let volume = 1000.0 + rng.gen_range(0.0..2000.0);

        candles.push(Candle {
            timestamp: 1700000000000 + i as u64 * 3600000,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}
