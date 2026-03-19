use anyhow::Result;
use behavioral_cloning_trading::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Chapter 307: Behavioral Cloning for Trading ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Fetch data from Bybit (or use synthetic if API unavailable)
    // -------------------------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT data from Bybit...");

    let (prices, volumes) = match fetch_bybit_data().await {
        Ok((p, v)) => {
            println!("  Fetched {} candles from Bybit", p.len());
            (p, v)
        }
        Err(e) => {
            println!("  Bybit API unavailable ({}), using synthetic data", e);
            let (p, v) = generate_synthetic_prices(50000.0, 500, 0.0005, 0.015);
            println!("  Generated {} synthetic price points", p.len());
            (p, v)
        }
    };

    // -------------------------------------------------------------------------
    // Step 2: Generate expert demonstrations
    // -------------------------------------------------------------------------
    println!("\nStep 2: Generating expert demonstrations (trend-following)...");

    let strategy = ExpertStrategy::TrendFollowing {
        short_window: 5,
        long_window: 20,
    };
    let dataset = ExpertDataset::from_prices(&prices, &volumes, strategy);
    println!("  Expert dataset: {} samples, state_dim={}", dataset.len(), dataset.state_dim);

    // Analyze expert action distribution
    let mut buy_count = 0;
    let mut hold_count = 0;
    let mut sell_count = 0;
    for sample in &dataset.samples {
        match sample.action {
            Action::Buy => buy_count += 1,
            Action::Hold => hold_count += 1,
            Action::Sell => sell_count += 1,
        }
    }
    println!(
        "  Action distribution: Buy={} ({:.1}%), Hold={} ({:.1}%), Sell={} ({:.1}%)",
        buy_count,
        buy_count as f64 / dataset.len() as f64 * 100.0,
        hold_count,
        hold_count as f64 / dataset.len() as f64 * 100.0,
        sell_count,
        sell_count as f64 / dataset.len() as f64 * 100.0,
    );

    // -------------------------------------------------------------------------
    // Step 3: Train vanilla BC policy
    // -------------------------------------------------------------------------
    println!("\nStep 3: Training vanilla BC policy...");

    let mut bc_policy = BCPolicy::new(8, 64, 32, Action::num_actions());
    let bc_losses = bc_policy.train(&dataset, 50, 0.005, 32);

    println!(
        "  Initial loss: {:.4}, Final loss: {:.4}",
        bc_losses.first().unwrap_or(&0.0),
        bc_losses.last().unwrap_or(&0.0),
    );

    let bc_accuracy = bc_policy.evaluate(&dataset);
    println!("  Training accuracy: {:.2}%", bc_accuracy * 100.0);

    // -------------------------------------------------------------------------
    // Step 4: Train DAgger policy
    // -------------------------------------------------------------------------
    println!("\nStep 4: Training DAgger policy...");

    let dagger_dataset = ExpertDataset::from_prices(&prices, &volumes, strategy);
    let dagger_policy = BCPolicy::new(8, 64, 32, Action::num_actions());
    let config = DAggerConfig {
        num_iterations: 5,
        initial_beta: 0.7,
        beta_decay: 0.5,
        epochs_per_iter: 20,
        learning_rate: 0.005,
        batch_size: 32,
        rollout_steps: 200,
    };

    let mut trainer = DAggerTrainer::new(dagger_policy, dagger_dataset, config);
    let _dagger_losses = trainer.run(&prices, &volumes, strategy);

    println!("  DAgger iteration accuracies:");
    for (i, acc) in trainer.iteration_accuracies.iter().enumerate() {
        println!("    Iteration {}: {:.2}%", i + 1, acc * 100.0);
    }
    println!(
        "  Final dataset size: {} (grew from aggregation)",
        trainer.dataset.len()
    );

    // -------------------------------------------------------------------------
    // Step 5: Covariate shift analysis
    // -------------------------------------------------------------------------
    println!("\nStep 5: Covariate shift analysis...");

    let expert_states: Vec<Vec<f64>> = dataset.samples.iter().map(|s| s.state.clone()).collect();

    // Simulate policy-visited states
    let lookback = 20;
    let policy_states: Vec<Vec<f64>> = (lookback..prices.len())
        .map(|i| ExpertDataset::extract_features(&prices, &volumes, i, lookback))
        .collect();

    // Note: since we use the same price series, the states are similar.
    // In practice, the policy's actions would change the trajectory.
    let kl_bc = CovariateShiftAnalyzer::estimate_kl_divergence(&expert_states, &policy_states);
    println!("  KL divergence (expert vs policy states): {:.6}", kl_bc);

    let drift = CovariateShiftAnalyzer::feature_drift(&expert_states, &policy_states);
    let feature_names = [
        "return_1", "return_5", "short_ma", "long_ma",
        "volatility", "vol_ratio", "momentum", "price_pos",
    ];
    println!("  Feature-wise drift (standardized):");
    for (name, d) in feature_names.iter().zip(drift.iter()) {
        println!("    {}: {:.4}", name, d);
    }

    // -------------------------------------------------------------------------
    // Step 6: Compare BC vs DAgger vs Random
    // -------------------------------------------------------------------------
    println!("\nStep 6: Strategy comparison (simulated returns)...");

    let bc_return = simulate_policy(&bc_policy, &prices, &volumes, lookback);
    let dagger_return = simulate_policy(&trainer.policy, &prices, &volumes, lookback);

    // Average random over multiple runs
    let mut random_returns = Vec::new();
    for _ in 0..20 {
        random_returns.push(simulate_random_policy(&prices, prices.len() - lookback));
    }
    let avg_random = random_returns.iter().sum::<f64>() / random_returns.len() as f64;

    println!("  BC policy return:     {:.2}%", bc_return * 100.0);
    println!("  DAgger policy return: {:.2}%", dagger_return * 100.0);
    println!("  Random policy return: {:.2}% (avg of 20 runs)", avg_random * 100.0);

    println!("\n=== Done ===");
    Ok(())
}

/// Attempt to fetch real data from Bybit
async fn fetch_bybit_data() -> Result<(Vec<f64>, Vec<f64>)> {
    let client = BybitClient::new();
    let klines = client.fetch_klines("BTCUSDT", "15", 500).await?;

    if klines.is_empty() {
        anyhow::bail!("No data returned");
    }

    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
    Ok((prices, volumes))
}
