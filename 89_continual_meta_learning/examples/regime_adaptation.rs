//! Market Regime Adaptation Example
//!
//! This example demonstrates how CML adapts to changing market regimes:
//! - Detecting market regime changes
//! - Quickly adapting to new regimes
//! - Maintaining knowledge of previous regimes
//!
//! Run with: cargo run --example regime_adaptation

use continual_meta_learning::prelude::*;
use continual_meta_learning::continual::learner::Task;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Market Regime Adaptation Example ===\n");

    // Create CML learner with trading-optimized config
    let config = CMLConfig {
        input_size: 9,      // Standard trading features
        hidden_size: 32,
        output_size: 1,
        inner_lr: 0.02,     // Faster adaptation
        outer_lr: 0.001,
        inner_steps: 10,    // More adaptation steps
        memory_size: 500,
        ewc_lambda: 500.0,  // Moderate EWC strength
    };

    let mut learner = ContinualMetaLearner::new(config);

    // Simulate a sequence of regime changes
    println!("Simulating market regime changes...\n");

    let regime_sequence = vec![
        (MarketRegime::Bull, 50),        // Start in bull market
        (MarketRegime::HighVolatility, 20), // Transition to high volatility
        (MarketRegime::Bear, 40),        // Bear market
        (MarketRegime::LowVolatility, 30), // Low volatility consolidation
        (MarketRegime::Bull, 30),        // Return to bull market
    ];

    let mut all_tasks = Vec::new();
    let mut regime_losses: Vec<(MarketRegime, Vec<f64>)> = Vec::new();

    for (task_id, (regime, duration)) in regime_sequence.iter().enumerate() {
        println!("--- Regime: {:?} (duration: {} steps) ---", regime, duration);

        // Create task for this regime
        let task = create_regime_task(*regime, task_id, *duration);
        all_tasks.push(task.clone());

        // Train on this regime
        let mut losses = Vec::new();
        for epoch in 0..5 {
            let loss = learner.meta_train_step(&task);
            losses.push(loss);

            if epoch == 0 || epoch == 4 {
                println!("  Epoch {}: loss = {:.6}", epoch + 1, loss);
            }
        }

        // Consolidate after learning regime
        learner.consolidate();

        // Track losses for this regime
        regime_losses.push((*regime, losses));

        // Evaluate retention of previous regimes
        if task_id > 0 {
            println!("  Evaluating retention of previous regimes:");
            let forgetting = learner.evaluate_forgetting(&all_tasks[..task_id]);
            for (i, loss) in forgetting.iter().enumerate() {
                println!(
                    "    Task {} ({:?}): loss = {:.6}",
                    i + 1,
                    all_tasks[i].regime,
                    loss
                );
            }
        }

        // Perform experience replay
        if let Some(replay_loss) = learner.replay_step(16) {
            println!("  Replay loss: {:.6}", replay_loss);
        }

        println!();
    }

    // Final evaluation
    println!("=== Final Evaluation ===\n");

    println!("Performance by regime:");
    let final_forgetting = learner.evaluate_forgetting(&all_tasks);

    for (i, loss) in final_forgetting.iter().enumerate() {
        println!(
            "  {:?}: loss = {:.6}",
            all_tasks[i].regime,
            loss
        );
    }

    // Test rapid adaptation
    println!("\n=== Rapid Adaptation Test ===\n");

    // Simulate sudden regime change
    let new_regime = MarketRegime::Bear;
    println!("Sudden change to {:?} regime", new_regime);

    // Create small adaptation set (few-shot learning)
    let adaptation_set: Vec<Experience> = (0..5)
        .map(|i| {
            let input = generate_features(new_regime, i);
            let target = generate_target(new_regime);
            Experience::new(input, target, 99)
        })
        .collect();

    // Adapt with just 5 samples
    let adapted_params = learner.adapt(&adaptation_set);

    // Compare predictions
    let test_input = generate_features(new_regime, 100);

    println!("Testing on {:?} regime input", new_regime);
    println!("  Before adaptation: {:.4}", learner.predict(&test_input, None)[0]);
    println!(
        "  After adaptation:  {:.4}",
        learner.predict(&test_input, Some(&adapted_params))[0]
    );
    println!("  Expected target:   {:.4}", generate_target(new_regime)[0]);

    // Memory buffer statistics
    println!("\n=== Memory Buffer Statistics ===\n");
    let memory_stats = learner.memory().stats();
    println!("Buffer size: {} / {}", memory_stats.size, memory_stats.capacity);
    println!("Tasks in memory: {}", memory_stats.num_tasks);
    println!("Total samples seen: {}", memory_stats.total_seen);
    println!("Average importance: {:.4}", memory_stats.avg_importance);

    println!("\nTask distribution in memory:");
    for (task_id, count) in &memory_stats.task_counts {
        println!("  Task {}: {} samples", task_id, count);
    }

    // EWC statistics
    println!("\n=== EWC Statistics ===\n");
    let ewc_stats = learner.ewc().stats();
    println!("EWC initialized: {}", ewc_stats.initialized);
    println!("Parameter count: {}", ewc_stats.param_count);
    println!("Fisher mean: {:.6}", ewc_stats.fisher_mean);
    println!("Fisher max: {:.6}", ewc_stats.fisher_max);
    println!("Lambda: {}", ewc_stats.lambda);
    println!("Samples used: {}", ewc_stats.samples_used);

    println!("\nDone!");
}

/// Create a task for a specific market regime.
fn create_regime_task(regime: MarketRegime, task_id: usize, samples: usize) -> Task {
    let support_size = samples * 2 / 3;
    let query_size = samples - support_size;

    let support: Vec<Experience> = (0..support_size)
        .map(|i| {
            let input = generate_features(regime, i);
            let target = generate_target(regime);
            Experience::new(input, target, task_id)
        })
        .collect();

    let query: Vec<Experience> = (support_size..support_size + query_size)
        .map(|i| {
            let input = generate_features(regime, i);
            let target = generate_target(regime);
            Experience::new(input, target, task_id)
        })
        .collect();

    Task::new(support, query, task_id, regime)
}

/// Generate features for a given market regime.
fn generate_features(regime: MarketRegime, seed: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let noise = || rng.gen_range(-0.1..0.1);

    // Features: [return, sma_ratio, ema_short, ema_long, rsi, volatility, momentum, price_to_sma, macd]
    match regime {
        MarketRegime::Bull => vec![
            0.02 + noise(),   // Positive returns
            1.05 + noise(),   // Price above SMA
            0.03 + noise(),   // Short EMA positive
            0.02 + noise(),   // Long EMA positive
            65.0 + noise() * 100.0, // RSI > 50
            0.01 + noise().abs(), // Low volatility
            0.05 + noise(),   // Positive momentum
            0.03 + noise(),   // Price above SMA
            0.01 + noise(),   // Positive MACD
        ],
        MarketRegime::Bear => vec![
            -0.02 + noise(),  // Negative returns
            0.95 + noise(),   // Price below SMA
            -0.03 + noise(),  // Short EMA negative
            -0.02 + noise(),  // Long EMA negative
            35.0 + noise() * 100.0, // RSI < 50
            0.02 + noise().abs(), // Moderate volatility
            -0.05 + noise(),  // Negative momentum
            -0.03 + noise(),  // Price below SMA
            -0.01 + noise(),  // Negative MACD
        ],
        MarketRegime::HighVolatility => vec![
            0.0 + noise() * 2.0,  // Mixed returns
            1.0 + noise(),    // Price near SMA
            0.0 + noise() * 2.0,  // Mixed short EMA
            0.0 + noise(),    // Mixed long EMA
            50.0 + noise() * 150.0, // RSI varies
            0.05 + noise().abs() * 2.0, // High volatility
            0.0 + noise() * 2.0,  // Mixed momentum
            0.0 + noise(),    // Price near SMA
            0.0 + noise(),    // Mixed MACD
        ],
        MarketRegime::LowVolatility => vec![
            0.001 + noise() * 0.5, // Tiny returns
            1.0 + noise() * 0.5,  // Price very close to SMA
            0.001 + noise() * 0.5, // Small short EMA
            0.001 + noise() * 0.5, // Small long EMA
            50.0 + noise() * 50.0, // RSI near 50
            0.005 + noise().abs() * 0.5, // Very low volatility
            0.001 + noise() * 0.5, // Minimal momentum
            0.001 + noise() * 0.5, // Price very close to SMA
            0.0 + noise() * 0.5,  // Minimal MACD
        ],
        MarketRegime::Sideways => vec![
            0.0 + noise(),
            1.0 + noise(),
            0.0 + noise(),
            0.0 + noise(),
            50.0 + noise() * 100.0,
            0.015 + noise().abs(),
            0.0 + noise(),
            0.0 + noise(),
            0.0 + noise(),
        ],
    }
}

/// Generate target for a given market regime.
fn generate_target(regime: MarketRegime) -> Vec<f64> {
    match regime {
        MarketRegime::Bull => vec![1.0],       // Strong buy
        MarketRegime::Bear => vec![-1.0],      // Strong sell
        MarketRegime::HighVolatility => vec![0.0], // Hold (uncertain)
        MarketRegime::LowVolatility => vec![0.2],  // Weak buy (accumulate)
        MarketRegime::Sideways => vec![0.0],   // Hold
    }
}
