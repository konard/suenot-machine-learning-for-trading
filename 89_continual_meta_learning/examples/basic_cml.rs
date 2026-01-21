//! Basic CML Example
//!
//! This example demonstrates the basic usage of the Continual Meta-Learning algorithm:
//! - Creating a CML model
//! - Training on multiple tasks (market regimes)
//! - Evaluating forgetting prevention
//!
//! Run with: cargo run --example basic_cml

use continual_meta_learning::prelude::*;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Continual Meta-Learning Basic Example ===\n");

    // Create CML configuration
    let config = CMLConfig {
        input_size: 4,      // Simple 4-feature input
        hidden_size: 16,    // Small hidden layer
        output_size: 1,     // Binary classification (buy/sell)
        inner_lr: 0.01,     // Inner loop learning rate
        outer_lr: 0.001,    // Outer loop learning rate
        inner_steps: 5,     // Adaptation steps
        memory_size: 100,   // Experience buffer size
        ewc_lambda: 1000.0, // EWC regularization strength
    };

    println!("Configuration:");
    println!("  Input size: {}", config.input_size);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Inner LR: {}", config.inner_lr);
    println!("  Outer LR: {}", config.outer_lr);
    println!("  EWC Lambda: {}\n", config.ewc_lambda);

    // Create the CML learner
    let mut learner = ContinualMetaLearner::new(config);

    // Create synthetic tasks representing different market regimes
    let tasks = create_synthetic_tasks();

    println!("Created {} tasks (market regimes)\n", tasks.len());

    // Train on each task sequentially
    println!("Training...\n");

    for (task_idx, task) in tasks.iter().enumerate() {
        println!(
            "Task {}: {:?} (support={}, query={})",
            task_idx + 1,
            task.regime,
            task.support_size(),
            task.query_size()
        );

        // Meta-train on this task
        let loss = learner.meta_train_step(task);
        println!("  Loss after training: {:.6}", loss);

        // Consolidate knowledge
        learner.consolidate();
    }

    println!("\n=== Evaluating Forgetting ===\n");

    // Evaluate on all previous tasks
    let forgetting = learner.evaluate_forgetting(&tasks);

    for (i, loss) in forgetting.iter().enumerate() {
        println!("Task {} ({:?}): Loss = {:.6}", i + 1, tasks[i].regime, loss);
    }

    // Calculate average forgetting
    let avg_forgetting: f64 = forgetting.iter().sum::<f64>() / forgetting.len() as f64;
    println!("\nAverage loss across tasks: {:.6}", avg_forgetting);

    // Get learner statistics
    let stats = learner.stats();
    println!("\n=== Learner Statistics ===\n");
    println!("Total training steps: {}", stats.total_steps);
    println!("Tasks seen: {}", stats.tasks_seen);
    println!("Memory buffer size: {}", stats.memory_size);
    println!("EWC initialized: {}", stats.ewc_initialized);

    println!("\n=== Fast Adaptation Demo ===\n");

    // Demonstrate fast adaptation to new data
    let new_task = &tasks[0]; // Reuse first task
    let adapted_params = learner.adapt(&new_task.support);

    // Compare predictions before and after adaptation
    let test_input = vec![0.5, 0.3, 0.2, 0.1];

    let pred_before = learner.predict(&test_input, None);
    let pred_after = learner.predict(&test_input, Some(&adapted_params));

    println!("Test input: {:?}", test_input);
    println!("Prediction (before adaptation): {:.4}", pred_before[0]);
    println!("Prediction (after adaptation):  {:.4}", pred_after[0]);

    println!("\nDone!");
}

/// Create synthetic tasks for demonstration.
fn create_synthetic_tasks() -> Vec<continual_meta_learning::continual::learner::Task> {
    use continual_meta_learning::continual::learner::Task;

    let regimes = [
        MarketRegime::Bull,
        MarketRegime::Bear,
        MarketRegime::HighVolatility,
        MarketRegime::LowVolatility,
    ];

    regimes
        .iter()
        .enumerate()
        .map(|(task_id, &regime)| {
            // Create support set (training data for this task)
            let support: Vec<Experience> = (0..10)
                .map(|i| {
                    let input = generate_regime_features(regime, i);
                    let target = generate_regime_target(regime);
                    Experience::new(input, target, task_id)
                })
                .collect();

            // Create query set (test data for this task)
            let query: Vec<Experience> = (10..15)
                .map(|i| {
                    let input = generate_regime_features(regime, i);
                    let target = generate_regime_target(regime);
                    Experience::new(input, target, task_id)
                })
                .collect();

            Task::new(support, query, task_id, regime)
        })
        .collect()
}

/// Generate features characteristic of a market regime.
fn generate_regime_features(regime: MarketRegime, seed: usize) -> Vec<f64> {
    let base = seed as f64 * 0.1;

    match regime {
        MarketRegime::Bull => {
            // Bullish: positive momentum, low volatility
            vec![0.5 + base, 0.7 + base * 0.1, 0.3, 0.2]
        }
        MarketRegime::Bear => {
            // Bearish: negative momentum, moderate volatility
            vec![-0.5 + base, 0.3 - base * 0.1, 0.5, 0.4]
        }
        MarketRegime::HighVolatility => {
            // High volatility: mixed signals, high vol indicator
            vec![0.0 + base * 0.5, 0.5, 0.8, 0.7]
        }
        MarketRegime::LowVolatility => {
            // Low volatility: stable, low vol indicator
            vec![0.1 + base * 0.05, 0.5, 0.1, 0.1]
        }
        MarketRegime::Sideways => {
            // Sideways: neutral
            vec![0.0, 0.5, 0.3, 0.3]
        }
    }
}

/// Generate target for a market regime.
fn generate_regime_target(regime: MarketRegime) -> Vec<f64> {
    match regime {
        MarketRegime::Bull => vec![1.0],      // Buy signal
        MarketRegime::Bear => vec![-1.0],     // Sell signal
        MarketRegime::HighVolatility => vec![0.0], // Hold
        MarketRegime::LowVolatility => vec![0.5],  // Weak buy
        MarketRegime::Sideways => vec![0.0],  // Hold
    }
}
