//! Basic Continual Meta-Learning example.
//!
//! This example demonstrates:
//! - Creating a Continual MAML trainer with EWC and replay
//! - Learning multiple market regimes sequentially
//! - Evaluating forgetting after learning new regimes
//! - Adapting to unseen data

use continual_meta_learning::{
    data::bybit::SimulatedDataGenerator,
    data::features::FeatureGenerator,
    continual::algorithm::{ContinualMAMLTrainer, TaskData},
    model::network::TradingModel,
};

fn main() {
    println!("=== Basic Continual Meta-Learning for Trading ===\n");

    // Step 1: Create the model
    println!("Step 1: Creating trading model...");
    let input_size = 11;
    let hidden_size = 32;
    let output_size = 1;

    let model = TradingModel::new(input_size, hidden_size, output_size);
    println!("  Model created with {} parameters\n", model.num_parameters());

    // Step 2: Create Continual MAML trainer
    println!("Step 2: Setting up Continual MAML trainer...");
    let mut trainer = ContinualMAMLTrainer::new(
        model,
        0.01,   // inner_lr
        0.001,  // outer_lr
        5,      // inner_steps
        true,   // first_order (FOMAML)
        100.0,  // ewc_lambda
        50,     // replay_buffer_size
    );
    println!("  Inner LR: 0.01, Outer LR: 0.001");
    println!("  FOMAML: true, EWC lambda: 100.0");
    println!("  Replay buffer: 50 tasks\n");

    // Step 3: Generate sequential regime tasks
    println!("Step 3: Generating sequential market regimes...");
    let feature_gen = FeatureGenerator::default_window();

    let regimes: Vec<(&str, f64, f64)> = vec![
        ("Bull Market", 0.015, 0.0003),
        ("Bear Market", 0.02, -0.0003),
        ("High Volatility", 0.035, 0.0),
    ];

    let mut all_regime_tasks: Vec<Vec<TaskData>> = Vec::new();

    for (name, vol, trend) in &regimes {
        let klines = SimulatedDataGenerator::generate_trending_klines(200, 50000.0, *vol, *trend);
        let features = feature_gen.compute_features(&klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

        let split_idx = features.len() / 2;
        let mut tasks = Vec::new();

        for _ in 0..3 {
            let support_features = features[..split_idx].to_vec();
            let support_labels = returns[..split_idx].to_vec();
            let query_features = features[split_idx..].to_vec();
            let query_labels = returns[split_idx..returns.len().min(split_idx + query_features.len())].to_vec();

            if support_labels.len() == support_features.len()
                && query_labels.len() == query_features.len()
            {
                tasks.push(TaskData::new(
                    support_features,
                    support_labels,
                    query_features,
                    query_labels,
                    all_regime_tasks.len(),
                ));
            }
        }

        println!("  {}: {} tasks created", name, tasks.len());
        all_regime_tasks.push(tasks);
    }
    println!();

    // Step 4: Learn regimes sequentially
    println!("Step 4: Learning regimes sequentially...");
    for (i, (name, _, _)) in regimes.iter().enumerate() {
        println!("\n  --- Learning Regime {}: {} ---", i, name);
        let losses = trainer.learn_regime(&all_regime_tasks[i], i, 10, 5);
        println!("  Final loss: {:.6}", losses.last().unwrap_or(&0.0));
        println!("  Replay buffer: {} tasks", trainer.replay_buffer_size());
    }

    // Step 5: Evaluate forgetting
    println!("\n\nStep 5: Evaluating forgetting across all regimes...");
    for (i, (name, _, _)) in regimes.iter().enumerate() {
        let (loss, accuracy) = trainer.evaluate_regime(&all_regime_tasks[i]);
        println!(
            "  Regime {} ({}): Loss={:.6}, Accuracy={:.1}%",
            i, name, loss, accuracy * 100.0
        );
    }

    // Step 6: Adapt to new data
    println!("\nStep 6: Adapting to unseen market data...");
    let new_klines = SimulatedDataGenerator::generate_klines(100, 45000.0, 0.025);
    let new_features = feature_gen.compute_features(&new_klines);
    let new_closes: Vec<f64> = new_klines.iter().map(|k| k.close).collect();
    let new_returns: Vec<f64> = new_closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

    let few_shot_features = new_features[..10].to_vec();
    let few_shot_labels = new_returns[..10].to_vec();

    println!("  Adapting with {} examples...", few_shot_features.len());
    let adapted_model = trainer.adapt(&few_shot_features, &few_shot_labels, Some(5));

    let mut correct_direction = 0;
    let mut total_predictions = 0;

    for i in 10..new_features.len().min(new_returns.len()) {
        let prediction = adapted_model.predict(&new_features[i]);
        let actual = new_returns[i];

        if (prediction > 0.0 && actual > 0.0) || (prediction < 0.0 && actual < 0.0) {
            correct_direction += 1;
        }
        total_predictions += 1;
    }

    let accuracy = if total_predictions > 0 {
        (correct_direction as f64 / total_predictions as f64) * 100.0
    } else {
        0.0
    };

    println!("\nResults:");
    println!("  Directional accuracy: {:.1}%", accuracy);
    println!("  ({}/{} predictions correct)", correct_direction, total_predictions);
    println!("  Regimes learned: {:?}", trainer.regimes_seen());

    println!("\n=== Example Complete ===");
}
