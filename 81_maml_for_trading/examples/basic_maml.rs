//! Basic MAML example demonstrating core meta-learning concepts.
//!
//! This example shows:
//! - Creating a MAML trainer
//! - Generating synthetic trading tasks
//! - Meta-training on multiple tasks
//! - Adapting to a new task with few examples

use maml_trading::{
    data::bybit::SimulatedDataGenerator,
    data::features::FeatureGenerator,
    maml::algorithm::{MAMLTrainer, TaskData},
    model::network::TradingModel,
};

fn main() {
    println!("=== Basic MAML for Trading Example ===\n");

    // Step 1: Create the model
    println!("Step 1: Creating trading model...");
    let input_size = 11; // Number of technical features
    let hidden_size = 32;
    let output_size = 1; // Predict returns

    let model = TradingModel::new(input_size, hidden_size, output_size);
    println!("  Model created with {} parameters\n", model.num_parameters());

    // Step 2: Create MAML trainer
    println!("Step 2: Setting up MAML trainer...");
    let inner_lr = 0.01; // Learning rate for task adaptation
    let outer_lr = 0.001; // Learning rate for meta-update
    let inner_steps = 5; // Gradient steps during adaptation
    let first_order = true; // Use FOMAML for efficiency

    let mut trainer = MAMLTrainer::new(model, inner_lr, outer_lr, inner_steps, first_order);
    println!("  Inner learning rate: {}", inner_lr);
    println!("  Outer learning rate: {}", outer_lr);
    println!("  Inner steps: {}", inner_steps);
    println!("  Using FOMAML: {}\n", first_order);

    // Step 3: Generate synthetic tasks (simulating different assets)
    println!("Step 3: Generating synthetic trading tasks...");
    let feature_gen = FeatureGenerator::default_window();
    let num_tasks = 5;
    let mut tasks = Vec::new();

    for i in 0..num_tasks {
        // Generate different market conditions for each task
        let volatility = 0.01 + (i as f64 * 0.005); // Increasing volatility
        let klines = SimulatedDataGenerator::generate_klines(200, 50000.0, volatility);
        let features = feature_gen.compute_features(&klines);

        // Create returns (labels)
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

        // Split into support and query sets
        let split_idx = features.len() / 2;
        let support_features = features[..split_idx].to_vec();
        let support_labels = returns[..split_idx].to_vec();
        let query_features = features[split_idx..].to_vec();
        let query_labels = returns[split_idx..returns.len().min(split_idx + query_features.len())]
            .to_vec();

        if support_labels.len() == support_features.len()
            && query_labels.len() == query_features.len()
        {
            tasks.push(TaskData {
                support_features,
                support_labels,
                query_features,
                query_labels,
            });
            println!(
                "  Task {} created (volatility: {:.3})",
                i + 1,
                volatility
            );
        }
    }
    println!();

    // Step 4: Meta-training
    println!("Step 4: Meta-training on {} tasks...", tasks.len());
    let num_epochs = 10;

    for epoch in 0..num_epochs {
        let meta_loss = trainer.meta_train_step(&tasks);
        if (epoch + 1) % 2 == 0 {
            println!("  Epoch {}: Meta-loss = {:.6}", epoch + 1, meta_loss);
        }
    }
    println!();

    // Step 5: Adapt to a new task
    println!("Step 5: Adapting to a new (unseen) task...");
    let new_klines = SimulatedDataGenerator::generate_klines(100, 45000.0, 0.025);
    let new_features = feature_gen.compute_features(&new_klines);
    let new_closes: Vec<f64> = new_klines.iter().map(|k| k.close).collect();
    let new_returns: Vec<f64> = new_closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

    // Use only first 10 examples for adaptation (few-shot learning)
    let few_shot_features = new_features[..10].to_vec();
    let few_shot_labels = new_returns[..10].to_vec();

    println!("  Adapting with only {} examples...", few_shot_features.len());
    let adapted_model = trainer.adapt(&few_shot_features, &few_shot_labels, Some(5));

    // Step 6: Make predictions with adapted model
    println!("\nStep 6: Making predictions with adapted model...");
    println!("  Testing on {} new data points\n", new_features.len() - 10);

    let mut correct_direction = 0;
    let mut total_predictions = 0;

    for i in 10..new_features.len().min(new_returns.len()) {
        let prediction = adapted_model.predict(&new_features[i]);
        let actual = new_returns[i];

        // Check if direction is correct
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

    println!("Results:");
    println!("  Directional accuracy: {:.1}%", accuracy);
    println!("  ({}/{} predictions correct)", correct_direction, total_predictions);

    // Compare with random baseline
    println!("\n  Note: Random baseline would be ~50%");
    if accuracy > 50.0 {
        println!("  MAML shows improvement over random!");
    }

    println!("\n=== Example Complete ===");
}
