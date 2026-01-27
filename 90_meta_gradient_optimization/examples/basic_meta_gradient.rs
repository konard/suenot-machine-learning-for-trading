//! Basic Meta-Gradient Optimization Example
//!
//! Demonstrates the core meta-gradient optimization algorithm:
//! - Creating a trading model
//! - Setting up the meta-gradient optimizer with learnable learning rates
//! - Training on simulated data
//! - Observing how learning rates adapt

use meta_gradient_trading::prelude::*;
use meta_gradient_trading::data::bybit::SimulatedDataGenerator;
use meta_gradient_trading::data::features::FeatureGenerator;
use meta_gradient_trading::meta_gradient::optimizer::TaskData;

fn main() {
    println!("=== Basic Meta-Gradient Optimization ===\n");

    // Create model and optimizer
    let model = TradingModel::new(11, 32, 1);
    let mut optimizer = MetaGradientOptimizer::new(
        model,
        0.01,   // initial inner learning rate
        0.001,  // meta learning rate
        3,      // inner steps
        false,  // per-parameter LR
        false,  // learn loss
    );

    println!("Model parameters: {}", optimizer.model().num_parameters());
    println!("Initial LR: {:.6}\n", optimizer.get_learning_rates()[0]);

    // Generate simulated data for multiple "assets"
    let feature_gen = FeatureGenerator::new(20);
    let assets = vec![
        ("BTC", SimulatedDataGenerator::generate_klines(200, 50000.0, 0.02)),
        ("ETH", SimulatedDataGenerator::generate_klines(200, 3000.0, 0.025)),
        ("SOL", SimulatedDataGenerator::generate_klines(200, 100.0, 0.03)),
    ];

    // Compute features and returns
    let mut asset_features: Vec<(String, Vec<Vec<f64>>, Vec<f64>)> = Vec::new();
    for (name, klines) in &assets {
        let features = feature_gen.compute_features(klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

        if !features.is_empty() {
            let offset = klines.len() - features.len();
            let labels: Vec<f64> = (0..features.len())
                .filter_map(|i| returns.get(i + offset).copied())
                .collect();
            let min_len = features.len().min(labels.len());
            asset_features.push((
                name.to_string(),
                features[..min_len].to_vec(),
                labels[..min_len].to_vec(),
            ));
        }
    }

    // Meta-training loop
    println!("Starting meta-gradient training...\n");
    let train_size = 20;
    let val_size = 10;
    let num_epochs = 50;

    for epoch in 0..num_epochs {
        let mut tasks = Vec::new();

        for (_, features, labels) in &asset_features {
            if features.len() >= train_size + val_size {
                let start = rand::random::<usize>() % (features.len() - train_size - val_size);

                let task = TaskData::new(
                    features[start..start + train_size].to_vec(),
                    labels[start..start + train_size].to_vec(),
                    features[start + train_size..start + train_size + val_size].to_vec(),
                    labels[start + train_size..start + train_size + val_size].to_vec(),
                );
                tasks.push(task);
            }
        }

        if !tasks.is_empty() {
            let loss = optimizer.meta_train_step(&tasks);
            let lr = optimizer.get_learning_rates()[0];

            if epoch % 5 == 0 {
                println!(
                    "Epoch {:3}: val_loss={:.6}, learned_lr={:.6}",
                    epoch, loss, lr
                );
            }
        }
    }

    // Test adaptation on new data
    println!("\n--- Testing Adaptation ---");
    let new_data = SimulatedDataGenerator::generate_klines(100, 45000.0, 0.018);
    let new_features = feature_gen.compute_features(&new_data);
    let new_closes: Vec<f64> = new_data.iter().map(|k| k.close).collect();
    let new_returns: Vec<f64> = new_closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

    if new_features.len() > 20 {
        let offset = new_data.len() - new_features.len();
        let adapt_features = new_features[..20].to_vec();
        let adapt_labels: Vec<f64> = (0..20)
            .filter_map(|i| new_returns.get(i + offset).copied())
            .collect();

        let adapted = optimizer.adapt(&adapt_features, &adapt_labels, Some(5));

        // Make predictions on remaining data
        let mut correct_direction = 0;
        let mut total = 0;
        for i in 20..new_features.len().min(30) {
            let pred = adapted.predict(&new_features[i]);
            if let Some(&actual) = new_returns.get(i + offset) {
                if (pred > 0.0) == (actual > 0.0) {
                    correct_direction += 1;
                }
                total += 1;
                println!(
                    "  Sample {}: pred={:.6}, actual={:.6}, correct={}",
                    i - 20,
                    pred,
                    actual,
                    (pred > 0.0) == (actual > 0.0)
                );
            }
        }
        if total > 0 {
            println!(
                "\nDirection accuracy: {}/{} ({:.1}%)",
                correct_direction,
                total,
                100.0 * correct_direction as f64 / total as f64
            );
        }
    }

    println!("\nFinal learned LR: {:.6}", optimizer.get_learning_rates()[0]);
    println!("\n=== Complete ===");
}
