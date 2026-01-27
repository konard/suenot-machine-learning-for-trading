//! Multi-Asset Meta-Gradient Training Example
//!
//! Demonstrates meta-gradient optimization across multiple cryptocurrency
//! assets using simulated Bybit-style data with regime changes.

use meta_gradient_trading::prelude::*;
use meta_gradient_trading::data::bybit::SimulatedDataGenerator;
use meta_gradient_trading::data::features::FeatureGenerator;
use meta_gradient_trading::meta_gradient::optimizer::TaskData;

fn main() {
    println!("=== Multi-Asset Meta-Gradient Training ===\n");

    // Create model with per-parameter learning rates and learned loss
    let model = TradingModel::new(11, 32, 1);
    let num_params = model.num_parameters();
    println!("Model has {} parameters", num_params);

    let mut optimizer = MetaGradientOptimizer::new(
        model,
        0.01,   // initial inner LR
        0.0005, // meta LR
        3,      // inner steps
        false,  // per-parameter LR
        true,   // learn loss function
    );

    println!("Learning loss function: {}", optimizer.is_learning_loss());
    println!("Initial LR: {:.6}\n", optimizer.get_learning_rates()[0]);

    // Generate diverse asset data with different characteristics
    let feature_gen = FeatureGenerator::new(20);
    let assets = vec![
        ("BTCUSDT", SimulatedDataGenerator::generate_regime_changing_klines(300, 50000.0)),
        ("ETHUSDT", SimulatedDataGenerator::generate_regime_changing_klines(300, 3000.0)),
        ("SOLUSDT", SimulatedDataGenerator::generate_klines(300, 100.0, 0.035)),
        ("AVAXUSDT", SimulatedDataGenerator::generate_klines(300, 30.0, 0.04)),
        ("DOTUSDT", SimulatedDataGenerator::generate_klines(300, 7.0, 0.03)),
    ];

    // Prepare features
    let mut asset_data: Vec<(String, Vec<Vec<f64>>, Vec<f64>)> = Vec::new();
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
            println!(
                "  {}: {} feature vectors, price range {:.0}-{:.0}",
                name,
                min_len,
                closes.iter().cloned().fold(f64::INFINITY, f64::min),
                closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );
            asset_data.push((
                name.to_string(),
                features[..min_len].to_vec(),
                labels[..min_len].to_vec(),
            ));
        }
    }

    // Meta-training across multiple assets
    println!("\nStarting cross-asset meta-gradient training...\n");
    let train_size = 25;
    let val_size = 10;
    let num_epochs = 100;
    let mut epoch_losses = Vec::new();

    for epoch in 0..num_epochs {
        let mut tasks = Vec::new();

        // Sample one task from each asset
        for (_, features, labels) in &asset_data {
            if features.len() >= train_size + val_size {
                let max_start = features.len() - train_size - val_size;
                let start = rand::random::<usize>() % (max_start + 1);

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
            epoch_losses.push(loss);

            if epoch % 10 == 0 {
                let lr = optimizer.get_learning_rates()[0];
                let loss_params = optimizer.loss_params();
                println!(
                    "Epoch {:3}: val_loss={:.6}, lr={:.6}, loss_dir_w={:.3}, loss_mag_w={:.3}, loss_asym={:.3}",
                    epoch, loss, lr,
                    loss_params.direction_weight,
                    loss_params.magnitude_weight,
                    loss_params.asymmetry,
                );
            }
        }
    }

    // Summary
    println!("\n--- Training Summary ---");
    if epoch_losses.len() >= 10 {
        let first_10: f64 = epoch_losses[..10].iter().sum::<f64>() / 10.0;
        let last_10: f64 = epoch_losses[epoch_losses.len() - 10..].iter().sum::<f64>() / 10.0;
        println!("Average loss (first 10 epochs): {:.6}", first_10);
        println!("Average loss (last 10 epochs):  {:.6}", last_10);
        println!(
            "Improvement: {:.1}%",
            (1.0 - last_10 / first_10) * 100.0
        );
    }

    println!("\nFinal learned LR: {:.6}", optimizer.get_learning_rates()[0]);
    let final_loss = optimizer.loss_params();
    println!(
        "Final loss params: dir_w={:.3}, mag_w={:.3}, asym={:.3}",
        final_loss.direction_weight, final_loss.magnitude_weight, final_loss.asymmetry
    );

    // Test on each asset individually
    println!("\n--- Per-Asset Adaptation Test ---");
    for (name, features, labels) in &asset_data {
        if features.len() > 30 {
            let adapted = optimizer.adapt(&features[..20], &labels[..20], Some(5));

            let mut correct = 0;
            let total = 10.min(features.len() - 20);
            for i in 20..20 + total {
                let pred = adapted.predict(&features[i]);
                if (pred > 0.0) == (labels[i] > 0.0) {
                    correct += 1;
                }
            }
            println!(
                "  {}: direction accuracy = {}/{} ({:.0}%)",
                name,
                correct,
                total,
                100.0 * correct as f64 / total as f64
            );
        }
    }

    println!("\n=== Complete ===");
}
