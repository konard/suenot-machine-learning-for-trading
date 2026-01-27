//! Basic Multi-Task Learning example.
//!
//! Demonstrates creating a multi-task model, training it on
//! simulated data, and evaluating predictions.

use mtl_trading::prelude::*;
use mtl_trading::data::bybit::SimulatedDataGenerator;
use mtl_trading::data::features::FeatureGenerator;
use mtl_trading::mtl::trainer::{MTLTrainer, TrainingBatch, TaskTargets};

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== Multi-Task Learning: Basic Example ===\n");

    // Generate simulated market data
    let klines = SimulatedDataGenerator::generate_klines(500, 50000.0, 0.02);
    println!("Generated {} simulated klines", klines.len());

    // Create features and targets
    let feature_gen = FeatureGenerator::new(20);
    let features = feature_gen.generate_features(&klines);
    let targets = feature_gen.generate_targets(&klines, 5);
    println!("Generated {} feature rows and {} target rows", features.len(), targets.len());

    // Align features and targets by timestamp
    let min_len = features.len().min(targets.len());
    let feature_vecs: Vec<Vec<f64>> = features[..min_len].iter().map(|f| f.features.clone()).collect();
    let target_vecs: Vec<TaskTargets> = targets[..min_len].iter().map(|t| t.targets.clone()).collect();

    let batch = TrainingBatch {
        features: feature_vecs,
        targets: target_vecs,
    };

    // Create model and trainer
    let model = MultiTaskModel::new(10, &[64, 32], 16);
    println!("\nModel parameters: {}", model.num_parameters());

    let mut trainer = MTLTrainer::new(model, 0.005, true);

    // Train
    println!("\nTraining...");
    let history = trainer.train(&batch, 50, 32, 10).unwrap();

    // Report final losses
    if let Some(last) = history.last() {
        println!("\nFinal losses:");
        println!("  Return:     {:.6}", last.return_loss);
        println!("  Volatility: {:.6}", last.volatility_loss);
        println!("  Direction:  {:.6}", last.direction_loss);
        println!("  Volume:     {:.6}", last.volume_loss);
        println!("  Total:      {:.6}", last.total_loss);
    }

    // Task weights
    let weights = trainer.task_weights();
    println!("\nLearned task weights:");
    let task_names = ["Return", "Volatility", "Direction", "Volume"];
    for (name, weight) in task_names.iter().zip(weights.iter()) {
        println!("  {}: {:.4}", name, weight);
    }

    // Sample predictions
    let model = trainer.model();
    let sample = &batch.features[0];
    let preds = model.forward(sample);
    println!("\nSample prediction:");
    println!("  Return:     {:.6}", preds.return_pred[0]);
    println!("  Volatility: {:.6}", preds.volatility_pred[0]);
    println!("  Direction:  {:.4}", preds.direction_pred[0]);
    println!("  Volume:     {:.4}", preds.volume_pred[0]);

    println!("\nDone!");
}
