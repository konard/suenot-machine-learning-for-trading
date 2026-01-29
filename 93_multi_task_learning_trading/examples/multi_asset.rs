//! Multi-asset MTL training example.
//!
//! Demonstrates training an MTL model across multiple simulated
//! assets with different volatility and trend characteristics.

use mtl_trading::prelude::*;
use mtl_trading::data::bybit::SimulatedDataGenerator;
use mtl_trading::data::features::FeatureGenerator;
use mtl_trading::mtl::trainer::{MTLTrainer, TrainingBatch, TaskTargets};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== Multi-Task Learning: Multi-Asset Training ===\n");

    // Simulate multiple asset types
    let assets = vec![
        ("BTCUSDT", 50000.0, 0.025),   // Bitcoin - high vol
        ("ETHUSDT", 3000.0, 0.030),     // Ethereum - higher vol
        ("SOLUSDT", 100.0, 0.035),      // Solana - highest vol
        ("AAPL", 180.0, 0.015),         // Apple - low vol
        ("MSFT", 400.0, 0.012),         // Microsoft - lowest vol
    ];

    let feature_gen = FeatureGenerator::new(20);
    let mut all_features = Vec::new();
    let mut all_targets = Vec::new();

    for (name, base_price, vol) in &assets {
        let klines = SimulatedDataGenerator::generate_klines(300, *base_price, *vol);
        let features = feature_gen.generate_features(&klines);
        let targets = feature_gen.generate_targets(&klines, 5);

        let min_len = features.len().min(targets.len());
        println!("{}: {} samples (vol={:.3})", name, min_len, vol);

        for i in 0..min_len {
            all_features.push(features[i].features.clone());
            all_targets.push(targets[i].targets.clone());
        }
    }

    println!("\nTotal training samples: {}", all_features.len());

    let batch = TrainingBatch {
        features: all_features,
        targets: all_targets,
    };

    // Create and train model
    let model = MultiTaskModel::new(10, &[128, 64, 32], 16);
    println!("Model parameters: {}", model.num_parameters());

    let mut trainer = MTLTrainer::new(model, 0.005, true);

    println!("\nTraining on multi-asset data...");
    let history = trainer.train(&batch, 30, 64, 10).unwrap();

    if let Some(last) = history.last() {
        println!("\nFinal losses:");
        println!("  Return:     {:.6}", last.return_loss);
        println!("  Volatility: {:.6}", last.volatility_loss);
        println!("  Direction:  {:.6}", last.direction_loss);
        println!("  Volume:     {:.6}", last.volume_loss);
    }

    // Evaluate per-asset
    println!("\nPer-asset predictions (first sample):");
    let model = trainer.model();

    let test_assets = vec![
        ("BTC-like", 50000.0, 0.025),
        ("Low-vol stock", 200.0, 0.010),
    ];

    for (name, base_price, vol) in &test_assets {
        let klines = SimulatedDataGenerator::generate_klines(100, *base_price, *vol);
        let features = feature_gen.generate_features(&klines);
        if let Some(feat) = features.first() {
            let preds = model.forward(&feat.features);
            println!("\n{}:", name);
            println!("  Predicted return:  {:.6}", preds.return_pred[0]);
            println!("  Predicted vol:     {:.6}", preds.volatility_pred[0]);
            println!("  Direction prob:    {:.4}", preds.direction_pred[0]);
        }
    }

    println!("\nDone!");
}
