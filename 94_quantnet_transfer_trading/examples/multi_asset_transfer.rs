//! Multi-asset transfer learning example.
//!
//! Demonstrates pretraining a QuantNet model across multiple assets
//! and then finetuning on individual assets via transfer learning.

use quantnet_trading::prelude::*;
use quantnet_trading::data::bybit::SimulatedDataGenerator;
use quantnet_trading::data::features::FeatureGenerator;
use quantnet_trading::quantnet::trainer::{QuantNetTrainer, QuantNetBatch};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== QuantNet Transfer Trading: Multi-Asset Transfer ===\n");

    // Define assets
    let assets = vec![
        ("BTCUSDT", 50000.0, 0.025),
        ("ETHUSDT", 3000.0, 0.030),
        ("SOLUSDT", 100.0, 0.035),
    ];

    let feature_gen = FeatureGenerator::new(20);

    // Build pretrain batches (one per asset)
    let mut pretrain_batches = Vec::new();
    for (i, (name, base_price, vol)) in assets.iter().enumerate() {
        let klines = SimulatedDataGenerator::generate_klines(400, *base_price, *vol);
        let features = feature_gen.generate_features(&klines);
        let targets = feature_gen.generate_targets(&klines, 5);
        let min_len = features.len().min(targets.len());

        println!("{}: {} samples (vol={:.3})", name, min_len, vol);

        pretrain_batches.push(QuantNetBatch {
            features: features[..min_len].iter().map(|f| f.features.clone()).collect(),
            targets: targets[..min_len].to_vec(),
            asset_index: i,
        });
    }

    // Create model with all asset heads
    let asset_names: Vec<&str> = assets.iter().map(|(n, _, _)| *n).collect();
    let model = QuantNetModel::new(10, 64, 32, &asset_names);
    println!("\nModel parameters: {}", model.num_parameters());

    let mut trainer = QuantNetTrainer::new(model, 0.005, 1.0);

    // Phase 1: Pretrain on all assets
    println!("\n--- Phase 1: Pretraining on all assets ---");
    let pretrain_history = trainer.pretrain(&pretrain_batches, 15, 5).unwrap();
    if let Some(last) = pretrain_history.last() {
        println!("Final pretrain reconstruction loss: {:.6}", last.reconstruction_loss);
    }

    // Phase 2: Finetune per asset
    println!("\n--- Phase 2: Finetuning per asset ---");
    trainer.set_learning_rate(0.001);
    for (i, (name, _, _)) in assets.iter().enumerate() {
        println!("\nFinetuning {} (head {})...", name, i);
        let history = trainer.finetune(&[pretrain_batches[i].clone()], 10, 5).unwrap();
        if let Some(last) = history.last() {
            println!("  Sharpe loss: {:.6}, Recon loss: {:.6}", last.sharpe_loss, last.reconstruction_loss);
        }
    }

    // Evaluate transfer: test on unseen asset characteristics
    println!("\n--- Transfer Evaluation ---");
    let model = trainer.model();
    let test_assets = vec![
        ("BTC-like", 50000.0, 0.025),
        ("High-vol token", 50.0, 0.045),
    ];

    for (name, base_price, vol) in &test_assets {
        let klines = SimulatedDataGenerator::generate_klines(100, *base_price, *vol);
        let features = feature_gen.generate_features(&klines);
        if let Some(feat) = features.first() {
            let preds = model.forward(&feat.features);
            println!("\n{}:", name);
            for (asset, output) in &preds.asset_outputs {
                println!("  {} -> signal: {:.4}, confidence: {:.4}",
                    asset, output.signal, output.confidence);
            }
        }
    }

    println!("\nDone!");
}
