//! Basic QuantNet example.
//!
//! Demonstrates creating a QuantNet model, pretraining on
//! reconstruction, and finetuning with Sharpe ratio loss.

use quantnet_trading::prelude::*;
use quantnet_trading::data::bybit::SimulatedDataGenerator;
use quantnet_trading::data::features::FeatureGenerator;
use quantnet_trading::quantnet::trainer::{QuantNetTrainer, QuantNetBatch};

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== QuantNet Transfer Trading: Basic Example ===\n");

    // Generate simulated market data
    let klines = SimulatedDataGenerator::generate_klines(500, 50000.0, 0.02);
    println!("Generated {} simulated klines", klines.len());

    // Create features and targets
    let feature_gen = FeatureGenerator::new(20);
    let features = feature_gen.generate_features(&klines);
    let targets = feature_gen.generate_targets(&klines, 5);
    println!("Generated {} feature rows and {} target rows", features.len(), targets.len());

    let min_len = features.len().min(targets.len());
    let batch = QuantNetBatch {
        features: features[..min_len].iter().map(|f| f.features.clone()).collect(),
        targets: targets[..min_len].to_vec(),
        asset_index: 0,
    };

    // Create model and trainer
    let model = QuantNetModel::new(10, 64, 32, &["BTCUSDT"]);
    println!("\nModel parameters: {}", model.num_parameters());
    println!("  Encoder: {}", model.num_encoder_parameters());
    println!("  Decoder: {}", model.num_decoder_parameters());

    let mut trainer = QuantNetTrainer::new(model, 0.005, 1.0);

    // Phase 1: Pretrain
    println!("\n--- Phase 1: Pretraining (reconstruction) ---");
    let pretrain_history = trainer.pretrain(&[batch.clone()], 20, 10).unwrap();
    if let Some(last) = pretrain_history.last() {
        println!("Final reconstruction loss: {:.6}", last.reconstruction_loss);
    }

    // Phase 2: Finetune
    println!("\n--- Phase 2: Finetuning (Sharpe ratio) ---");
    trainer.set_learning_rate(0.001);
    let finetune_history = trainer.finetune(&[batch], 20, 10).unwrap();
    if let Some(last) = finetune_history.last() {
        println!("Final Sharpe loss: {:.6}", last.sharpe_loss);
        println!("Final reconstruction loss: {:.6}", last.reconstruction_loss);
    }

    // Sample prediction
    let model = trainer.model();
    let sample = &features[0].features;
    let preds = model.forward(sample);
    println!("\nSample prediction:");
    for (name, output) in &preds.asset_outputs {
        println!("  {} - signal: {:.4}, return: {:.6}, confidence: {:.4}",
            name, output.signal, output.return_pred, output.confidence);
    }

    println!("\nDone!");
}
