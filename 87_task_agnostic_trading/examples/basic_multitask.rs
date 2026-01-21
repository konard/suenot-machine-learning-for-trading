//! Basic Multi-Task Learning Example
//!
//! This example demonstrates the core task-agnostic learning functionality
//! with synthetic market data.
//!
//! Run with: cargo run --example basic_multitask

use task_agnostic_trading::encoder::{EncoderConfig, EncoderType};
use task_agnostic_trading::training::{TrainerConfig, MultiTaskTrainer, HarmonizerType, WeightingStrategy};
use task_agnostic_trading::tasks::TaskType;
use task_agnostic_trading::fusion::{DecisionFusion, FusionConfig, FusionMethod};
use ndarray::Array2;

fn main() {
    println!("=== Task-Agnostic Trading: Basic Multi-Task Example ===\n");

    // Generate synthetic market features
    let (features, _targets) = generate_synthetic_data();

    println!("Generated {} samples with {} features", features.nrows(), features.ncols());
    println!("Tasks: Direction, Volatility, Regime, Returns\n");

    // Configure multi-task model
    let config = TrainerConfig::default()
        .with_encoder(
            EncoderConfig::default()
                .with_encoder_type(EncoderType::Transformer)
                .with_input_dim(features.ncols())
                .with_embedding_dim(32)
        )
        .with_harmonizer(HarmonizerType::PCGrad)
        .with_weighting(WeightingStrategy::Uncertainty)
        .with_learning_rate(0.001);

    let trainer = MultiTaskTrainer::new(config);

    println!("--- Model Configuration ---");
    println!("Encoder: Transformer");
    println!("Embedding dim: 32");
    println!("Gradient harmonization: PCGrad");
    println!("Task weighting: Uncertainty-based\n");

    // Make predictions on test samples
    println!("--- Predictions ---\n");

    // Create decision fusion
    let fusion = DecisionFusion::new(FusionConfig {
        method: FusionMethod::WeightedConfidence,
        min_confidence: 0.5,
        ..Default::default()
    });

    for i in 0..5.min(features.nrows()) {
        let sample = features.row(i).to_owned();
        let prediction = trainer.predict(&sample);

        println!("Sample {}:", i + 1);

        if let Some(ref dir) = prediction.direction {
            println!("  Direction: {} (confidence: {:.1}%)",
                dir.direction, dir.confidence * 100.0);
        }

        if let Some(ref vol) = prediction.volatility {
            println!("  Volatility: {:.2}% - {} (confidence: {:.1}%)",
                vol.volatility_pct, vol.level, vol.confidence * 100.0);
        }

        if let Some(ref regime) = prediction.regime {
            println!("  Regime: {} (risk level: {}, confidence: {:.1}%)",
                regime.regime, regime.risk_level, regime.confidence * 100.0);
        }

        if let Some(ref ret) = prediction.returns {
            println!("  Expected return: {:.2}% [{:.2}%, {:.2}%]",
                ret.return_pct, ret.lower_bound, ret.upper_bound);
        }

        // Fuse predictions into trading decision
        let decision = fusion.fuse(&prediction);
        println!("  → Trading decision: {} (position size: {:.1}%)",
            decision.decision, decision.position_size * 100.0);
        println!("  → Confidence: {:.1}% (agreement: {:.1}%)",
            decision.confidence.overall * 100.0,
            decision.confidence.task_agreement * 100.0);

        println!();
    }

    // Test different encoder types
    println!("--- Encoder Comparison ---\n");

    let encoder_types = [
        (EncoderType::Transformer, "Transformer"),
        (EncoderType::CNN, "CNN"),
        (EncoderType::MoE, "Mixture of Experts"),
    ];

    for (encoder_type, name) in encoder_types {
        let config = TrainerConfig::default()
            .with_encoder(
                EncoderConfig::default()
                    .with_encoder_type(encoder_type)
                    .with_input_dim(features.ncols())
                    .with_embedding_dim(32)
            );

        let trainer = MultiTaskTrainer::new(config);
        let sample = features.row(0).to_owned();
        let prediction = trainer.predict(&sample);

        let avg_conf = prediction.average_confidence();
        println!("{}: Average confidence = {:.1}%", name, avg_conf * 100.0);
    }

    // Test different fusion methods
    println!("\n--- Fusion Method Comparison ---\n");

    let config = TrainerConfig::default()
        .with_encoder(EncoderConfig::default().with_input_dim(features.ncols()));
    let trainer = MultiTaskTrainer::new(config);
    let sample = features.row(0).to_owned();
    let prediction = trainer.predict(&sample);

    let fusion_methods = [
        (FusionMethod::Voting, "Voting"),
        (FusionMethod::WeightedConfidence, "Weighted Confidence"),
        (FusionMethod::RuleBased, "Rule-Based"),
    ];

    for (method, name) in fusion_methods {
        let fusion = DecisionFusion::new(FusionConfig {
            method,
            ..Default::default()
        });
        let result = fusion.fuse(&prediction);

        println!("{}: {} (size: {:.1}%)",
            name, result.decision, result.position_size * 100.0);
    }

    println!("\n=== Example Complete ===");
}

/// Generate synthetic market data for demonstration
fn generate_synthetic_data() -> (Array2<f64>, std::collections::HashMap<TaskType, Array2<f64>>) {
    use std::collections::HashMap;

    let n_samples = 100;
    let n_features = 20;

    // Generate random features (simulating normalized market indicators)
    let mut feature_data = Vec::with_capacity(n_samples * n_features);
    let mut rng_state: u64 = 42;

    for _ in 0..n_samples {
        for _ in 0..n_features {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random = ((rng_state >> 16) & 0x7FFF) as f64 / 32768.0 * 2.0 - 1.0;
            feature_data.push(random);
        }
    }

    let features = Array2::from_shape_vec((n_samples, n_features), feature_data)
        .expect("Failed to create feature array");

    // Generate target labels (synthetic)
    let mut targets = HashMap::new();

    // Direction targets (one-hot: [up, down, sideways])
    let mut dir_targets = Vec::new();
    for i in 0..n_samples {
        let class = i % 3;
        let one_hot = match class {
            0 => vec![1.0, 0.0, 0.0], // Up
            1 => vec![0.0, 1.0, 0.0], // Down
            _ => vec![0.0, 0.0, 1.0], // Sideways
        };
        dir_targets.extend(one_hot);
    }
    targets.insert(TaskType::Direction,
        Array2::from_shape_vec((n_samples, 3), dir_targets).unwrap());

    // Volatility targets (mean)
    let vol_targets: Vec<f64> = (0..n_samples)
        .map(|i| 0.01 + 0.02 * (i as f64 / n_samples as f64))
        .collect();
    targets.insert(TaskType::Volatility,
        Array2::from_shape_vec((n_samples, 1), vol_targets).unwrap());

    // Regime targets (one-hot: 5 classes)
    let mut regime_targets = Vec::new();
    for i in 0..n_samples {
        let class = i % 5;
        let mut one_hot = vec![0.0; 5];
        one_hot[class] = 1.0;
        regime_targets.extend(one_hot);
    }
    targets.insert(TaskType::Regime,
        Array2::from_shape_vec((n_samples, 5), regime_targets).unwrap());

    // Returns targets (mean)
    let ret_targets: Vec<f64> = (0..n_samples)
        .map(|i| -0.02 + 0.04 * (i as f64 / n_samples as f64))
        .collect();
    targets.insert(TaskType::Returns,
        Array2::from_shape_vec((n_samples, 1), ret_targets).unwrap());

    (features, targets)
}
