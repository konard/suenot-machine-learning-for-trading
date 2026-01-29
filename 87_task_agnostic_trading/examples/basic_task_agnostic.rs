//! Basic Task-Agnostic Trading Example
//!
//! Demonstrates the core task-agnostic model that simultaneously handles
//! trend prediction, volatility forecasting, regime detection, and risk assessment.
//!
//! Run with: cargo run --example basic_task_agnostic

use task_agnostic_trading::model::{
    TaskAgnosticConfig, TaskAgnosticModel,
};
use task_agnostic_trading::strategy::{SignalConfig, SignalGenerator};
use ndarray::Array2;

fn main() {
    println!("=== Task-Agnostic Trading: Basic Example ===\n");

    // Create a task-agnostic model with default configuration
    let config = TaskAgnosticConfig::default()
        .with_input_dim(20)
        .with_repr_dim(16);

    let model = TaskAgnosticModel::new(config);

    println!("Model created with {} task heads:", model.num_tasks());
    for head in &model.heads {
        println!(
            "  - {} (output dim: {}, classification: {})",
            head.config.task_type.name(),
            head.config.output_dim,
            head.config.task_type.is_classification()
        );
    }
    println!();

    // Generate synthetic market features
    let features = generate_market_features(10, 20);
    println!("Input: {} samples x {} features\n", features.nrows(), features.ncols());

    // Run all task heads
    println!("--- Individual Task Predictions ---");
    let task_outputs = model.predict_all_tasks(&features);

    for output in &task_outputs {
        println!("Task: {}", output.task_name);
        let labels = output.task_type.class_labels();
        for (i, row) in output.predictions.rows().into_iter().enumerate().take(3) {
            if output.task_type.is_classification() {
                let best_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let confidence = row[best_idx];
                println!(
                    "  Sample {}: {} (confidence: {:.1}%)",
                    i,
                    labels[best_idx],
                    confidence * 100.0
                );
            } else {
                let value = row[0];
                println!("  Sample {}: {:.4}", i, value);
            }
        }
        if features.nrows() > 3 {
            println!("  ... ({} more samples)", features.nrows() - 3);
        }
        println!();
    }

    // Fused prediction
    println!("--- Fused Trading Decision ---");
    let fused = model.predict(&features);

    for i in 0..features.nrows().min(5) {
        println!(
            "Sample {}: signal={:+.3}, confidence={:.1}%, regime={}, risk={:.2}",
            i,
            fused.signal[i],
            fused.confidence[i] * 100.0,
            fused.regime[i],
            fused.risk_level[i],
        );
    }
    println!();

    // Generate trading signals
    println!("--- Trading Signals ---");
    let signal_gen = SignalGenerator::new(SignalConfig::default());
    let signals = signal_gen.generate(&fused);

    for (i, signal) in signals.iter().enumerate().take(5) {
        println!(
            "Sample {}: {} (position size: {:.2}, raw: {:+.3})",
            i,
            signal.signal_type.label(),
            signal.position_size,
            signal.raw_signal,
        );
    }
    println!();

    // Task contribution analysis
    println!("--- Task Contributions ---");
    for (task_name, weight) in &fused.task_contributions {
        println!("  {}: {:.1}%", task_name, weight * 100.0);
    }

    println!("\n=== Example Complete ===");
}

/// Generate synthetic market features for demonstration
fn generate_market_features(n_samples: usize, n_features: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        // Deterministic pseudo-random features
        let seed = (i * 31 + j * 17) as f64;
        let val = ((seed * 0.1).sin() + 1.0) / 2.0;

        // Add different patterns for different feature groups
        match j {
            0..=4 => val * 0.1 - 0.05,     // Returns-like features
            5..=9 => val * 0.5 + 0.3,       // Momentum features
            10..=14 => val * 0.3 + 0.1,     // Volatility features
            _ => val * 2.0 - 1.0,           // Volume/pattern features
        }
    })
}
