//! # Train Portfolio ODE Example
//!
//! Demonstrates how to train a Neural ODE model for portfolio optimization.
//!
//! Run with:
//! ```bash
//! cargo run --example train_portfolio_ode
//! ```

use anyhow::Result;
use neural_ode_crypto::data::Features;
use neural_ode_crypto::model::{
    NeuralODEPortfolio,
    Trainer,
    TrainingConfig,
    LossFunction,
    TrainingSample,
};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== Neural ODE Portfolio Training ===");

    // Configuration
    let n_assets = 3;
    let n_features = 12;
    let hidden_dim = 16;

    // Create model
    info!("Creating model with {} assets, {} features, {} hidden dim",
        n_assets, n_features, hidden_dim);

    let mut model = NeuralODEPortfolio::new(n_assets, n_features, hidden_dim);
    info!("Model has {} parameters", model.num_params());

    // Create synthetic training data
    info!("Generating synthetic training data...");
    let samples = generate_training_samples(n_assets, n_features, 100);
    info!("Generated {} training samples", samples.len());

    // Configure training
    let config = TrainingConfig {
        learning_rate: 1e-3,
        epochs: 30,
        batch_size: 32,
        loss_fn: LossFunction::Portfolio {
            risk_aversion: 1.0,
            cost_weight: 0.01,
        },
        grad_clip: 1.0,
        weight_decay: 1e-5,
        patience: 10,
    };

    // Create trainer
    let mut trainer = Trainer::new(config);

    // Train model
    info!("Starting training...");
    let loss_history = trainer.train_evolution(&mut model, &samples);

    info!("\nTraining complete!");
    info!("Final loss: {:.6}", loss_history.last().unwrap_or(&0.0));

    // Test the trained model
    info!("\n=== Testing Trained Model ===");
    test_model(&model, n_assets, n_features);

    Ok(())
}

fn generate_training_samples(
    n_assets: usize,
    n_features: usize,
    n_samples: usize,
) -> Vec<TrainingSample> {
    let mut samples = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Generate random initial weights
        let mut initial_weights: Vec<f64> = (0..n_assets)
            .map(|_| rand::random::<f64>())
            .collect();
        let sum: f64 = initial_weights.iter().sum();
        for w in &mut initial_weights {
            *w /= sum;
        }

        // Generate random features
        let data: Vec<Vec<f64>> = (0..n_assets)
            .map(|_| {
                (0..n_features)
                    .map(|_| rand::random::<f64>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();

        let features = Features {
            n_assets,
            n_features,
            data,
            names: (0..n_features).map(|i| format!("feature_{}", i)).collect(),
        };

        // Generate target weights (slightly shifted towards momentum)
        let momentum_factor = 0.1;
        let mut target_weights = initial_weights.clone();
        for j in 0..n_assets {
            // Shift based on "momentum" feature (first feature)
            let momentum = features.data[j][0];
            target_weights[j] += momentum * momentum_factor;
        }
        // Normalize
        let sum: f64 = target_weights.iter().sum();
        for w in &mut target_weights {
            *w = (*w).max(0.0) / sum;
        }

        // Generate simulated returns
        let returns: Vec<f64> = (0..n_assets)
            .map(|j| {
                // Return correlated with momentum
                let base = features.data[j][0] * 0.01;
                base + (rand::random::<f64>() - 0.5) * 0.02
            })
            .collect();

        samples.push(TrainingSample {
            initial_weights,
            features,
            target_weights,
            returns: Some(returns),
        });
    }

    samples
}

fn test_model(model: &NeuralODEPortfolio, n_assets: usize, n_features: usize) {
    // Create test input
    let initial_weights = vec![1.0 / n_assets as f64; n_assets];

    let features = Features {
        n_assets,
        n_features,
        data: (0..n_assets)
            .map(|i| vec![0.1 * (i + 1) as f64; n_features])
            .collect(),
        names: (0..n_features).map(|i| format!("f{}", i)).collect(),
    };

    info!("Initial weights: {:?}", initial_weights);

    // Get trajectory
    let trajectory = model.solve_trajectory(
        &initial_weights,
        &features,
        (0.0, 1.0),
        11,
    );

    info!("\nTrajectory:");
    for (t, weights) in &trajectory {
        let weights_str: Vec<String> = weights.iter()
            .map(|w| format!("{:.3}", w))
            .collect();
        info!("  t={:.1}: [{}]", t, weights_str.join(", "));
    }

    // Show target weights
    let target = model.get_target_weights(&initial_weights, &features, 1.0);
    info!("\nTarget weights at t=1.0: {:?}",
        target.iter().map(|w| format!("{:.3}", w)).collect::<Vec<_>>()
    );

    // Calculate turnover
    let turnover: f64 = initial_weights.iter()
        .zip(target.iter())
        .map(|(i, t)| (i - t).abs())
        .sum();
    info!("Total turnover: {:.2}%", turnover * 100.0);
}
