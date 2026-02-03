//! Example: Running inference with TimeParticle SSM
//!
//! This example demonstrates how to use the TimeParticle model
//! to generate trading signals from market data.

use timeparticle_ssm::data::features::FeatureEngine;
use timeparticle_ssm::data::loader::generate_simulated_data;
use timeparticle_ssm::model::timeparticle::TimeParticleSSM;

fn main() {
    println!("TimeParticle SSM - Inference Example\n");
    println!("=====================================\n");

    // Generate simulated data
    println!("1. Generating simulated market data...");
    let data = generate_simulated_data(200, 42);
    println!("   Generated {} data points", data.len());

    // Compute features
    println!("\n2. Computing multiscale features...");
    let engine = FeatureEngine::new();
    let features = engine.compute_features(&data);
    println!("   Feature matrix shape: {} x {}", features.nrows(), features.ncols());

    // Create model
    println!("\n3. Creating TimeParticle model...");
    let model = TimeParticleSSM::new(
        features.ncols(), // n_features
        64,               // d_hidden
        32,               // d_state
        4,                // n_particles
        3,                // n_layers
    );
    println!("   Model created with:");
    println!("   - {} particles", model.n_particles);
    println!("   - {} layers", model.n_layers);
    println!("   - {} hidden dim", model.d_hidden);

    // Run inference
    println!("\n4. Running inference...");

    // Use last 100 timesteps for prediction
    let start_idx = features.nrows().saturating_sub(100);
    let input = features.slice(ndarray::s![start_idx.., ..]).to_owned();

    let signal = model.predict(&input);
    let probs = model.predict_proba(&input);

    println!("\n   Results:");
    println!("   ---------");
    println!("   Signal: {} (confidence: {:.2}%)",
        signal.signal_type(),
        signal.confidence() * 100.0
    );
    println!("   Probabilities:");
    println!("     - Sell: {:.2}%", probs[0] * 100.0);
    println!("     - Hold: {:.2}%", probs[1] * 100.0);
    println!("     - Buy:  {:.2}%", probs[2] * 100.0);

    // Multiscale analysis
    println!("\n5. Multiscale Analysis:");
    let analysis = model.multiscale_analysis(&input);
    for (scale, momentum, signal) in &analysis {
        println!("   {} scale: {} (momentum: {:.4})",
            scale, signal, momentum);
    }

    // Run batch inference
    println!("\n6. Batch Inference Example:");
    let mut sequences = Vec::new();
    for i in 0..5 {
        let start = i * 20;
        let end = start + 100;
        if end <= features.nrows() {
            sequences.push(features.slice(ndarray::s![start..end, ..]).to_owned());
        }
    }

    let signals = model.predict_batch(&sequences);
    println!("   Generated {} signals:", signals.len());
    for (i, sig) in signals.iter().enumerate() {
        println!("     Sequence {}: {} ({:.2}%)",
            i + 1,
            sig.signal_type(),
            sig.confidence() * 100.0
        );
    }

    println!("\n=====================================");
    println!("Inference example complete!");
}
