//! Example: Generate live trading signals
//!
//! This example demonstrates how to generate trading signals
//! using the VAE model with uncertainty estimation.

use variational_inference_trading::prelude::*;
use ndarray::Array1;
use rand::Rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Live Signal Generation Demo ===\n");

    // Create VAE model
    let config = VAEConfig::new(13, 8, vec![64, 32])
        .with_beta(2.0)
        .with_sequence_length(60);

    let vae = VAE::new(config.clone());

    // Create signal generator
    let signal_gen = SignalGenerator::new(0.6)
        .with_uncertainty_threshold(0.05)
        .with_position_limits(0.01, 0.1);

    println!("Signal Generator Configuration:");
    println!("  Confidence threshold: 0.6");
    println!("  Uncertainty threshold: 0.05");
    println!("  Position limits: 1% - 10%");

    // Simulate live data
    println!("\nSimulating live market data stream...\n");

    let mut rng = rand::thread_rng();
    let input_size = config.input_dim * config.sequence_length;

    println!("{:<8} {:>8} {:>10} {:>10} {:>8} {:>8} {:>10}",
             "Time", "Signal", "E[return]", "Uncert", "P(r>0)", "Regime", "PosSiz");
    println!("{:-<75}", "");

    for i in 0..20 {
        // Generate synthetic features
        let features = Array1::from_vec(
            (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect()
        );

        // Get prediction
        let prediction = vae.predict(&features, 50);

        // Generate signal
        let signal = signal_gen.generate(&prediction);

        // Display
        let time = format!("T+{}", i);
        let signal_str = match signal.signal_type {
            SignalType::Long => "LONG",
            SignalType::Short => "SHORT",
            SignalType::Hold => "HOLD",
        };

        println!("{:<8} {:>8} {:>10.4} {:>10.4} {:>7.1}% {:>8} {:>9.2}%",
                 time,
                 signal_str,
                 signal.expected_return,
                 signal.uncertainty,
                 signal.prob_positive * 100.0,
                 prediction.regime_name(),
                 signal.position_size * 100.0);
    }

    // Summary statistics
    println!("\n--- Signal Statistics ---");

    let mut long_count = 0;
    let mut short_count = 0;
    let mut hold_count = 0;

    for _ in 0..100 {
        let features = Array1::from_vec(
            (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect()
        );
        let prediction = vae.predict(&features, 50);
        let signal = signal_gen.generate(&prediction);

        match signal.signal_type {
            SignalType::Long => long_count += 1,
            SignalType::Short => short_count += 1,
            SignalType::Hold => hold_count += 1,
        }
    }

    println!("Out of 100 simulated signals:");
    println!("  LONG:  {} ({:.0}%)", long_count, long_count as f64);
    println!("  SHORT: {} ({:.0}%)", short_count, short_count as f64);
    println!("  HOLD:  {} ({:.0}%)", hold_count, hold_count as f64);

    println!("\n=== Signal Generation Demo Complete ===");

    Ok(())
}
