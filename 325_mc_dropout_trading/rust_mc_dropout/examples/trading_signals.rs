//! Example: Trading Signals with Uncertainty
//!
//! This example demonstrates how to generate trading signals
//! using MC Dropout uncertainty estimation.

use mc_dropout_trading::prelude::*;
use ndarray::Array1;
use rand::Rng;

fn main() {
    println!("=== Trading Signals with Uncertainty ===\n");

    // Create model
    let model_config = MCDropoutConfig::new(20)
        .with_hidden_dims(vec![64, 32])
        .with_dropout_rate(0.15)
        .with_mc_samples(50);

    let model = MCDropoutModel::new(model_config);

    // Create signal generator with different configurations
    println!("=== Testing Different Strategy Configurations ===\n");

    let configs = vec![
        ("Conservative", SignalConfig::conservative()),
        ("Default", SignalConfig::default()),
        ("Aggressive", SignalConfig::aggressive()),
    ];

    let mut rng = rand::thread_rng();

    // Generate sample inputs
    let inputs: Vec<Array1<f64>> = (0..10)
        .map(|i| {
            let scale = 0.5 + (i as f64) * 0.1;
            Array1::from_vec(
                (0..20).map(|_| rng.gen_range(-1.0..1.0) * scale).collect()
            )
        })
        .collect();

    for (name, config) in &configs {
        println!("--- {} Strategy ---", name);
        println!("  Confidence threshold: {:.2}", config.confidence_threshold);
        println!("  Uncertainty threshold: {:.4}", config.uncertainty_threshold);
        println!("  Max position size: {:.2}", config.max_position_size);
        println!();

        let generator = SignalGenerator::new(config.clone());

        let mut long_count = 0;
        let mut short_count = 0;
        let mut hold_count = 0;

        for (i, input) in inputs.iter().enumerate() {
            let prediction = model.predict_with_uncertainty(input, None);
            let signal = generator.generate(&prediction);

            match signal.signal_type {
                SignalType::Long => long_count += 1,
                SignalType::Short => short_count += 1,
                SignalType::Hold => hold_count += 1,
            }

            println!(
                "  Sample {:2}: {} | Pred: {:+.4} | Unc: {:.4} | Conf: {:.2} | Size: {:.4}",
                i + 1,
                signal.signal_type,
                signal.predicted_return,
                signal.uncertainty,
                signal.confidence,
                signal.position_size
            );

            if !signal.reason.is_empty() && signal.signal_type == SignalType::Hold {
                println!("             Reason: {}", signal.reason);
            }
        }

        println!("\n  Summary: LONG={}, SHORT={}, HOLD={}\n", long_count, short_count, hold_count);
    }

    // Demonstrate uncertainty-based position sizing
    println!("=== Position Sizing Demo ===\n");

    let generator = SignalGenerator::new(SignalConfig::default());

    println!("Showing how position size varies with uncertainty:\n");
    println!("{:<15} {:<15} {:<15} {:<15}", "Uncertainty", "Pred Return", "Confidence", "Position Size");
    println!("{}", "-".repeat(60));

    // Create predictions with varying uncertainty
    for uncertainty in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03] {
        let prediction = PredictionResult {
            mean: 0.02,  // Fixed positive prediction
            std: uncertainty,
            variance: uncertainty * uncertainty,
            samples: vec![],
            n_samples: 50,
        };

        let signal = generator.generate(&prediction);

        println!(
            "{:<15.4} {:<15.4} {:<15.2} {:<15.4}",
            uncertainty,
            signal.predicted_return,
            signal.confidence,
            signal.position_size
        );
    }

    println!("\n=== Trading signals example complete! ===");
}
