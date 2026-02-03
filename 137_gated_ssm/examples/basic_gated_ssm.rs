//! Basic Gated SSM example.
//!
//! Demonstrates creating a Gated SSM model and processing synthetic data.

use gated_ssm::model::gated_ssm::GatedSSMModel;

fn main() {
    println!("=== Gated SSM Basic Example ===\n");

    // Create model: 4 features, 16 state dims, 2 layers
    let mut model = GatedSSMModel::new(4, 16, 2, 2);

    // Generate synthetic sequence (60 time steps, 4 features)
    let sequence: Vec<Vec<f64>> = (0..60)
        .map(|i| {
            let t = i as f64 * 0.1;
            vec![
                t.sin() * 0.02,   // Simulated log return
                0.01,              // Simulated volatility
                0.5 + t.cos() * 0.1, // Simulated RSI
                -0.005,            // Simulated volume change
            ]
        })
        .collect();

    // Get prediction
    let prediction = model.predict(&sequence);
    println!("Prediction (probability of up): {:.4}", prediction);
    println!(
        "Signal: {}",
        if prediction > 0.55 {
            "LONG"
        } else if prediction < 0.45 {
            "SHORT"
        } else {
            "NEUTRAL"
        }
    );

    // Process multiple sequences
    println!("\n--- Processing multiple sequences ---");
    for trial in 0..5 {
        let offset = trial as f64 * 0.5;
        let seq: Vec<Vec<f64>> = (0..60)
            .map(|i| {
                let t = i as f64 * 0.1 + offset;
                vec![t.sin() * 0.02, 0.015, 0.5, t.cos() * 0.01]
            })
            .collect();

        model.reset();
        let pred = model.predict(&seq);
        println!("Trial {}: prediction = {:.4}", trial + 1, pred);
    }

    println!("\nDone!");
}
