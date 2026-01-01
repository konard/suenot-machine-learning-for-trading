//! Basic Squeeze-and-Excitation Block Demonstration
//!
//! This example shows how to use the SE block for feature attention
//! in a trading context.

use ndarray::Array2;
use se_trading::prelude::*;
use se_trading::models::se_block::SqueezeType;
use se_trading::data::features::FEATURE_NAMES;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Squeeze-and-Excitation Block Demo                        ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Create sample feature data
    // Simulating 100 time steps with 12 features (indicators)
    let num_timesteps = 100;
    let num_features = 12;

    println!("Creating sample trading features...");
    println!("  - Time steps: {}", num_timesteps);
    println!("  - Features: {}", num_features);
    println!();

    // Generate synthetic feature data
    let features = generate_synthetic_features(num_timesteps, num_features);

    // Create SE block with reduction ratio of 4
    println!("Creating SE Block with reduction ratio = 4...\n");
    let se_block = SEBlock::new(num_features, 4);

    // Forward pass
    let output = se_block.forward(&features);

    // Get attention weights
    let attention = se_block.get_attention_weights(&features);

    // Display results
    println!("═══════════════════════════════════════════════════════════");
    println!("                    ATTENTION WEIGHTS                       ");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create sorted list of features by attention weight
    let mut feature_attention: Vec<(&str, f64)> = FEATURE_NAMES
        .iter()
        .zip(attention.iter())
        .map(|(&name, &weight)| (name, weight))
        .collect();

    feature_attention.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Feature importance (sorted by attention weight):\n");
    for (i, (name, weight)) in feature_attention.iter().enumerate() {
        let bar_len = (weight * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!(
            "  {:2}. {:<15} {:5.3} │{}│",
            i + 1,
            name,
            weight,
            bar
        );
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("                    OUTPUT ANALYSIS                         ");
    println!("═══════════════════════════════════════════════════════════\n");

    // Compare input and output statistics
    println!("Input vs Output statistics (last time step):\n");
    let input_last = features.row(num_timesteps - 1);
    let output_last = output.row(num_timesteps - 1);

    println!("  {:15} {:>10} {:>10} {:>10}", "Feature", "Input", "Output", "Attention");
    println!("  {}", "-".repeat(50));

    for (i, &name) in FEATURE_NAMES.iter().enumerate() {
        if i < num_features {
            println!(
                "  {:15} {:>10.4} {:>10.4} {:>10.4}",
                name,
                input_last[i],
                output_last[i],
                attention[i]
            );
        }
    }

    // Demonstrate different squeeze types
    println!("\n═══════════════════════════════════════════════════════════");
    println!("                 SQUEEZE TYPE COMPARISON                    ");
    println!("═══════════════════════════════════════════════════════════\n");

    let squeeze_types = [
        ("GlobalAveragePooling", SqueezeType::GlobalAveragePooling),
        ("GlobalMaxPooling", SqueezeType::GlobalMaxPooling),
        ("LastValue", SqueezeType::LastValue),
        ("ExponentialWeighted", SqueezeType::ExponentialWeighted { alpha: 0.1 }),
    ];

    for (name, squeeze_type) in squeeze_types {
        let se = SEBlock::with_squeeze_type(num_features, 4, squeeze_type);
        let weights = se.get_attention_weights(&features);

        // Get top 3 features for this squeeze type
        let mut indexed: Vec<(usize, f64)> = weights
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        print!("  {:25} Top 3: ", name);
        for (i, (idx, w)) in indexed.iter().take(3).enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}({:.3})", FEATURE_NAMES.get(*idx).unwrap_or(&"?"), w);
        }
        println!();
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("                    INTERPRETATION                          ");
    println!("═══════════════════════════════════════════════════════════\n");

    // Interpret the results
    let top_feature = feature_attention.first().unwrap();
    let bottom_feature = feature_attention.last().unwrap();

    println!("The SE block has determined that:");
    println!(
        "  - Most important feature: {} (weight: {:.3})",
        top_feature.0, top_feature.1
    );
    println!(
        "  - Least important feature: {} (weight: {:.3})",
        bottom_feature.0, bottom_feature.1
    );
    println!();
    println!("This means the model is paying more attention to {}",
             top_feature.0);
    println!("and less attention to {} for the current market conditions.",
             bottom_feature.0);

    println!("\n✓ SE Block demonstration complete!");
}

/// Generate synthetic feature data for demonstration
fn generate_synthetic_features(timesteps: usize, features: usize) -> Array2<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    Array2::from_shape_fn((timesteps, features), |(i, j)| {
        // Create different patterns for different features
        let base = match j {
            0 => (i as f64 * 0.01).sin() * 0.5,      // returns - oscillating
            1 => (i as f64 * 0.01).cos() * 0.3,      // log_returns
            2 => 0.02 + rng.gen_range(-0.01..0.01),  // volatility - stable
            3 => ((i as f64 * 0.05).sin() + 1.0) / 2.0 - 0.5, // RSI - oscillating
            4 => i as f64 * 0.001 - 0.05,            // MACD - trending
            5 => i as f64 * 0.0005,                  // MACD signal
            6 => rng.gen_range(-0.02..0.02),         // MACD histogram
            7 => 0.015 + rng.gen_range(-0.005..0.005), // ATR
            8 => (i as f64 * 0.02).sin(),            // Bollinger %
            9 => rng.gen_range(-0.5..0.5),           // Volume MA ratio
            10 => (i as f64 * 0.01).cos() * 0.5,     // OBV normalized
            11 => (i as f64 * 0.03).sin() * 0.3,     // Momentum
            _ => rng.gen_range(-1.0..1.0),
        };

        base + rng.gen_range(-0.01..0.01) // Add noise
    })
}
