//! Example: Demonstrate FNet model forward pass and frequency analysis.
//!
//! Usage:
//!   cargo run --example model_demo

use anyhow::Result;
use ndarray::Array3;

use fnet_trading::{FNet, FourierLayer};

fn main() -> Result<()> {
    println!("FNet Trading - Model Demo");
    println!("==========================");
    println!();

    // Model configuration
    let n_features = 8;
    let d_model = 64;
    let n_layers = 2;
    let d_ff = 128;
    let max_seq_len = 100;

    println!("Model Configuration:");
    println!("  Input features: {}", n_features);
    println!("  Model dimension: {}", d_model);
    println!("  Encoder layers: {}", n_layers);
    println!("  Feed-forward dimension: {}", d_ff);
    println!("  Max sequence length: {}", max_seq_len);
    println!();

    // Create FNet model
    println!("Creating FNet model...");
    let model = FNet::new(n_features, d_model, n_layers, d_ff, max_seq_len);

    // Print model info
    println!("  Total parameters: {}", model.num_parameters());
    println!();

    // Create sample input [batch_size, seq_len, n_features]
    let batch_size = 4;
    let seq_len = 50;

    println!("Creating sample input...");
    println!("  Batch size: {}", batch_size);
    println!("  Sequence length: {}", seq_len);
    println!("  Features: {}", n_features);

    // Generate random input (simulating normalized features)
    let input = Array3::from_shape_fn((batch_size, seq_len, n_features), |(b, s, f)| {
        ((b * 1000 + s * 10 + f) as f64 * 0.1).sin() * 0.5
    });

    println!("  Input shape: {:?}", input.dim());
    println!();

    // Run forward pass
    println!("Running forward pass...");
    let output = model.forward(&input);
    println!("  Output shape: {:?}", output.dim());

    // Show sample predictions
    println!("\nSample predictions (first batch):");
    for i in 0..output.dim().0.min(4) {
        println!("  Batch {}: {:.6}", i, output[[i, 0]]);
    }

    // Demonstrate frequency analysis
    println!("\n\nFourier Layer Analysis");
    println!("======================");
    demonstrate_fourier_layer(seq_len, d_model);

    println!("\nModel demo complete!");
    Ok(())
}

fn demonstrate_fourier_layer(seq_len: usize, d_model: usize) {
    // Create Fourier layer
    let fourier_layer = FourierLayer::new();

    // Create sample embedding [batch_size, seq_len, d_model]
    let batch_size = 1;
    let input = Array3::from_shape_fn((batch_size, seq_len, d_model), |(_, s, d)| {
        // Create a signal with multiple frequency components
        let t = s as f64 / seq_len as f64;
        (2.0 * std::f64::consts::PI * 5.0 * t).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            + 0.3 * (2.0 * std::f64::consts::PI * 20.0 * t).cos()
            + 0.1 * ((d as f64 / d_model as f64) - 0.5)
    });

    println!("  Input signal characteristics:");
    println!("    Shape: {:?}", input.dim());

    // Get frequency analysis
    let (_output, magnitudes) = fourier_layer.forward_with_magnitudes(&input);

    println!("    Frequency magnitudes shape: {:?}", magnitudes.dim());

    // Find dominant frequencies
    let mag_slice = magnitudes.slice(ndarray::s![0, .., 0]);
    let mut freq_mags: Vec<(usize, f64)> = mag_slice.iter().enumerate().map(|(i, &m)| (i, m)).collect();
    freq_mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n  Top 5 frequency components:");
    for (i, (freq_idx, magnitude)) in freq_mags.iter().take(5).enumerate() {
        let freq_hz = *freq_idx as f64 / seq_len as f64;
        println!("    {}: Index {} (freq={:.4} cycles/sample), Magnitude: {:.4}",
                 i + 1, freq_idx, freq_hz, magnitude);
    }

    // Show DC component (average)
    let dc_component = magnitudes[[0, 0, 0]];
    println!("\n  DC component (average): {:.4}", dc_component);

    // Calculate signal energy in frequency domain
    let total_energy: f64 = mag_slice.iter().map(|&m| m * m).sum();
    println!("  Total signal energy: {:.4}", total_energy.sqrt());
}
