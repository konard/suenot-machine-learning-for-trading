//! Example: Train Linformer model on financial data.
//!
//! Run with: cargo run --example train

use linformer::prelude::*;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("Linformer Training Example");
    println!("==========================\n");

    // Model configuration
    let seq_len = 128;
    let n_features = 6;
    let d_model = 64;
    let n_heads = 4;
    let k = 32;
    let n_layers = 2;

    println!("Model Configuration:");
    println!("  Sequence length: {}", seq_len);
    println!("  Features: {}", n_features);
    println!("  Model dimension: {}", d_model);
    println!("  Attention heads: {}", n_heads);
    println!("  Projection k: {}", k);
    println!("  Layers: {}", n_layers);

    // Create model
    let config = LinformerConfig::new(d_model, n_heads, seq_len, k, n_layers)
        .with_n_features(n_features)
        .with_n_outputs(1)
        .with_dropout(0.1);

    let model = Linformer::new(config)?;

    println!("\n{}", model.summary());

    // Generate synthetic training data
    println!("\nGenerating synthetic training data...");
    let n_samples = 500;
    let prices: Vec<f64> = (0..n_samples)
        .map(|i| 100.0 + 10.0 * (i as f64 * 0.1).sin() + rand::random::<f64>() * 2.0)
        .collect();

    let prices_arr = ndarray::Array1::from_vec(prices.clone());
    let features = TechnicalFeatures::calculate_all(&prices_arr);
    let normalized_features = TechnicalFeatures::normalize_zscore(&features);

    // Create sequence dataset
    let dataset = SequenceDataset::from_features(&normalized_features, &prices, seq_len, 1);

    println!("Created {} sequences", dataset.len());

    // Split into train/val
    let (train_idx, val_idx) = dataset.train_val_split(0.8);
    println!("Train set: {} samples", train_idx.len());
    println!("Val set: {} samples", val_idx.len());

    // Simulate training loop
    println!("\nSimulating training (inference only, no gradient descent in this demo)...");
    let batch_size = 16;
    let n_epochs = 3;

    for epoch in 0..n_epochs {
        let mut epoch_loss = 0.0;
        let n_batches = (train_idx.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(train_idx.len());
            let batch_indices: Vec<usize> = train_idx[start..end].to_vec();

            let (batch_features, batch_targets, _) = dataset.get_batch(&batch_indices);

            // Forward pass for each sample in batch
            let (batch_len, _, _) = batch_features.dim();
            for b in 0..batch_len {
                let x = batch_features.slice(ndarray::s![b, .., ..]).to_owned();
                let pred = model.forward(&x);
                let target = batch_targets[b][0];

                // Simple MSE loss
                let loss = (pred[0] - target).powi(2);
                epoch_loss += loss;
            }
        }

        println!(
            "Epoch {}/{}: Avg Loss = {:.6}",
            epoch + 1,
            n_epochs,
            epoch_loss / train_idx.len() as f64
        );
    }

    // Validation
    println!("\nValidation...");
    let val_batch: Vec<usize> = val_idx.iter().take(50).cloned().collect();
    let (val_features, val_targets, val_directions) = dataset.get_batch(&val_batch);

    let mut correct = 0;
    let (n_val, _, _) = val_features.dim();

    for i in 0..n_val {
        let x = val_features.slice(ndarray::s![i, .., ..]).to_owned();
        let pred = model.forward(&x);
        let pred_direction = if pred[0] > 0.0 { 1 } else { 0 };

        if pred_direction == val_directions[i] {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / n_val as f64;
    println!("Validation Direction Accuracy: {:.2}%", accuracy * 100.0);

    println!("\nNote: This is a demonstration. For actual training,");
    println!("you would need gradient descent optimization (e.g., using tch-rs).");

    Ok(())
}
