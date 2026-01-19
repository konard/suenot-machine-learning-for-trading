//! Training Example
//!
//! Demonstrates how to train the Cross-Attention Multi-Asset model.
//!
//! Usage:
//!     cargo run --release --example train -- --epochs 50 --batch-size 32
//!
//! Note: The --fetch-data flag is currently not implemented. The example only supports mock data.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use cross_attention_multi_asset::model::{CrossAttentionMultiAsset, ModelConfig, OutputType};
use rand::prelude::*;

#[derive(Debug)]
struct TrainArgs {
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    use_mock_data: bool,
}

impl Default for TrainArgs {
    fn default() -> Self {
        Self {
            epochs: 50,
            batch_size: 32,
            learning_rate: 0.001,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            use_mock_data: true,
        }
    }
}

fn parse_args() -> TrainArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut train_args = TrainArgs::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--epochs" => {
                if i + 1 < args.len() {
                    train_args.epochs = args[i + 1].parse().unwrap_or(50);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--batch-size" => {
                if i + 1 < args.len() {
                    train_args.batch_size = args[i + 1].parse().unwrap_or(32);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--learning-rate" => {
                if i + 1 < args.len() {
                    train_args.learning_rate = args[i + 1].parse().unwrap_or(0.001);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--d-model" => {
                if i + 1 < args.len() {
                    train_args.d_model = args[i + 1].parse().unwrap_or(64);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--n-heads" => {
                if i + 1 < args.len() {
                    train_args.n_heads = args[i + 1].parse().unwrap_or(4);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--fetch-data" => {
                train_args.use_mock_data = false;
                i += 1;
            }
            _ => i += 1,
        }
    }

    train_args
}

/// Generate mock data for testing (when Bybit API is not available)
fn generate_mock_data(n_samples: usize, n_assets: usize, seq_len: usize, n_features: usize)
    -> (Vec<f32>, Vec<f32>)
{
    let mut rng = thread_rng();
    let mut x = Vec::with_capacity(n_samples * n_assets * seq_len * n_features);
    let mut y = Vec::with_capacity(n_samples * n_assets);

    for i in 0..n_samples {
        // Generate input features
        for a in 0..n_assets {
            for t in 0..seq_len {
                for f in 0..n_features {
                    // Simple sinusoidal patterns with noise
                    let val = ((i + t + a + f) as f64 * 0.1).sin() * 0.1
                        + (rng.gen::<f64>() - 0.5) * 0.01;
                    x.push(val as f32);
                }
            }
        }

        // Generate target returns (small random values)
        let mut weights: Vec<f32> = (0..n_assets)
            .map(|a| (1.0 + ((i + a) as f64 * 0.1).sin() * 0.2) as f32)
            .collect();

        // Normalize to sum to 1
        let total: f32 = weights.iter().sum();
        for w in &mut weights {
            *w /= total;
        }

        y.extend(weights);
    }

    (x, y)
}

fn main() -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("Cross-Attention Multi-Asset Trading - Training Example");
    println!("{}", "=".repeat(60));

    let args = parse_args();
    println!("\nTraining configuration:");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.learning_rate);
    println!("  Model dimension: {}", args.d_model);
    println!("  Attention heads: {}", args.n_heads);
    println!("  Encoder layers: {}", args.n_layers);
    println!("  Using mock data: {}", args.use_mock_data);

    // Setup device
    let device = Device::Cpu;
    println!("\nUsing device: {:?}", device);

    // Data parameters
    let n_assets = 5;
    let n_features = 6;
    let seq_len = 50;  // Reduced for faster training
    let n_train_samples = 500;
    let n_val_samples = 100;

    // Generate data
    println!("\n{}", "-".repeat(40));
    println!("Preparing data...");
    println!("{}", "-".repeat(40));

    // Check if user requested real data fetching (not yet implemented)
    if !args.use_mock_data {
        eprintln!("Error: --fetch-data is not implemented for this example yet.");
        eprintln!("Please run without --fetch-data to use mock data for demonstration.");
        std::process::exit(1);
    }

    let (train_x, train_y) = generate_mock_data(n_train_samples, n_assets, seq_len, n_features);
    let (val_x, val_y) = generate_mock_data(n_val_samples, n_assets, seq_len, n_features);

    println!("Train samples: {}", n_train_samples);
    println!("Val samples: {}", n_val_samples);
    println!("Input shape: [{}, {}, {}, {}]", n_train_samples, n_assets, seq_len, n_features);

    // Create tensors
    let train_x_tensor = Tensor::from_vec(
        train_x,
        (n_train_samples, n_assets, seq_len, n_features),
        &device,
    )?;
    let train_y_tensor = Tensor::from_vec(
        train_y,
        (n_train_samples, n_assets),
        &device,
    )?;

    let val_x_tensor = Tensor::from_vec(
        val_x,
        (n_val_samples, n_assets, seq_len, n_features),
        &device,
    )?;
    let val_y_tensor = Tensor::from_vec(
        val_y,
        (n_val_samples, n_assets),
        &device,
    )?;

    // Create model
    println!("\n{}", "-".repeat(40));
    println!("Creating model...");
    println!("{}", "-".repeat(40));

    let model_config = ModelConfig {
        n_assets,
        n_features,
        seq_len,
        d_model: args.d_model,
        n_heads: args.n_heads,
        n_layers: args.n_layers,
        d_ff: args.d_model * 4,
        dropout: 0.1,
        output_type: OutputType::Portfolio,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = CrossAttentionMultiAsset::new(&model_config, vb)?;

    let n_params: usize = varmap
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();
    println!("Model parameters: {}", n_params);

    // Create optimizer
    let adamw_params = ParamsAdamW {
        lr: args.learning_rate,
        weight_decay: 0.01,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

    // Training loop
    println!("\n{}", "-".repeat(40));
    println!("Training...");
    println!("{}", "-".repeat(40));

    let mut best_val_loss = f64::INFINITY;

    for epoch in 0..args.epochs {
        // Training
        let mut train_loss = 0.0;
        let n_batches = (n_train_samples + args.batch_size - 1) / args.batch_size;

        for batch_idx in 0..n_batches {
            let start = batch_idx * args.batch_size;
            let end = (start + args.batch_size).min(n_train_samples);
            let batch_size = end - start;

            // Get batch
            let batch_x = train_x_tensor.narrow(0, start, batch_size)?;
            let batch_y = train_y_tensor.narrow(0, start, batch_size)?;

            // Forward pass
            let (predictions, _) = model.forward(&batch_x, true)?;

            // Portfolio loss: negative expected return
            let portfolio_return = (&predictions * &batch_y)?.sum_all()?;
            let loss = portfolio_return.neg()?;

            // Backward pass
            optimizer.backward_step(&loss)?;

            train_loss += loss.to_scalar::<f32>()? as f64;
        }

        train_loss /= n_batches as f64;

        // Validation
        let (val_predictions, _) = model.forward(&val_x_tensor, false)?;
        let val_return = (&val_predictions * &val_y_tensor)?.sum_all()?;
        let val_loss = -val_return.to_scalar::<f32>()? as f64;

        // Track best model
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            // Save model weights here if needed
        }

        // Log progress
        if epoch % 10 == 0 || epoch == args.epochs - 1 {
            println!(
                "Epoch {:4}/{}: Train Loss = {:.6}, Val Loss = {:.6}{}",
                epoch + 1,
                args.epochs,
                train_loss,
                val_loss,
                if val_loss <= best_val_loss { " *" } else { "" }
            );
        }
    }

    println!("\nTraining complete!");
    println!("Best validation loss: {:.6}", best_val_loss);

    // Final evaluation
    println!("\n{}", "-".repeat(40));
    println!("Final Evaluation");
    println!("{}", "-".repeat(40));

    let (final_predictions, attention_weights) = model.forward(&val_x_tensor, false)?;

    // Print sample predictions
    println!("\nSample predictions (first 5):");
    println!("{:>10} {:>10} {:>10} {:>10} {:>10}", "Asset 0", "Asset 1", "Asset 2", "Asset 3", "Asset 4");

    for i in 0..5.min(n_val_samples) {
        let weights: Vec<f32> = final_predictions.get(i)?.to_vec1()?;
        println!(
            "{:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            weights[0], weights[1], weights[2], weights[3], weights[4]
        );
    }

    // Check that weights sum to 1
    let weight_sums: Vec<f32> = final_predictions.sum(1)?.to_vec1()?;
    let avg_sum: f32 = weight_sums.iter().sum::<f32>() / weight_sums.len() as f32;
    println!("\nAverage weight sum: {:.6} (should be 1.0)", avg_sum);

    // Attention analysis
    if let Some((_, cross_attn)) = attention_weights.last() {
        println!("\nCross-Asset Attention (last layer, first sample):");
        let attn_sample = cross_attn.get(0)?.mean(0)?; // Average over heads
        let attn_matrix: Vec<Vec<f32>> = (0..n_assets)
            .map(|i| attn_sample.get(i).unwrap().to_vec1().unwrap())
            .collect();

        print!("{:>8}", "");
        for i in 0..n_assets {
            print!("{:>8}", format!("A{}", i));
        }
        println!();

        for i in 0..n_assets {
            print!("{:>8}", format!("A{}", i));
            for j in 0..n_assets {
                if i == j {
                    print!("{:>8}", "-");
                } else {
                    print!("{:>8.3}", attn_matrix[i][j]);
                }
            }
            println!();
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Training example complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}
