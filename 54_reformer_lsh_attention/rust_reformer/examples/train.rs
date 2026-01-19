//! Example: Train a Reformer model on cryptocurrency data
//!
//! This example demonstrates how to:
//! 1. Fetch historical data from Bybit
//! 2. Prepare a training dataset
//! 3. Create and configure a Reformer model
//! 4. Run a training loop (simplified)

use clap::Parser;
use reformer::{BybitClient, DataLoader, ReformerConfig, ReformerModel, AttentionType};
use indicatif::{ProgressBar, ProgressStyle};

/// Train a Reformer model
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of training epochs
    #[arg(short, long, default_value = "10")]
    epochs: usize,

    /// Sequence length
    #[arg(long, default_value = "168")]
    seq_len: usize,

    /// Prediction horizon
    #[arg(long, default_value = "24")]
    horizon: usize,

    /// Model dimension
    #[arg(long, default_value = "64")]
    d_model: usize,

    /// Number of attention heads
    #[arg(long, default_value = "4")]
    n_heads: usize,

    /// Number of hash rounds for LSH
    #[arg(long, default_value = "4")]
    n_hash_rounds: usize,

    /// Use full attention instead of LSH
    #[arg(long)]
    full_attention: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("=== Reformer Training Example ===");
    println!("Symbol: {}", args.symbol);
    println!("Sequence Length: {}", args.seq_len);
    println!("Prediction Horizon: {}", args.horizon);
    println!("Model Dimension: {}", args.d_model);
    println!("Attention Type: {}", if args.full_attention { "Full" } else { "LSH" });
    println!();

    // Fetch data
    println!("Fetching historical data...");
    let client = BybitClient::new();
    let klines = client.get_extended_klines(&args.symbol, "60", 2000).await?;
    println!("Fetched {} klines", klines.len());

    // Prepare dataset
    println!("\nPreparing dataset...");
    let loader = DataLoader::new();
    let dataset = loader.prepare_dataset(&klines, args.seq_len, args.horizon)?;

    let stats = dataset.statistics();
    println!("{}", stats);

    // Split data
    let (train_data, val_data) = dataset.train_val_split(0.8);
    println!("Training samples: {}", train_data.len());
    println!("Validation samples: {}", val_data.len());

    // Create model
    println!("\nCreating Reformer model...");
    let config = ReformerConfig {
        seq_len: args.seq_len,
        n_features: dataset.n_features(),
        d_model: args.d_model,
        n_heads: args.n_heads,
        d_ff: args.d_model * 4,
        n_layers: 4,
        n_hash_rounds: args.n_hash_rounds,
        n_buckets: 16,
        chunk_size: 32,
        prediction_horizon: args.horizon,
        attention_type: if args.full_attention {
            AttentionType::Full
        } else {
            AttentionType::LSH
        },
        ..Default::default()
    };

    let model = ReformerModel::new(config);
    println!("Model parameters: ~{}", model.num_parameters());
    println!("Estimated memory: {} bytes", model.config().estimated_memory());

    // Training loop (simplified - in practice you'd use autograd)
    println!("\nStarting training...");

    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    for epoch in 0..args.epochs {
        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        // Process training samples
        for sample in &train_data.samples {
            let prediction = model.predict(&sample.features);

            // Calculate MSE loss
            let loss: f64 = prediction
                .iter()
                .zip(sample.target.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>() / prediction.len() as f64;

            epoch_loss += loss;
            n_batches += 1;
        }

        epoch_loss /= n_batches as f64;

        // Validation
        let mut val_loss = 0.0;
        let mut val_batches = 0;

        for sample in &val_data.samples {
            let prediction = model.predict(&sample.features);

            let loss: f64 = prediction
                .iter()
                .zip(sample.target.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>() / prediction.len() as f64;

            val_loss += loss;
            val_batches += 1;
        }

        val_loss /= val_batches.max(1) as f64;

        pb.inc(1);

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            pb.println(format!(
                "Epoch {}/{}: Train Loss: {:.6}, Val Loss: {:.6}",
                epoch + 1, args.epochs, epoch_loss, val_loss
            ));
        }
    }

    pb.finish_with_message("Training complete!");

    // Make sample prediction
    println!("\n=== Sample Prediction ===");
    if let Some(sample) = val_data.samples.first() {
        let prediction = model.predict(&sample.features);

        println!("Predicted returns (next {} hours):", args.horizon);
        for (i, &pred) in prediction.iter().enumerate().take(5) {
            println!("  Hour {}: {:.4}%", i + 1, pred * 100.0);
        }

        println!("\nActual returns:");
        for (i, &actual) in sample.target.iter().enumerate().take(5) {
            println!("  Hour {}: {:.4}%", i + 1, actual * 100.0);
        }
    }

    println!("\nTraining example complete!");
    println!("Note: This is a demonstration. Real training would use");
    println!("automatic differentiation and proper optimization.");

    Ok(())
}
