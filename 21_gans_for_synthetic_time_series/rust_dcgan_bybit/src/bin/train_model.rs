//! Standalone binary for training DCGAN model
//!
//! Usage:
//!   cargo run --bin train_model -- --data data/BTCUSDT_60_30d.csv --epochs 100

use anyhow::Result;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use rust_dcgan_bybit::{
    data::{DataLoader, OHLCVDataset, create_sequences, normalize_data},
    model::DCGAN,
    training::{Trainer, TrainingConfig},
    utils::{load_checkpoint, Config},
};

/// Train DCGAN model on cryptocurrency data
#[derive(Parser)]
#[command(name = "train_model")]
#[command(about = "Train DCGAN on OHLCV data")]
struct Args {
    /// Path to training data CSV
    #[arg(short, long)]
    data: String,

    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value = "64")]
    batch_size: usize,

    /// Sequence length (number of timesteps)
    #[arg(short, long, default_value = "24")]
    sequence_length: usize,

    /// Latent dimension size
    #[arg(long, default_value = "100")]
    latent_dim: i64,

    /// Generator learning rate
    #[arg(long, default_value = "0.0002")]
    gen_lr: f64,

    /// Discriminator learning rate
    #[arg(long, default_value = "0.0002")]
    disc_lr: f64,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: String,

    /// Save checkpoint every N epochs
    #[arg(long, default_value = "10")]
    checkpoint_every: usize,

    /// Resume from checkpoint directory
    #[arg(long)]
    resume: Option<String>,

    /// Use GPU if available
    #[arg(long)]
    gpu: bool,
}

fn main() -> Result<()> {
    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    // Determine device
    let device = if args.gpu && tch::Cuda::is_available() {
        info!("Using CUDA GPU");
        tch::Device::Cuda(0)
    } else {
        info!("Using CPU");
        tch::Device::Cpu
    };

    // Load data
    info!("Loading data from {}", args.data);
    let dataset = OHLCVDataset::load_csv(&args.data, "DATA".to_string(), "".to_string())?;
    info!("Loaded {} records", dataset.len());

    // Convert to feature matrix
    let features = dataset.to_feature_matrix();
    let num_samples = features.len();
    let num_features = 5;

    // Create ndarray
    let mut data = ndarray::Array2::<f64>::zeros((num_samples, num_features));
    for (i, row) in features.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            data[[i, j]] = val;
        }
    }

    // Normalize data
    info!("Normalizing data to [-1, 1] range");
    let (normalized, norm_params) = normalize_data(&data);

    // Save normalization parameters for later denormalization
    let norm_params_json = serde_json::json!({
        "min_vals": norm_params.min_vals,
        "max_vals": norm_params.max_vals,
    });
    std::fs::create_dir_all(&args.checkpoint_dir)?;
    std::fs::write(
        format!("{}/norm_params.json", args.checkpoint_dir),
        serde_json::to_string_pretty(&norm_params_json)?,
    )?;

    // Create sequences
    let sequences = create_sequences(&normalized, args.sequence_length, 1);
    info!(
        "Created {} sequences of length {} with {} features",
        sequences.shape()[0],
        args.sequence_length,
        num_features
    );

    // Verify we have enough data
    if sequences.shape()[0] < args.batch_size {
        anyhow::bail!(
            "Not enough sequences ({}) for batch size ({}). \
             Try reducing sequence_length or batch_size, or use more data.",
            sequences.shape()[0],
            args.batch_size
        );
    }

    // Create data loader
    let mut data_loader = DataLoader::new(sequences, args.batch_size, true, true);

    info!(
        "DataLoader: {} batches of size {}",
        data_loader.num_batches(),
        args.batch_size
    );

    // Create model
    let mut model = DCGAN::with_defaults(
        args.sequence_length as i64,
        num_features as i64,
        args.latent_dim,
        device,
    );

    info!(
        "Created DCGAN: latent_dim={}, seq_len={}, features={}",
        args.latent_dim, args.sequence_length, num_features
    );

    // Resume from checkpoint if specified
    if let Some(checkpoint_path) = &args.resume {
        info!("Resuming from checkpoint: {}", checkpoint_path);
        let (epoch, metrics) = load_checkpoint(&mut model, checkpoint_path)?;
        info!(
            "Resumed from epoch {} (G_loss: {:.4}, D_loss: {:.4})",
            epoch,
            metrics.latest_gen_loss().unwrap_or(0.0),
            metrics.latest_disc_loss().unwrap_or(0.0)
        );
    }

    // Create training config
    let training_config = TrainingConfig {
        epochs: args.epochs,
        gen_lr: args.gen_lr,
        disc_lr: args.disc_lr,
        disc_steps: 1,
        checkpoint_every: args.checkpoint_every,
        checkpoint_dir: args.checkpoint_dir.clone(),
        label_smoothing: true,
        smooth_real: 0.9,
        smooth_fake: 0.1,
    };

    // Create trainer
    let mut trainer = Trainer::new(training_config, device);

    // Train
    info!("Starting training for {} epochs", args.epochs);
    info!("  Generator LR: {}", args.gen_lr);
    info!("  Discriminator LR: {}", args.disc_lr);
    info!("  Label smoothing: enabled (real=0.9, fake=0.1)");

    let metrics = trainer.train(&mut model, &mut data_loader);

    // Print final results
    info!("Training complete!");
    info!(
        "Final metrics: G_loss={:.4}, D_loss={:.4}",
        metrics.latest_gen_loss().unwrap_or(0.0),
        metrics.latest_disc_loss().unwrap_or(0.0)
    );
    info!(
        "Model saved to {}/generator_final.pt",
        args.checkpoint_dir
    );

    Ok(())
}
