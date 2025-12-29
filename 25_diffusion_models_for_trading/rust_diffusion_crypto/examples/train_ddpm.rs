//! Example: Train a DDPM model on cryptocurrency data.
//!
//! Usage:
//! ```bash
//! cargo run --example train_ddpm -- --data data/ohlcv.csv --epochs 100
//! ```

use anyhow::Result;
use clap::Parser;
use ndarray::Array1;
use tch::Device;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use diffusion_crypto::{
    data::{
        BybitClient, FeatureEngineer, OHLCVDataset,
        normalize, create_sequences, TimeSeriesDataset, DataLoader,
    },
    model::{DDPM, NoiseSchedule},
    training::{Trainer, TrainingConfig},
};

#[derive(Parser)]
#[command(name = "train_ddpm")]
#[command(about = "Train DDPM model on cryptocurrency data")]
struct Args {
    /// Path to OHLCV data file
    #[arg(short, long)]
    data: String,

    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value = "32")]
    batch_size: usize,

    /// Sequence length
    #[arg(long, default_value = "100")]
    sequence_length: usize,

    /// Forecast horizon
    #[arg(long, default_value = "24")]
    forecast_horizon: usize,

    /// Number of diffusion steps
    #[arg(long, default_value = "500")]
    diffusion_steps: usize,

    /// Learning rate
    #[arg(long, default_value = "0.0001")]
    learning_rate: f64,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: String,

    /// Use GPU if available
    #[arg(long)]
    gpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    // Select device
    let device = if args.gpu && tch::Cuda::is_available() {
        info!("Using CUDA GPU");
        Device::Cuda(0)
    } else {
        info!("Using CPU");
        Device::Cpu
    };

    // Load data
    info!("Loading data from: {}", args.data);
    let dataset = OHLCVDataset::from_csv(&args.data, "BTCUSDT".to_string(), "60".to_string())?;
    info!("Loaded {} records", dataset.len());

    // Compute features
    info!("Computing features...");
    let engineer = FeatureEngineer::new();
    let features = engineer.compute_all(&dataset.data);
    info!("Computed {} features: {:?}", features.ncols(), engineer.feature_names());

    // Normalize features
    let (normalized_features, norm_params) = normalize(&features);
    info!("Normalized features");

    // Get targets (close prices)
    let closes: Vec<f64> = dataset.closes();
    let target_mean = closes.iter().sum::<f64>() / closes.len() as f64;
    let target_std = (closes.iter().map(|x| (x - target_mean).powi(2)).sum::<f64>() / closes.len() as f64).sqrt();

    // Normalize targets
    let normalized_targets: Vec<f64> = closes.iter().map(|x| (x - target_mean) / target_std).collect();
    let targets = Array1::from_vec(normalized_targets);

    // Create sequences
    info!(
        "Creating sequences (seq_len={}, horizon={})...",
        args.sequence_length, args.forecast_horizon
    );
    let (x, y) = create_sequences(
        &normalized_features,
        &targets,
        args.sequence_length,
        args.forecast_horizon,
    );
    info!("Created {} training sequences", x.dim().0);

    // Create dataset
    let ts_dataset = TimeSeriesDataset::new(
        x, y,
        norm_params,
        target_mean,
        target_std,
        device,
    );

    // Create dataloader
    let mut train_loader = DataLoader::new(ts_dataset, args.batch_size, true);
    info!("Created dataloader with {} batches", train_loader.num_batches());

    // Create noise schedule
    let schedule = NoiseSchedule::cosine(args.diffusion_steps);
    info!("Created cosine noise schedule with {} steps", args.diffusion_steps);

    // Create model
    let model = DDPM::new(
        features.ncols() as i64,
        args.sequence_length as i64,
        args.forecast_horizon as i64,
        256,  // hidden_dim
        &schedule,
        device,
    );
    info!("Created DDPM model");

    // Create trainer
    let config = TrainingConfig {
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        grad_clip: 1.0,
        log_interval: 10,
        checkpoint_interval: 20,
        checkpoint_dir: args.checkpoint_dir.clone(),
    };

    let mut trainer = Trainer::new(model, config, device);

    // Train
    info!("Starting training for {} epochs...", args.epochs);
    let losses = trainer.train(&mut train_loader)?;

    // Print final statistics
    info!("Training complete!");
    info!("Final loss: {:.6}", losses.last().unwrap_or(&0.0));
    info!("Best loss: {:.6}", losses.iter().cloned().fold(f64::INFINITY, f64::min));

    Ok(())
}
