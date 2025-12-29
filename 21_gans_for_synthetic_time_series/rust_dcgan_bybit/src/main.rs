//! DCGAN for Cryptocurrency Time Series Generation
//!
//! Main entry point providing CLI interface for:
//! - Fetching data from Bybit
//! - Training DCGAN model
//! - Generating synthetic samples

use anyhow::Result;
use clap::{Parser, Subcommand};
use tch::Device;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use rust_dcgan_bybit::{
    data::{BybitClient, DataLoader, OHLCVDataset, create_sequences, normalize_data},
    model::DCGAN,
    training::{Trainer, TrainingConfig},
    utils::{Config, load_checkpoint, save_checkpoint},
};

/// DCGAN for Synthetic Cryptocurrency Time Series
#[derive(Parser)]
#[command(name = "dcgan_bybit")]
#[command(author = "ML Trading Examples")]
#[command(version = "0.1.0")]
#[command(about = "Generate synthetic cryptocurrency price data using DCGAN")]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.json")]
    config: String,

    /// Verbosity level
    #[arg(short, long, default_value = "info")]
    verbosity: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch historical data from Bybit
    Fetch {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Kline interval (1, 5, 15, 60, 240, D)
        #[arg(short, long, default_value = "60")]
        interval: String,

        /// Number of days to fetch
        #[arg(short, long, default_value = "30")]
        days: u32,

        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Train the DCGAN model
    Train {
        /// Path to training data CSV
        #[arg(short, long)]
        data: String,

        /// Number of epochs
        #[arg(short, long, default_value = "100")]
        epochs: usize,

        /// Resume from checkpoint
        #[arg(long)]
        resume: Option<String>,
    },

    /// Generate synthetic samples
    Generate {
        /// Path to trained model checkpoint
        #[arg(short, long)]
        model: String,

        /// Number of samples to generate
        #[arg(short, long, default_value = "100")]
        num_samples: i64,

        /// Output file path
        #[arg(short, long, default_value = "synthetic_samples.csv")]
        output: String,
    },

    /// Evaluate synthetic data quality
    Evaluate {
        /// Path to real data CSV
        #[arg(long)]
        real: String,

        /// Path to synthetic data CSV
        #[arg(long)]
        synthetic: String,
    },

    /// Initialize default configuration file
    Init {
        /// Output configuration file path
        #[arg(short, long, default_value = "config.json")]
        output: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let level = match cli.verbosity.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    match cli.command {
        Commands::Fetch {
            symbol,
            interval,
            days,
            output,
        } => {
            fetch_data(&symbol, &interval, days, output).await?;
        }
        Commands::Train {
            data,
            epochs,
            resume,
        } => {
            train_model(&cli.config, &data, epochs, resume).await?;
        }
        Commands::Generate {
            model,
            num_samples,
            output,
        } => {
            generate_samples(&cli.config, &model, num_samples, &output)?;
        }
        Commands::Evaluate { real, synthetic } => {
            evaluate_data(&real, &synthetic)?;
        }
        Commands::Init { output } => {
            init_config(&output)?;
        }
    }

    Ok(())
}

/// Fetch historical data from Bybit
async fn fetch_data(
    symbol: &str,
    interval: &str,
    days: u32,
    output: Option<String>,
) -> Result<()> {
    info!("Fetching {} days of {} data for {}", days, interval, symbol);

    let client = BybitClient::new();

    // Calculate time range
    let end_time = chrono::Utc::now().timestamp_millis();
    let start_time = end_time - (days as i64 * 24 * 60 * 60 * 1000);

    let dataset = client
        .fetch_historical_klines(symbol, interval, start_time, end_time)
        .await?;

    info!("Fetched {} klines", dataset.len());

    // Save to file
    let output_path = output.unwrap_or_else(|| {
        format!("data/{}_{}_{}d.csv", symbol, interval, days)
    });

    // Create directory if needed
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    dataset.save_csv(&output_path)?;
    info!("Saved data to {}", output_path);

    Ok(())
}

/// Train the DCGAN model
async fn train_model(
    config_path: &str,
    data_path: &str,
    epochs: usize,
    resume: Option<String>,
) -> Result<()> {
    // Load configuration
    let config = if std::path::Path::new(config_path).exists() {
        Config::from_json(config_path)?
    } else {
        info!("Config file not found, using defaults");
        Config::default()
    };

    // Determine device
    let device = config.get_device();
    info!("Using device: {:?}", device);

    // Load and preprocess data
    info!("Loading data from {}", data_path);
    let dataset = OHLCVDataset::load_csv(
        data_path,
        config.data.symbol.clone(),
        config.data.interval.clone(),
    )?;

    info!("Loaded {} records", dataset.len());

    // Convert to feature matrix
    let features: Vec<[f64; 5]> = dataset.to_feature_matrix();
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
    let (normalized, _params) = normalize_data(&data);

    // Create sequences
    let sequence_length = config.data.sequence_length;
    let sequences = create_sequences(&normalized, sequence_length, 1);

    info!(
        "Created {} sequences of length {}",
        sequences.shape()[0],
        sequence_length
    );

    // Create data loader
    let mut data_loader = DataLoader::new(
        sequences,
        config.data.batch_size,
        true,  // shuffle
        true,  // drop_last
    );

    // Create model
    let mut model = DCGAN::with_defaults(
        sequence_length as i64,
        num_features as i64,
        config.model.latent_dim,
        device,
    );

    // Resume from checkpoint if specified
    let start_epoch = if let Some(checkpoint_path) = resume {
        let (epoch, _metrics) = load_checkpoint(&mut model, &checkpoint_path)?;
        info!("Resumed from epoch {}", epoch);
        epoch
    } else {
        0
    };

    // Create trainer
    let training_config = TrainingConfig {
        epochs,
        gen_lr: config.training.gen_lr,
        disc_lr: config.training.disc_lr,
        disc_steps: config.training.disc_steps,
        checkpoint_every: config.training.checkpoint_every,
        checkpoint_dir: config.training.checkpoint_dir.clone(),
        label_smoothing: config.training.label_smoothing,
        ..Default::default()
    };

    let mut trainer = Trainer::new(training_config, device);

    // Train
    info!("Starting training for {} epochs", epochs);
    let metrics = trainer.train(&mut model, &mut data_loader);

    info!(
        "Training complete. Final G_loss: {:.4}, D_loss: {:.4}",
        metrics.latest_gen_loss().unwrap_or(0.0),
        metrics.latest_disc_loss().unwrap_or(0.0)
    );

    Ok(())
}

/// Generate synthetic samples
fn generate_samples(
    config_path: &str,
    model_path: &str,
    num_samples: i64,
    output_path: &str,
) -> Result<()> {
    // Load configuration
    let config = if std::path::Path::new(config_path).exists() {
        Config::from_json(config_path)?
    } else {
        Config::default()
    };

    let device = config.get_device();

    // Create model
    let mut model = DCGAN::with_defaults(
        config.data.sequence_length as i64,
        config.model.num_features,
        config.model.latent_dim,
        device,
    );

    // Load checkpoint
    let gen_path = format!("{}/generator.pt", model_path);
    let disc_path = format!("{}/discriminator.pt", model_path);
    model.load(&gen_path, &disc_path)?;

    info!("Loaded model from {}", model_path);

    // Generate samples
    info!("Generating {} synthetic samples", num_samples);
    let samples = model.generate(num_samples);

    // Convert to Vec and save
    let samples_vec: Vec<f64> = samples.flatten(0, -1).try_into()?;

    let seq_len = config.data.sequence_length;
    let num_features = config.model.num_features as usize;

    // Save to CSV
    let mut writer = csv::Writer::from_path(output_path)?;
    writer.write_record(["sample_id", "timestep", "open", "high", "low", "close", "volume"])?;

    for sample_idx in 0..num_samples as usize {
        for t in 0..seq_len {
            let base_idx = sample_idx * seq_len * num_features + t * num_features;
            writer.write_record([
                sample_idx.to_string(),
                t.to_string(),
                samples_vec[base_idx].to_string(),
                samples_vec[base_idx + 1].to_string(),
                samples_vec[base_idx + 2].to_string(),
                samples_vec[base_idx + 3].to_string(),
                samples_vec[base_idx + 4].to_string(),
            ])?;
        }
    }

    writer.flush()?;
    info!("Saved synthetic samples to {}", output_path);

    Ok(())
}

/// Evaluate synthetic data quality
fn evaluate_data(real_path: &str, synthetic_path: &str) -> Result<()> {
    info!("Evaluating synthetic data quality...");
    info!("Real data: {}", real_path);
    info!("Synthetic data: {}", synthetic_path);

    // Load real data
    let real_dataset = OHLCVDataset::load_csv(
        real_path,
        "REAL".to_string(),
        "".to_string(),
    )?;

    // Calculate basic statistics for real data
    let real_returns = real_dataset.calculate_log_returns();
    let real_mean: f64 = real_returns.iter().sum::<f64>() / real_returns.len() as f64;
    let real_var: f64 = real_returns
        .iter()
        .map(|r| (r - real_mean).powi(2))
        .sum::<f64>()
        / real_returns.len() as f64;
    let real_std = real_var.sqrt();

    info!("Real data statistics:");
    info!("  - Mean return: {:.6}", real_mean);
    info!("  - Std return: {:.6}", real_std);
    info!("  - Num samples: {}", real_dataset.len());

    // Note: Full evaluation would require loading synthetic data
    // and computing distributional distances, discriminative scores, etc.
    info!("Full evaluation metrics would include:");
    info!("  - PCA/t-SNE visualization comparison");
    info!("  - Discriminative score (classifier accuracy)");
    info!("  - Predictive score (downstream task performance)");
    info!("  - Maximum Mean Discrepancy (MMD)");

    Ok(())
}

/// Initialize default configuration file
fn init_config(output_path: &str) -> Result<()> {
    let config = Config::default();

    if output_path.ends_with(".toml") {
        config.save_toml(output_path)?;
    } else {
        config.save_json(output_path)?;
    }

    info!("Created default configuration at {}", output_path);
    Ok(())
}
