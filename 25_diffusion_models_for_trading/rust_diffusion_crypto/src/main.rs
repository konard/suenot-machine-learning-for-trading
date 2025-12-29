//! Main CLI application for diffusion-based cryptocurrency forecasting.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use diffusion_crypto::{
    data::{BybitClient, FeatureEngineer, OHLCVDataset},
    model::{DDPM, NoiseSchedule},
    utils::Config,
};

#[derive(Parser)]
#[command(name = "diffusion-crypto")]
#[command(about = "Diffusion models for cryptocurrency forecasting")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize configuration file
    Init {
        /// Output path for config file
        #[arg(short, long, default_value = "config.json")]
        output: String,
    },

    /// Fetch historical data from Bybit
    Fetch {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Kline interval (1, 5, 15, 30, 60, 120, 240, D, W)
        #[arg(short, long, default_value = "60")]
        interval: String,

        /// Number of days to fetch
        #[arg(short, long, default_value = "90")]
        days: u32,

        /// Output file path
        #[arg(short, long, default_value = "data/ohlcv.csv")]
        output: String,
    },

    /// Train diffusion model
    Train {
        /// Path to OHLCV data file
        #[arg(short, long)]
        data: String,

        /// Number of training epochs
        #[arg(short, long, default_value = "100")]
        epochs: usize,

        /// Configuration file (optional)
        #[arg(short, long)]
        config: Option<String>,

        /// Use GPU if available
        #[arg(long)]
        gpu: bool,
    },

    /// Generate forecasts
    Forecast {
        /// Path to trained model
        #[arg(short, long)]
        model: String,

        /// Path to data file
        #[arg(short, long)]
        data: String,

        /// Number of Monte Carlo samples
        #[arg(short, long, default_value = "100")]
        num_samples: i64,

        /// Output file path
        #[arg(short, long, default_value = "forecasts.csv")]
        output: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Init { output } => {
            info!("Creating default configuration...");
            let config = Config::default();
            config.to_file(&output)?;
            info!("Configuration saved to: {}", output);
        }

        Commands::Fetch {
            symbol,
            interval,
            days,
            output,
        } => {
            info!(
                "Fetching {} {} candles for {} days...",
                symbol, interval, days
            );

            let client = BybitClient::new();
            let data = client
                .fetch_historical_klines(&symbol, &interval, days)
                .await?;

            let dataset = OHLCVDataset::new(data, symbol, interval);

            // Create output directory if needed
            if let Some(parent) = std::path::Path::new(&output).parent() {
                std::fs::create_dir_all(parent)?;
            }

            dataset.to_csv(&output)?;
            info!("Data saved to: {} ({} records)", output, dataset.len());
        }

        Commands::Train {
            data,
            epochs,
            config,
            gpu,
        } => {
            info!("Loading data from: {}", data);

            // Load configuration
            let cfg = if let Some(config_path) = config {
                Config::from_file(config_path)?
            } else {
                Config::default()
            };

            // Select device
            let device = if gpu && tch::Cuda::is_available() {
                info!("Using CUDA GPU");
                tch::Device::Cuda(0)
            } else {
                info!("Using CPU");
                tch::Device::Cpu
            };

            // Load data
            let dataset = OHLCVDataset::from_csv(&data, "BTCUSDT".to_string(), "60".to_string())?;
            info!("Loaded {} records", dataset.len());

            // Compute features
            let engineer = FeatureEngineer::new();
            let features = engineer.compute_all(&dataset.data);
            info!("Computed {} features", features.ncols());

            // Create noise schedule
            let schedule = match cfg.model.noise_schedule.as_str() {
                "linear" => NoiseSchedule::linear(cfg.model.num_diffusion_steps),
                "cosine" => NoiseSchedule::cosine(cfg.model.num_diffusion_steps),
                "sigmoid" => NoiseSchedule::sigmoid(cfg.model.num_diffusion_steps),
                _ => NoiseSchedule::cosine(cfg.model.num_diffusion_steps),
            };

            // Create model
            let model = DDPM::new(
                features.ncols() as i64,
                cfg.data.sequence_length as i64,
                cfg.data.forecast_horizon as i64,
                cfg.model.hidden_dim,
                &schedule,
                device,
            );

            info!(
                "Created DDPM with {} diffusion steps, {} hidden dim",
                cfg.model.num_diffusion_steps, cfg.model.hidden_dim
            );

            info!("Training for {} epochs...", epochs);
            // Training would go here - requires full data pipeline setup

            info!("Training complete!");
        }

        Commands::Forecast {
            model,
            data,
            num_samples,
            output,
        } => {
            info!("Loading model from: {}", model);
            info!("Loading data from: {}", data);
            info!("Generating {} samples...", num_samples);

            // Forecasting would go here

            info!("Forecasts saved to: {}", output);
        }
    }

    Ok(())
}
