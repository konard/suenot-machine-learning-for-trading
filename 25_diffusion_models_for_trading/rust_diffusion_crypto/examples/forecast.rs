//! Example: Generate forecasts using a trained DDPM model.
//!
//! Usage:
//! ```bash
//! cargo run --example forecast -- --model checkpoints/ddpm_final.pt --data data/ohlcv.csv
//! ```

use anyhow::Result;
use clap::Parser;
use tch::{Device, Tensor, Kind};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use diffusion_crypto::{
    data::{FeatureEngineer, OHLCVDataset, normalize},
    model::{DDPM, NoiseSchedule},
};

#[derive(Parser)]
#[command(name = "forecast")]
#[command(about = "Generate forecasts using trained DDPM model")]
struct Args {
    /// Path to trained model
    #[arg(short, long)]
    model: String,

    /// Path to data file
    #[arg(short, long)]
    data: String,

    /// Context length (sequence length)
    #[arg(long, default_value = "100")]
    context_length: usize,

    /// Forecast horizon
    #[arg(long, default_value = "24")]
    forecast_horizon: usize,

    /// Number of Monte Carlo samples
    #[arg(short, long, default_value = "100")]
    num_samples: i64,

    /// Output file path
    #[arg(short, long, default_value = "forecasts.csv")]
    output: String,

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

    // Normalize
    let (normalized_features, norm_params) = normalize(&features);

    // Get target statistics for inverse transform
    let closes = dataset.closes();
    let target_mean = closes.iter().sum::<f64>() / closes.len() as f64;
    let target_std = (closes.iter().map(|x| (x - target_mean).powi(2)).sum::<f64>() / closes.len() as f64).sqrt();

    // Create noise schedule
    let schedule = NoiseSchedule::cosine(500);

    // Create and load model
    let mut model = DDPM::new(
        features.ncols() as i64,
        args.context_length as i64,
        args.forecast_horizon as i64,
        256,
        &schedule,
        device,
    );

    info!("Loading model from: {}", args.model);
    model.load(&args.model)?;

    // Get the last context_length observations
    let n = normalized_features.nrows();
    if n < args.context_length {
        anyhow::bail!("Not enough data for context length {}", args.context_length);
    }

    let context_start = n - args.context_length;
    let context: Vec<f64> = (context_start..n)
        .flat_map(|i| (0..features.ncols()).map(move |j| normalized_features[[i, j]]))
        .collect();

    let context_tensor = Tensor::from_slice(&context)
        .reshape(&[1, args.context_length as i64, features.ncols() as i64])
        .to_kind(Kind::Float)
        .to(device);

    // Generate forecast
    info!("Generating {} samples...", args.num_samples);
    let forecast = model.forecast(&context_tensor, args.num_samples);

    // Convert to vectors
    let mean_vec = forecast.mean_vec();
    let std_vec = forecast.std_vec();

    // Inverse transform
    let mean_prices: Vec<f64> = mean_vec.iter().map(|x| x * target_std + target_mean).collect();
    let std_prices: Vec<f64> = std_vec.iter().map(|x| x * target_std).collect();

    // Print results
    info!("\n=== Forecast Results ===");
    info!("Current price: ${:.2}", closes.last().unwrap_or(&0.0));
    info!("\nHour | Mean Price | Std Dev | 95% CI");
    info!("{}", "-".repeat(50));

    for (i, (mean, std)) in mean_prices.iter().zip(std_prices.iter()).enumerate() {
        let ci_low = mean - 1.96 * std;
        let ci_high = mean + 1.96 * std;
        info!(
            "{:>4} | ${:>10.2} | ${:>7.2} | [${:.2} - ${:.2}]",
            i + 1, mean, std, ci_low, ci_high
        );
    }

    // Save to CSV
    let mut writer = csv::Writer::from_path(&args.output)?;
    writer.write_record(&["hour", "mean", "std", "ci_low", "ci_high"])?;

    for (i, (mean, std)) in mean_prices.iter().zip(std_prices.iter()).enumerate() {
        writer.write_record(&[
            (i + 1).to_string(),
            format!("{:.2}", mean),
            format!("{:.2}", std),
            format!("{:.2}", mean - 1.96 * std),
            format!("{:.2}", mean + 1.96 * std),
        ])?;
    }

    writer.flush()?;
    info!("\nForecasts saved to: {}", args.output);

    Ok(())
}
