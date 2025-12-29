//! Standalone binary for generating synthetic samples
//!
//! Usage:
//!   cargo run --bin generate_samples -- --model checkpoints --num-samples 100

use anyhow::Result;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use rust_dcgan_bybit::{
    data::NormalizationParams,
    model::DCGAN,
};

/// Generate synthetic cryptocurrency time series
#[derive(Parser)]
#[command(name = "generate_samples")]
#[command(about = "Generate synthetic time series using trained DCGAN")]
struct Args {
    /// Path to checkpoint directory or model files
    #[arg(short, long)]
    model: String,

    /// Number of samples to generate
    #[arg(short, long, default_value = "100")]
    num_samples: i64,

    /// Sequence length (must match training)
    #[arg(short, long, default_value = "24")]
    sequence_length: i64,

    /// Latent dimension (must match training)
    #[arg(long, default_value = "100")]
    latent_dim: i64,

    /// Output CSV file
    #[arg(short, long, default_value = "synthetic_samples.csv")]
    output: String,

    /// Use GPU if available
    #[arg(long)]
    gpu: bool,

    /// Denormalize output using saved parameters
    #[arg(long)]
    denormalize: bool,

    /// Generate interpolated samples between random points
    #[arg(long)]
    interpolate: bool,

    /// Number of interpolation steps
    #[arg(long, default_value = "10")]
    interp_steps: i64,
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

    // Create model
    let num_features = 5i64;
    let mut model = DCGAN::with_defaults(
        args.sequence_length,
        num_features,
        args.latent_dim,
        device,
    );

    // Load model weights
    let gen_path = if std::path::Path::new(&args.model).is_dir() {
        format!("{}/generator_final.pt", args.model)
    } else {
        args.model.clone()
    };

    let disc_path = gen_path.replace("generator", "discriminator");

    info!("Loading generator from {}", gen_path);
    model.load(&gen_path, &disc_path)?;

    // Load normalization parameters if denormalizing
    let norm_params: Option<NormalizationParams> = if args.denormalize {
        let norm_path = if std::path::Path::new(&args.model).is_dir() {
            format!("{}/norm_params.json", args.model)
        } else {
            let parent = std::path::Path::new(&args.model)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            format!("{}/norm_params.json", parent.display())
        };

        if std::path::Path::new(&norm_path).exists() {
            let content = std::fs::read_to_string(&norm_path)?;
            let json: serde_json::Value = serde_json::from_str(&content)?;
            Some(NormalizationParams {
                min_vals: json["min_vals"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
                max_vals: json["max_vals"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            })
        } else {
            info!("Normalization parameters not found at {}", norm_path);
            None
        }
    } else {
        None
    };

    // Generate samples
    let samples = if args.interpolate {
        info!(
            "Generating {} interpolated sequences with {} steps each",
            args.num_samples, args.interp_steps
        );

        let mut all_samples = Vec::new();
        for i in 0..args.num_samples {
            let z1 = tch::Tensor::randn([args.latent_dim], (tch::Kind::Float, device));
            let z2 = tch::Tensor::randn([args.latent_dim], (tch::Kind::Float, device));
            let interp = model.interpolate(&z1, &z2, args.interp_steps);
            all_samples.push(interp);
        }
        tch::Tensor::cat(&all_samples, 0)
    } else {
        info!("Generating {} random samples", args.num_samples);
        model.generate(args.num_samples)
    };

    let total_samples = samples.size()[0];
    info!("Generated {} samples", total_samples);

    // Convert to Vec
    let samples_vec: Vec<f64> = samples.flatten(0, -1).try_into()?;
    let seq_len = args.sequence_length as usize;
    let num_features_usize = num_features as usize;

    // Denormalize if requested
    let samples_vec = if let Some(ref params) = norm_params {
        info!("Denormalizing samples");
        denormalize_vec(&samples_vec, params, seq_len, num_features_usize)
    } else {
        samples_vec
    };

    // Save to CSV
    let mut writer = csv::Writer::from_path(&args.output)?;

    if args.interpolate {
        writer.write_record([
            "interp_group",
            "interp_step",
            "timestep",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ])?;

        let interp_steps = args.interp_steps as usize;
        for group_idx in 0..(total_samples as usize / interp_steps) {
            for step in 0..interp_steps {
                let sample_idx = group_idx * interp_steps + step;
                for t in 0..seq_len {
                    let base_idx = sample_idx * seq_len * num_features_usize + t * num_features_usize;
                    writer.write_record([
                        group_idx.to_string(),
                        step.to_string(),
                        t.to_string(),
                        format!("{:.6}", samples_vec[base_idx]),
                        format!("{:.6}", samples_vec[base_idx + 1]),
                        format!("{:.6}", samples_vec[base_idx + 2]),
                        format!("{:.6}", samples_vec[base_idx + 3]),
                        format!("{:.6}", samples_vec[base_idx + 4]),
                    ])?;
                }
            }
        }
    } else {
        writer.write_record([
            "sample_id",
            "timestep",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ])?;

        for sample_idx in 0..total_samples as usize {
            for t in 0..seq_len {
                let base_idx = sample_idx * seq_len * num_features_usize + t * num_features_usize;
                writer.write_record([
                    sample_idx.to_string(),
                    t.to_string(),
                    format!("{:.6}", samples_vec[base_idx]),
                    format!("{:.6}", samples_vec[base_idx + 1]),
                    format!("{:.6}", samples_vec[base_idx + 2]),
                    format!("{:.6}", samples_vec[base_idx + 3]),
                    format!("{:.6}", samples_vec[base_idx + 4]),
                ])?;
            }
        }
    }

    writer.flush()?;
    info!("Saved synthetic samples to {}", args.output);

    // Print sample statistics
    print_sample_stats(&samples_vec, seq_len, num_features_usize);

    Ok(())
}

/// Denormalize vector of samples
fn denormalize_vec(
    data: &[f64],
    params: &NormalizationParams,
    seq_len: usize,
    num_features: usize,
) -> Vec<f64> {
    let mut result = data.to_vec();

    for i in (0..data.len()).step_by(num_features) {
        for f in 0..num_features {
            let idx = i + f;
            if idx < result.len() {
                let range = params.max_vals[f] - params.min_vals[f];
                result[idx] = (data[idx] + 1.0) / 2.0 * range + params.min_vals[f];
            }
        }
    }

    result
}

/// Print basic statistics of generated samples
fn print_sample_stats(data: &[f64], seq_len: usize, num_features: usize) {
    let num_samples = data.len() / (seq_len * num_features);

    let feature_names = ["Open", "High", "Low", "Close", "Volume"];

    info!("Sample statistics ({} samples):", num_samples);

    for (f, name) in feature_names.iter().enumerate() {
        let values: Vec<f64> = data
            .iter()
            .enumerate()
            .filter(|(i, _)| i % num_features == f)
            .map(|(_, &v)| v)
            .collect();

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let min: f64 = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max: f64 = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        info!(
            "  {}: mean={:.4}, min={:.4}, max={:.4}",
            name, mean, min, max
        );
    }
}
