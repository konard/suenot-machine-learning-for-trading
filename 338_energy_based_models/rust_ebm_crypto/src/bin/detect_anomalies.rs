//! Detect anomalies in market data using EBM
//!
//! Usage:
//!   cargo run --bin detect_anomalies -- --symbol BTCUSDT --limit 500

use clap::Parser;
use log::info;
use rust_ebm_crypto::data::{BybitClient, StandardScaler};
use rust_ebm_crypto::ebm::{EnergyModel, MarketRegime, OnlineEnergyEstimator};
use rust_ebm_crypto::features::FeatureEngine;

#[derive(Parser, Debug)]
#[command(author, version, about = "Detect anomalies in market data")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to analyze
    #[arg(short, long, default_value = "500")]
    limit: usize,

    /// Anomaly threshold (in standard deviations)
    #[arg(short, long, default_value = "2.0")]
    threshold: f64,

    /// Use online estimator (vs batch)
    #[arg(long)]
    online: bool,

    /// Show only anomalies
    #[arg(long)]
    anomalies_only: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    info!(
        "Fetching {} candles for {} ({})",
        args.limit, args.symbol, args.interval
    );

    // Fetch data
    let client = BybitClient::public();
    let data = client.get_klines(&args.symbol, &args.interval, args.limit.min(1000) as u32, None, None)?;

    info!("Fetched {} candles", data.len());

    // Extract features
    let engine = FeatureEngine::default();
    let features = engine.compute(&data.data);

    if args.online {
        detect_online(&data.data, &features, &engine, args.threshold, args.anomalies_only);
    } else {
        detect_batch(&data.data, &features, &engine, args.threshold, args.anomalies_only);
    }

    Ok(())
}

fn detect_online(
    candles: &[rust_ebm_crypto::data::Candle],
    features: &ndarray::Array2<f64>,
    _engine: &FeatureEngine,
    threshold: f64,
    anomalies_only: bool,
) {
    println!("\n=== Online Anomaly Detection ===");
    println!("Threshold: {} standard deviations\n", threshold);

    let mut estimator = OnlineEnergyEstimator::new(100, 0.1);
    let mut anomaly_count = 0;

    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "Timestamp", "Close", "Energy", "Normalized", "Regime", "Anomaly"
    );
    println!("{}", "-".repeat(90));

    for (i, candle) in candles.iter().enumerate() {
        let feature_row = features.row(i).to_owned();
        let result = estimator.update(&feature_row);

        let is_anomaly = result.normalized_energy > threshold;
        if is_anomaly {
            anomaly_count += 1;
        }

        if !anomalies_only || is_anomaly {
            println!(
                "{:<20} {:>12.4} {:>12.4} {:>12.4} {:>10} {:>10}",
                candle.datetime().format("%Y-%m-%d %H:%M"),
                candle.close,
                result.energy,
                result.normalized_energy,
                result.regime.as_str(),
                if is_anomaly { "YES" } else { "-" }
            );
        }
    }

    println!("{}", "-".repeat(90));
    println!(
        "\nTotal anomalies: {} / {} ({:.2}%)",
        anomaly_count,
        candles.len(),
        anomaly_count as f64 / candles.len() as f64 * 100.0
    );

    // Print statistics
    let stats = estimator.get_stats();
    println!("\nEnergy Statistics:");
    println!("  Mean:        {:.4}", stats.mean);
    println!("  Std:         {:.4}", stats.std);
    println!("  Min:         {:.4}", stats.min);
    println!("  Max:         {:.4}", stats.max);
    println!("  Buffer Size: {}", stats.buffer_size);
}

fn detect_batch(
    candles: &[rust_ebm_crypto::data::Candle],
    features: &ndarray::Array2<f64>,
    engine: &FeatureEngine,
    threshold: f64,
    anomalies_only: bool,
) {
    println!("\n=== Batch Anomaly Detection ===");
    println!("Training EBM on {} samples...\n", features.nrows());

    // Normalize features
    let mut scaler = StandardScaler::new();
    let normalized = scaler.fit_transform(features);

    // Train model
    let mut model = EnergyModel::with_architecture(features.ncols(), &[32, 16]);
    model.train(&normalized, 30);

    // Get energy scores
    let energies = model.energy_batch(&normalized);
    let stats = model.energy_stats(&normalized);

    // Detect anomalies
    let mut anomaly_count = 0;

    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>10}",
        "Timestamp", "Close", "Energy", "Z-Score", "Anomaly"
    );
    println!("{}", "-".repeat(70));

    for (i, candle) in candles.iter().enumerate() {
        let energy = energies[i];
        let zscore = stats.normalize(energy);
        let is_anomaly = zscore.abs() > threshold;

        if is_anomaly {
            anomaly_count += 1;
        }

        if !anomalies_only || is_anomaly {
            println!(
                "{:<20} {:>12.4} {:>12.4} {:>12.4} {:>10}",
                candle.datetime().format("%Y-%m-%d %H:%M"),
                candle.close,
                energy,
                zscore,
                if is_anomaly { "YES" } else { "-" }
            );
        }
    }

    println!("{}", "-".repeat(70));
    println!(
        "\nTotal anomalies: {} / {} ({:.2}%)",
        anomaly_count,
        candles.len(),
        anomaly_count as f64 / candles.len() as f64 * 100.0
    );

    println!("\nEnergy Statistics:");
    println!("  Mean:   {:.4}", stats.mean);
    println!("  Std:    {:.4}", stats.std);
    println!("  Min:    {:.4}", stats.min);
    println!("  Max:    {:.4}", stats.max);
    println!("  Median: {:.4}", stats.median);
}
