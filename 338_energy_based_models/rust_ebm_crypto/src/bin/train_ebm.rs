//! Train Energy-Based Model on market data
//!
//! Usage:
//!   cargo run --bin train_ebm -- --input data.csv --epochs 100

use clap::Parser;
use log::info;
use ndarray::Array2;
use rust_ebm_crypto::data::{OhlcvData, StandardScaler};
use rust_ebm_crypto::ebm::{EnergyModel, EnergyStats, RBM, ScoreMatchingTrainer};
use rust_ebm_crypto::features::FeatureEngine;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train Energy-Based Model")]
struct Args {
    /// Input CSV file path
    #[arg(short, long)]
    input: String,

    /// Symbol name
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Interval
    #[arg(long, default_value = "60")]
    interval: String,

    /// Number of training epochs
    #[arg(short, long, default_value = "50")]
    epochs: usize,

    /// Model type: "ebm", "rbm", or "score"
    #[arg(short, long, default_value = "ebm")]
    model: String,

    /// Hidden layer dimensions (comma-separated)
    #[arg(long, default_value = "64,32,16")]
    hidden: String,

    /// Train/test split ratio
    #[arg(long, default_value = "0.8")]
    train_ratio: f64,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    info!("Loading data from {}", args.input);

    // Load data
    let data = OhlcvData::from_csv(&args.input, &args.symbol, &args.interval)?;
    info!("Loaded {} candles", data.len());

    // Extract features
    let engine = FeatureEngine::default();
    let features = engine.compute(&data.data);
    info!(
        "Computed {} features for {} samples",
        features.ncols(),
        features.nrows()
    );

    // Print feature statistics
    let stats = engine.feature_stats(&features);
    println!("\n=== Feature Statistics ===");
    stats.print();

    // Train/test split
    let split_idx = (features.nrows() as f64 * args.train_ratio) as usize;
    let train_features = features.slice(ndarray::s![..split_idx, ..]).to_owned();
    let test_features = features.slice(ndarray::s![split_idx.., ..]).to_owned();

    info!(
        "Train samples: {}, Test samples: {}",
        train_features.nrows(),
        test_features.nrows()
    );

    // Normalize features
    let mut scaler = StandardScaler::new();
    let train_normalized = scaler.fit_transform(&train_features);
    let test_normalized = scaler.transform(&test_features);

    // Parse hidden dimensions
    let hidden_dims: Vec<usize> = args
        .hidden
        .split(',')
        .map(|s| s.trim().parse().unwrap_or(32))
        .collect();

    // Train model
    match args.model.as_str() {
        "ebm" => {
            train_energy_model(&train_normalized, &test_normalized, hidden_dims, args.epochs);
        }
        "rbm" => {
            train_rbm(&train_normalized, &test_normalized, hidden_dims[0], args.epochs);
        }
        "score" => {
            train_score_matching(
                &train_normalized,
                &test_normalized,
                hidden_dims,
                args.epochs,
            );
        }
        _ => {
            eprintln!("Unknown model type: {}. Use 'ebm', 'rbm', or 'score'", args.model);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn train_energy_model(
    train: &Array2<f64>,
    test: &Array2<f64>,
    hidden_dims: Vec<usize>,
    epochs: usize,
) {
    println!("\n=== Training Energy-Based Model ===");
    println!("Hidden layers: {:?}", hidden_dims);

    let mut model = EnergyModel::with_architecture(train.ncols(), &hidden_dims);

    // Train
    model.train(train, epochs);

    // Evaluate
    println!("\n=== Evaluation ===");

    let train_stats = model.energy_stats(train);
    let test_stats = model.energy_stats(test);

    println!("Train Energy Statistics:");
    println!("  Mean:   {:.4}", train_stats.mean);
    println!("  Std:    {:.4}", train_stats.std);
    println!("  Min:    {:.4}", train_stats.min);
    println!("  Max:    {:.4}", train_stats.max);
    println!("  Median: {:.4}", train_stats.median);

    println!("\nTest Energy Statistics:");
    println!("  Mean:   {:.4}", test_stats.mean);
    println!("  Std:    {:.4}", test_stats.std);
    println!("  Min:    {:.4}", test_stats.min);
    println!("  Max:    {:.4}", test_stats.max);
    println!("  Median: {:.4}", test_stats.median);

    // Detect anomalies
    let test_anomalies = model.detect_anomalies(test, 0.9);
    let anomaly_count = test_anomalies.iter().filter(|&&a| a).count();
    println!(
        "\nAnomalies detected in test set: {} ({:.2}%)",
        anomaly_count,
        anomaly_count as f64 / test.nrows() as f64 * 100.0
    );
}

fn train_rbm(train: &Array2<f64>, test: &Array2<f64>, n_hidden: usize, epochs: usize) {
    println!("\n=== Training Restricted Boltzmann Machine ===");
    println!("Hidden units: {}", n_hidden);

    let mut rbm = RBM::new(train.ncols(), n_hidden);

    // Train
    rbm.train_cd(train, epochs, 1);

    // Evaluate
    println!("\n=== Evaluation ===");

    let train_energies = rbm.free_energy_batch(train);
    let test_energies = rbm.free_energy_batch(test);

    let train_mean: f64 = train_energies.iter().sum::<f64>() / train_energies.len() as f64;
    let test_mean: f64 = test_energies.iter().sum::<f64>() / test_energies.len() as f64;

    println!("Train Free Energy (mean): {:.4}", train_mean);
    println!("Test Free Energy (mean):  {:.4}", test_mean);

    // Detect anomalies
    let test_anomalies = rbm.detect_anomalies(test, 0.9);
    let anomaly_count = test_anomalies.iter().filter(|&&a| a).count();
    println!(
        "\nAnomalies detected in test set: {} ({:.2}%)",
        anomaly_count,
        anomaly_count as f64 / test.nrows() as f64 * 100.0
    );
}

fn train_score_matching(
    train: &Array2<f64>,
    test: &Array2<f64>,
    hidden_dims: Vec<usize>,
    epochs: usize,
) {
    println!("\n=== Training with Score Matching ===");
    println!("Hidden layers: {:?}", hidden_dims);

    let mut model = EnergyModel::with_architecture(train.ncols(), &hidden_dims);
    let trainer = ScoreMatchingTrainer::new(0.1, 0.001);

    // Train
    trainer.train(&mut model, train, epochs);

    // Evaluate
    println!("\n=== Evaluation ===");

    let train_stats = model.energy_stats(train);
    let test_stats = model.energy_stats(test);

    println!("Train Energy Statistics:");
    println!("  Mean:   {:.4}", train_stats.mean);
    println!("  Std:    {:.4}", train_stats.std);

    println!("\nTest Energy Statistics:");
    println!("  Mean:   {:.4}", test_stats.mean);
    println!("  Std:    {:.4}", test_stats.std);
}
