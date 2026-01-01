//! Train ESN model on historical data
//!
//! Usage: cargo run --bin train_esn -- --data data.csv --output model.bin

use anyhow::Result;
use clap::Parser;
use esn_trading::{
    EchoStateNetwork, ESNConfig,
    trading::FeatureEngineering,
    utils::PredictionMetrics,
    api::Kline,
};
use ndarray::Array1;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Parser, Debug)]
#[command(author, version, about = "Train ESN model")]
struct Args {
    /// Input data file (CSV)
    #[arg(short, long)]
    data: String,

    /// Output model file
    #[arg(short, long, default_value = "esn_model.bin")]
    output: String,

    /// Reservoir size
    #[arg(long, default_value = "500")]
    reservoir_size: usize,

    /// Spectral radius
    #[arg(long, default_value = "0.95")]
    spectral_radius: f64,

    /// Leaking rate
    #[arg(long, default_value = "0.3")]
    leaking_rate: f64,

    /// Train/test split ratio
    #[arg(long, default_value = "0.8")]
    train_ratio: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("Loading data from {}...", args.data);
    let klines = load_klines(&args.data)?;
    println!("Loaded {} klines", klines.len());

    // Feature engineering
    println!("\nEngineering features...");
    let fe = FeatureEngineering::new()
        .add_returns(10)
        .add_volatility(20)
        .add_rsi(14)
        .add_momentum(5)
        .add_bollinger(20, 2.0);

    let features = fe.transform(&klines);
    println!("Generated {} feature vectors with {} features each",
             features.len(), features.get(0).map(|f| f.len()).unwrap_or(0));

    // Prepare targets (next period return)
    let returns: Vec<f64> = klines.windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();

    // Align features and targets
    let lookback = fe.required_lookback();
    let targets: Vec<Array1<f64>> = returns[lookback..]
        .iter()
        .map(|&r| Array1::from_vec(vec![r]))
        .collect();

    let n = features.len().min(targets.len());
    let features = &features[..n];
    let targets = &targets[..n];

    // Split data
    let split_idx = (n as f64 * args.train_ratio) as usize;
    let (train_features, test_features) = features.split_at(split_idx);
    let (train_targets, test_targets) = targets.split_at(split_idx);

    println!("\nData split:");
    println!("  Training samples: {}", train_features.len());
    println!("  Testing samples:  {}", test_features.len());

    // Configure ESN
    let config = ESNConfig::new(train_features[0].len(), 1)
        .reservoir_size(args.reservoir_size)
        .spectral_radius(args.spectral_radius)
        .leaking_rate(args.leaking_rate)
        .washout(100)
        .regularization(1e-6);

    println!("\nESN Configuration:");
    println!("  Reservoir size: {}", config.reservoir_size);
    println!("  Spectral radius: {}", config.spectral_radius);
    println!("  Leaking rate: {}", config.leaking_rate);

    // Train ESN
    println!("\nTraining ESN...");
    let mut esn = EchoStateNetwork::new(config);
    esn.train(train_features, train_targets);
    println!("Training complete!");

    // Evaluate on test set
    println!("\nEvaluating on test set...");
    esn.reset_state();

    let mut predictions = Vec::new();
    let mut actuals = Vec::new();

    for (feature, target) in test_features.iter().zip(test_targets.iter()) {
        let pred = esn.step(feature);
        predictions.push(pred[0]);
        actuals.push(target[0]);
    }

    // Calculate metrics
    let metrics = PredictionMetrics::calculate(&predictions, &actuals);
    metrics.print_summary();

    // Save model
    println!("\nSaving model to {}...", args.output);
    esn.save(&args.output)?;
    println!("Model saved!");

    Ok(())
}

fn load_klines(path: &str) -> Result<Vec<Kline>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut klines = Vec::new();
    let mut first_line = true;

    for line in reader.lines() {
        let line = line?;
        if first_line {
            first_line = false;
            continue; // Skip header
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 7 {
            klines.push(Kline {
                start_time: parts[0].parse().unwrap_or(0),
                open: parts[1].parse().unwrap_or(0.0),
                high: parts[2].parse().unwrap_or(0.0),
                low: parts[3].parse().unwrap_or(0.0),
                close: parts[4].parse().unwrap_or(0.0),
                volume: parts[5].parse().unwrap_or(0.0),
                turnover: parts[6].parse().unwrap_or(0.0),
            });
        }
    }

    // Sort by timestamp
    klines.sort_by_key(|k| k.start_time);

    Ok(klines)
}
