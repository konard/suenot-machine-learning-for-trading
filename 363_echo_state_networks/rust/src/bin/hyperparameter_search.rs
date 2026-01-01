//! Hyperparameter search for ESN
//!
//! Usage: cargo run --bin hyperparameter_search -- --data data.csv

use anyhow::Result;
use clap::Parser;
use esn_trading::{
    EchoStateNetwork, ESNConfig,
    api::Kline,
    trading::FeatureEngineering,
    utils::PredictionMetrics,
};
use ndarray::Array1;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Parser, Debug)]
#[command(author, version, about = "Hyperparameter search for ESN")]
struct Args {
    /// Input data file (CSV)
    #[arg(short, long)]
    data: String,

    /// Output results file
    #[arg(short, long, default_value = "hyperparameter_results.csv")]
    output: String,

    /// Train/test split ratio
    #[arg(long, default_value = "0.7")]
    train_ratio: f64,

    /// Validation ratio (from training set)
    #[arg(long, default_value = "0.2")]
    val_ratio: f64,
}

#[derive(Clone)]
struct HyperParams {
    reservoir_size: usize,
    spectral_radius: f64,
    leaking_rate: f64,
    input_scaling: f64,
    regularization: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("Loading data from {}...", args.data);
    let klines = load_klines(&args.data)?;
    println!("Loaded {} klines", klines.len());

    // Feature engineering
    let fe = FeatureEngineering::new()
        .add_returns(10)
        .add_volatility(20)
        .add_rsi(14)
        .add_momentum(5)
        .add_bollinger(20, 2.0);

    let features = fe.transform(&klines);
    let lookback = fe.required_lookback();

    // Prepare targets
    let returns: Vec<f64> = klines.windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();

    let targets: Vec<Array1<f64>> = returns[lookback..]
        .iter()
        .map(|&r| Array1::from_vec(vec![r]))
        .collect();

    let n = features.len().min(targets.len());
    let features = &features[..n];
    let targets = &targets[..n];

    // Split data: train / validation / test
    let train_end = (n as f64 * args.train_ratio * (1.0 - args.val_ratio)) as usize;
    let val_end = (n as f64 * args.train_ratio) as usize;

    let train_features = &features[..train_end];
    let train_targets = &targets[..train_end];
    let val_features = &features[train_end..val_end];
    let val_targets = &targets[train_end..val_end];
    let test_features = &features[val_end..];
    let test_targets = &targets[val_end..];

    println!("\nData splits:");
    println!("  Train: {}", train_features.len());
    println!("  Val:   {}", val_features.len());
    println!("  Test:  {}", test_features.len());

    // Define parameter grid
    let reservoir_sizes = vec![100, 200, 500, 1000];
    let spectral_radii = vec![0.8, 0.9, 0.95, 0.99];
    let leaking_rates = vec![0.1, 0.3, 0.5, 0.7];
    let input_scalings = vec![0.05, 0.1, 0.2];
    let regularizations = vec![1e-8, 1e-6, 1e-4];

    let total_combinations = reservoir_sizes.len()
        * spectral_radii.len()
        * leaking_rates.len()
        * input_scalings.len()
        * regularizations.len();

    println!("\nTotal combinations to test: {}", total_combinations);
    println!("Starting hyperparameter search...\n");

    let mut results: Vec<(HyperParams, f64, f64)> = Vec::new();
    let mut best_val_mse = f64::INFINITY;
    let mut best_params: Option<HyperParams> = None;
    let mut count = 0;

    let input_dim = train_features[0].len();

    for &res_size in &reservoir_sizes {
        for &sr in &spectral_radii {
            for &lr in &leaking_rates {
                for &is in &input_scalings {
                    for &reg in &regularizations {
                        count += 1;

                        let params = HyperParams {
                            reservoir_size: res_size,
                            spectral_radius: sr,
                            leaking_rate: lr,
                            input_scaling: is,
                            regularization: reg,
                        };

                        // Train ESN
                        let config = ESNConfig::new(input_dim, 1)
                            .reservoir_size(res_size)
                            .spectral_radius(sr)
                            .leaking_rate(lr)
                            .input_scaling(is)
                            .regularization(reg)
                            .washout(50);

                        let mut esn = EchoStateNetwork::new(config);
                        esn.train(train_features, train_targets);

                        // Evaluate on validation set
                        esn.reset_state();
                        let mut predictions = Vec::new();
                        let mut actuals = Vec::new();

                        for (f, t) in val_features.iter().zip(val_targets.iter()) {
                            let pred = esn.step(f);
                            predictions.push(pred[0]);
                            actuals.push(t[0]);
                        }

                        let metrics = PredictionMetrics::calculate(&predictions, &actuals);

                        results.push((params.clone(), metrics.mse, metrics.directional_accuracy));

                        if metrics.mse < best_val_mse {
                            best_val_mse = metrics.mse;
                            best_params = Some(params.clone());
                            println!("[{}/{}] New best! MSE={:.8}, Dir Acc={:.2}%",
                                count, total_combinations, metrics.mse, metrics.directional_accuracy * 100.0);
                            println!("        res={}, sr={}, lr={}, is={}, reg={}",
                                res_size, sr, lr, is, reg);
                        }

                        if count % 50 == 0 {
                            println!("[{}/{}] Progress: {:.1}%",
                                count, total_combinations, count as f64 / total_combinations as f64 * 100.0);
                        }
                    }
                }
            }
        }
    }

    // Save results
    let mut file = File::create(&args.output)?;
    writeln!(file, "reservoir_size,spectral_radius,leaking_rate,input_scaling,regularization,val_mse,val_dir_acc")?;
    for (params, mse, dir_acc) in &results {
        writeln!(file, "{},{},{},{},{},{},{}",
            params.reservoir_size,
            params.spectral_radius,
            params.leaking_rate,
            params.input_scaling,
            params.regularization,
            mse,
            dir_acc
        )?;
    }
    println!("\nResults saved to {}", args.output);

    // Evaluate best model on test set
    if let Some(best) = best_params {
        println!("\n=== Best Hyperparameters ===");
        println!("Reservoir size: {}", best.reservoir_size);
        println!("Spectral radius: {}", best.spectral_radius);
        println!("Leaking rate: {}", best.leaking_rate);
        println!("Input scaling: {}", best.input_scaling);
        println!("Regularization: {}", best.regularization);

        // Retrain on full training set and evaluate on test
        let config = ESNConfig::new(input_dim, 1)
            .reservoir_size(best.reservoir_size)
            .spectral_radius(best.spectral_radius)
            .leaking_rate(best.leaking_rate)
            .input_scaling(best.input_scaling)
            .regularization(best.regularization)
            .washout(50);

        let full_train_features = &features[..val_end];
        let full_train_targets = &targets[..val_end];

        let mut esn = EchoStateNetwork::new(config);
        esn.train(full_train_features, full_train_targets);

        esn.reset_state();
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();

        for (f, t) in test_features.iter().zip(test_targets.iter()) {
            let pred = esn.step(f);
            predictions.push(pred[0]);
            actuals.push(t[0]);
        }

        println!("\n=== Test Set Performance ===");
        let metrics = PredictionMetrics::calculate(&predictions, &actuals);
        metrics.print_summary();

        // Save best model
        esn.save("best_esn_model.bin")?;
        println!("\nBest model saved to best_esn_model.bin");
    }

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
            continue;
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

    klines.sort_by_key(|k| k.start_time);
    Ok(klines)
}
