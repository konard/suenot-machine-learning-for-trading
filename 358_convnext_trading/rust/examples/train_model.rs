//! Example: Train ConvNeXt model
//!
//! This example demonstrates how to train a ConvNeXt model on historical data.
//!
//! Usage:
//!   cargo run --example train_model -- --data data/btcusdt_1h.json --epochs 50

use anyhow::Result;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use convnext_trading::convnext::{ConvNeXt, ConvNeXtConfig};
use convnext_trading::data::{Candle, Dataset, FeatureBuilder};
use convnext_trading::utils::Metrics;

#[derive(Parser)]
#[command(name = "train_model")]
#[command(about = "Train a ConvNeXt model for trading")]
struct Args {
    /// Path to training data (JSON file with candles)
    #[arg(short, long)]
    data: String,

    /// Number of training epochs
    #[arg(short, long, default_value = "50")]
    epochs: u32,

    /// Batch size
    #[arg(short, long, default_value = "32")]
    batch_size: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.0004")]
    learning_rate: f64,

    /// Sequence length for model input
    #[arg(short, long, default_value = "256")]
    seq_length: usize,

    /// Test set ratio
    #[arg(short, long, default_value = "0.2")]
    test_ratio: f64,

    /// Output model path
    #[arg(short, long)]
    output: Option<String>,

    /// Model variant (tiny, small, base)
    #[arg(short, long, default_value = "tiny")]
    model: String,
}

fn main() -> Result<()> {
    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    info!("=== ConvNeXt Training ===");
    info!("Data: {}", args.data);
    info!("Epochs: {}", args.epochs);
    info!("Batch size: {}", args.batch_size);
    info!("Learning rate: {}", args.learning_rate);
    info!("Sequence length: {}", args.seq_length);
    info!("Model variant: {}", args.model);

    // Load data
    info!("\nLoading data...");
    let data = std::fs::read_to_string(&args.data)?;
    let candles: Vec<Candle> = serde_json::from_str(&data)?;
    info!("Loaded {} candles", candles.len());

    // Build features
    info!("Building features...");
    let feature_builder = FeatureBuilder::new();
    let features = feature_builder.build(&candles)?;
    info!(
        "Feature matrix shape: [{}, {}]",
        features.dim().0,
        features.dim().1
    );

    // Create dataset
    info!("Creating dataset...");
    let dataset = Dataset::from_features(features, args.seq_length)?;
    info!("Dataset size: {} samples", dataset.len());

    // Print class distribution
    let dist = dataset.class_distribution();
    info!(
        "Class distribution: Long={}, Short={}, Hold={}",
        dist[0], dist[1], dist[2]
    );

    // Split into train and test
    let (train_set, test_set) = dataset.train_test_split(args.test_ratio);
    info!(
        "Train set: {} samples, Test set: {} samples",
        train_set.len(),
        test_set.len()
    );

    // Create model
    info!("\nCreating model...");
    let config = match args.model.as_str() {
        "tiny" => ConvNeXtConfig::tiny(),
        "small" => ConvNeXtConfig::small(),
        "base" => ConvNeXtConfig::base(),
        _ => ConvNeXtConfig::tiny(),
    };
    let mut model = ConvNeXt::new(config);
    info!("Model parameters: {}", model.count_params());

    // Training loop
    info!("\nStarting training...");
    let mut best_loss = f64::INFINITY;

    for epoch in 0..args.epochs {
        let mut total_loss = 0.0;
        let mut n_batches = 0;
        let mut predictions = Vec::new();
        let mut targets = Vec::new();

        for (x, y) in train_set.batches(args.batch_size) {
            // Forward pass
            let output = model.forward(&x);

            // Calculate loss (cross-entropy)
            let mut batch_loss = 0.0;
            for i in 0..y.len() {
                let target = y[i];
                let prob = output[[i, target]].max(1e-10);
                batch_loss -= prob.ln();

                // Track predictions
                let pred = (0..3)
                    .max_by(|&a, &b| output[[i, a]].partial_cmp(&output[[i, b]]).unwrap())
                    .unwrap();
                predictions.push(pred);
                targets.push(target);
            }
            batch_loss /= y.len() as f64;

            // Backward pass (simplified)
            model.backward(&output, &y, args.learning_rate);

            total_loss += batch_loss;
            n_batches += 1;
        }

        let avg_loss = total_loss / n_batches as f64;

        // Calculate training accuracy
        let pred_array = ndarray::Array1::from(predictions.clone());
        let target_array = ndarray::Array1::from(targets.clone());
        let train_acc = Metrics::accuracy(&pred_array, &target_array);

        // Evaluate on test set
        let (test_acc, test_loss) = evaluate_model(&model, &test_set, args.batch_size)?;

        // Log progress
        if epoch % 5 == 0 || epoch == args.epochs - 1 {
            info!(
                "Epoch {}/{}: Train Loss={:.4}, Train Acc={:.2}%, Test Loss={:.4}, Test Acc={:.2}%",
                epoch + 1,
                args.epochs,
                avg_loss,
                train_acc * 100.0,
                test_loss,
                test_acc * 100.0
            );
        }

        // Track best model
        if test_loss < best_loss {
            best_loss = test_loss;
        }
    }

    // Final evaluation
    info!("\n=== Final Evaluation ===");
    let (test_acc, _) = evaluate_model(&model, &test_set, args.batch_size)?;
    info!("Test Accuracy: {:.2}%", test_acc * 100.0);

    // Save model
    let output_path = args
        .output
        .unwrap_or_else(|| format!("models/convnext_{}.json", args.model));

    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    model.save(&output_path)?;
    info!("\nSaved model to {}", output_path);

    info!("\n=== Training Complete ===");
    Ok(())
}

fn evaluate_model(model: &ConvNeXt, dataset: &Dataset, batch_size: usize) -> Result<(f64, f64)> {
    let mut predictions = Vec::new();
    let mut targets = Vec::new();
    let mut total_loss = 0.0;
    let mut n_batches = 0;

    for (x, y) in dataset.batches(batch_size) {
        let output = model.forward(&x);

        // Calculate loss
        let mut batch_loss = 0.0;
        for i in 0..y.len() {
            let target = y[i];
            let prob = output[[i, target]].max(1e-10);
            batch_loss -= prob.ln();

            let pred = (0..3)
                .max_by(|&a, &b| output[[i, a]].partial_cmp(&output[[i, b]]).unwrap())
                .unwrap();
            predictions.push(pred);
            targets.push(target);
        }

        total_loss += batch_loss / y.len() as f64;
        n_batches += 1;
    }

    let pred_array = ndarray::Array1::from(predictions);
    let target_array = ndarray::Array1::from(targets);
    let accuracy = Metrics::accuracy(&pred_array, &target_array);
    let avg_loss = total_loss / n_batches as f64;

    Ok((accuracy, avg_loss))
}
