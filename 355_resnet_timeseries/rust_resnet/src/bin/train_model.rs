//! Train ResNet model on cryptocurrency data
//!
//! This binary demonstrates the training loop for ResNet on time series data.
//! Note: This is a demonstration of the architecture. For actual training,
//! you would typically use a deep learning framework with GPU support.
//!
//! Usage:
//!   cargo run --bin train_model -- --data data/BTCUSDT_1_10000candles.csv

use anyhow::Result;
use ndarray::Array3;
use rust_resnet::{
    api::Candle,
    data::{Dataset, StandardScaler},
    model::ResNet18,
    utils::Metrics,
};
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Load candles from CSV file
fn load_csv(path: &str) -> Result<Vec<Candle>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut candles = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;

        // Skip header
        if i == 0 {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 7 {
            candles.push(Candle::new(
                parts[0].parse().unwrap_or(0),
                parts[2].parse().unwrap_or(0.0),
                parts[3].parse().unwrap_or(0.0),
                parts[4].parse().unwrap_or(0.0),
                parts[5].parse().unwrap_or(0.0),
                parts[6].parse().unwrap_or(0.0),
                parts.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            ));
        }
    }

    Ok(candles)
}

/// Training configuration
struct TrainConfig {
    data_path: String,
    sequence_length: usize,
    forward_window: usize,
    threshold: f32,
    train_ratio: f32,
    val_ratio: f32,
    batch_size: usize,
    epochs: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_path: "data/BTCUSDT_1_10000candles.csv".to_string(),
            sequence_length: 256,
            forward_window: 12,
            threshold: 0.002,
            train_ratio: 0.7,
            val_ratio: 0.15,
            batch_size: 32,
            epochs: 10,
        }
    }
}

fn parse_args() -> TrainConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut config = TrainConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" | "-d" => {
                if i + 1 < args.len() {
                    config.data_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--seq-len" => {
                if i + 1 < args.len() {
                    config.sequence_length = args[i + 1].parse().unwrap_or(256);
                    i += 1;
                }
            }
            "--forward" => {
                if i + 1 < args.len() {
                    config.forward_window = args[i + 1].parse().unwrap_or(12);
                    i += 1;
                }
            }
            "--threshold" => {
                if i + 1 < args.len() {
                    config.threshold = args[i + 1].parse().unwrap_or(0.002);
                    i += 1;
                }
            }
            "--batch-size" => {
                if i + 1 < args.len() {
                    config.batch_size = args[i + 1].parse().unwrap_or(32);
                    i += 1;
                }
            }
            "--epochs" => {
                if i + 1 < args.len() {
                    config.epochs = args[i + 1].parse().unwrap_or(10);
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("Train ResNet model on cryptocurrency data\n");
                println!("Usage: train_model [OPTIONS]\n");
                println!("Options:");
                println!("  -d, --data <PATH>        Path to CSV data file");
                println!("  --seq-len <N>            Sequence length (default: 256)");
                println!("  --forward <N>            Forward window for labels (default: 12)");
                println!("  --threshold <F>          Return threshold (default: 0.002)");
                println!("  --batch-size <N>         Batch size (default: 32)");
                println!("  --epochs <N>             Number of epochs (default: 10)");
                println!("  -h, --help               Show this help message");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    config
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let config = parse_args();

    println!("=== ResNet Training Demo ===\n");

    // Load data
    println!("Loading data from: {}", config.data_path);
    let candles = load_csv(&config.data_path)?;
    println!("Loaded {} candles", candles.len());

    if candles.len() < config.sequence_length + config.forward_window + 100 {
        println!(
            "\nError: Not enough data. Need at least {} candles.",
            config.sequence_length + config.forward_window + 100
        );
        println!("Run 'cargo run --bin fetch_data' first to download data.");
        return Ok(());
    }

    // Create dataset
    println!("\nCreating dataset...");
    println!("  Sequence length: {}", config.sequence_length);
    println!("  Forward window:  {}", config.forward_window);
    println!("  Threshold:       {:.2}%", config.threshold * 100.0);

    let dataset = Dataset::from_candles(
        candles,
        config.sequence_length,
        config.forward_window,
        config.threshold,
    )?;

    println!("  Total samples:   {}", dataset.len());

    // Split dataset
    let (train_data, val_data, test_data) =
        dataset.split(config.train_ratio, config.val_ratio);

    println!("  Train samples:   {}", train_data.len());
    println!("  Val samples:     {}", val_data.len());
    println!("  Test samples:    {}", test_data.len());

    // Class distribution
    let train_dist = train_data.class_distribution();
    println!("\nClass distribution (train):");
    println!("  Down:    {} ({:.1}%)", train_dist[0], 100.0 * train_dist[0] as f32 / train_data.len() as f32);
    println!("  Neutral: {} ({:.1}%)", train_dist[1], 100.0 * train_dist[1] as f32 / train_data.len() as f32);
    println!("  Up:      {} ({:.1}%)", train_dist[2], 100.0 * train_dist[2] as f32 / train_data.len() as f32);

    // Normalize data
    println!("\nNormalizing data...");
    let mut scaler = StandardScaler::new(dataset.num_features);
    let train_x = scaler.fit_transform(&train_data.x);
    let val_x = scaler.transform(&val_data.x);
    let test_x = scaler.transform(&test_data.x);

    // Create model
    println!("\nCreating ResNet-18 model...");
    let model = ResNet18::new(dataset.num_features, 3);
    println!("  Input channels:  {}", dataset.num_features);
    println!("  Output classes:  3");
    println!("  Parameters:      ~{}", model.num_params());

    // Training loop demonstration
    println!("\n=== Training Demo ===");
    println!("Note: This demo shows the forward pass only.");
    println!("For actual training, use a deep learning framework with GPU support.\n");

    for epoch in 0..config.epochs.min(3) {
        println!("Epoch {}/{}", epoch + 1, config.epochs);

        let mut batch_count = 0;
        let mut total_samples = 0;

        for (batch_x, batch_y) in train_data.batch_iter(config.batch_size, true) {
            // Normalize batch
            let normalized_batch = scaler.transform(&batch_x);

            // Forward pass
            let _logits = model.forward(&normalized_batch);

            batch_count += 1;
            total_samples += batch_y.len();

            if batch_count % 10 == 0 {
                print!("\r  Batch {}: {} samples processed", batch_count, total_samples);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
        }
        println!("\r  Processed {} batches, {} samples total", batch_count, total_samples);
    }

    // Evaluate on test set
    println!("\n=== Evaluation ===");

    let mut all_preds = Vec::new();
    let mut all_labels = Vec::new();

    for (batch_x, batch_y) in test_data.batch_iter(config.batch_size, false) {
        let normalized_batch = scaler.transform(&batch_x);
        let predictions = model.predict(&normalized_batch);

        all_preds.extend(predictions.into_iter().map(|p| p as u8));
        all_labels.extend(batch_y);
    }

    // Calculate metrics
    let metrics = Metrics::new(all_labels, all_preds, 3);

    println!("\nTest Set Results (random weights - for demo only):");
    println!("{}", metrics.classification_report());

    println!("\nNote: These results are from a randomly initialized model.");
    println!("Actual training requires:");
    println!("  1. Proper gradient computation (autodiff)");
    println!("  2. Optimizer (Adam, SGD, etc.)");
    println!("  3. GPU acceleration for practical training times");
    println!("\nConsider using frameworks like:");
    println!("  - tch-rs (PyTorch bindings)");
    println!("  - burn (native Rust deep learning)");
    println!("  - candle (Hugging Face Rust ML)");

    println!("\nDemo complete!");

    Ok(())
}
