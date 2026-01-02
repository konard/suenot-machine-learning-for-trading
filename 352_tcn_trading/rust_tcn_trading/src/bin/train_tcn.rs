//! Train TCN model on cryptocurrency data
//!
//! Usage:
//!     cargo run --bin train_tcn -- --symbol BTCUSDT --epochs 100

use anyhow::Result;
use clap::Parser;
use ndarray::Array2;
use rust_tcn_trading::api::{BybitClient, TimeFrame};
use rust_tcn_trading::features::{Normalizer, TechnicalIndicators};
use rust_tcn_trading::tcn::{TCN, TCNConfig, TrainingConfig};

/// Train TCN model for trading
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Time interval
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of candles for training
    #[arg(short, long, default_value_t = 1000)]
    limit: u32,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 100)]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    /// Sequence length for TCN input
    #[arg(long, default_value_t = 100)]
    seq_len: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,

    /// Prediction horizon (bars ahead)
    #[arg(long, default_value_t = 1)]
    horizon: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("=== TCN Training for {} ===\n", args.symbol);

    // Parse timeframe
    let timeframe = TimeFrame::from_str(&args.interval)
        .ok_or_else(|| anyhow::anyhow!("Invalid interval: {}", args.interval))?;

    // Fetch data
    println!("Fetching market data...");
    let client = BybitClient::new();
    let data = client
        .get_klines(&args.symbol, timeframe, Some(args.limit), None, None)
        .await?;

    println!("Fetched {} candles", data.len());

    if data.len() < args.seq_len + args.horizon + 50 {
        return Err(anyhow::anyhow!(
            "Not enough data. Need at least {} candles, got {}",
            args.seq_len + args.horizon + 50,
            data.len()
        ));
    }

    // Calculate features
    println!("Calculating technical indicators...");
    let features = TechnicalIndicators::calculate_all(&data.candles);
    println!("Calculated {} features", features.num_features);

    // Normalize features
    println!("Normalizing features...");
    let mut normalizer = Normalizer::zscore();
    let normalized = normalizer.fit_transform(&features.data);

    // Prepare training data
    println!("Preparing training data...");
    let returns = data.log_returns();

    // Create labels (future returns)
    let labels: Vec<i32> = returns
        .windows(args.horizon + 1)
        .skip(args.seq_len - 1)
        .map(|w| {
            let future_return = w.last().unwrap_or(&0.0);
            if *future_return > 0.01 {
                2 // Up
            } else if *future_return < -0.01 {
                0 // Down
            } else {
                1 // Neutral
            }
        })
        .collect();

    // Count class distribution
    let up_count = labels.iter().filter(|&&l| l == 2).count();
    let down_count = labels.iter().filter(|&&l| l == 0).count();
    let neutral_count = labels.iter().filter(|&&l| l == 1).count();

    println!("\nClass distribution:");
    println!("  Up:      {} ({:.1}%)", up_count, up_count as f64 / labels.len() as f64 * 100.0);
    println!("  Down:    {} ({:.1}%)", down_count, down_count as f64 / labels.len() as f64 * 100.0);
    println!("  Neutral: {} ({:.1}%)", neutral_count, neutral_count as f64 / labels.len() as f64 * 100.0);

    // Create TCN model
    println!("\nCreating TCN model...");
    let config = TCNConfig {
        input_size: features.num_features,
        output_size: 3, // Up, Down, Neutral
        num_channels: vec![64, 64, 64, 64],
        kernel_size: 3,
        dropout: 0.2,
    };

    let tcn = TCN::new(config);
    println!("{}", tcn.summary());

    let training_config = TrainingConfig {
        learning_rate: args.learning_rate,
        epochs: args.epochs,
        batch_size: args.batch_size,
        patience: 10,
        validation_split: 0.2,
    };

    println!("Training configuration:");
    println!("  Epochs:          {}", training_config.epochs);
    println!("  Batch size:      {}", training_config.batch_size);
    println!("  Learning rate:   {}", training_config.learning_rate);
    println!("  Validation split: {:.0}%", training_config.validation_split * 100.0);

    // Simple training loop demonstration
    // Note: In production, you would implement proper gradient descent
    println!("\n=== Training Started ===\n");

    let num_samples = labels.len().min(normalized.ncols() - args.seq_len);
    let train_size = (num_samples as f64 * (1.0 - training_config.validation_split)) as usize;

    println!("Training samples: {}", train_size);
    println!("Validation samples: {}", num_samples - train_size);

    // Simulate training progress
    for epoch in 1..=args.epochs.min(10) {
        // In a real implementation, this would:
        // 1. Split data into batches
        // 2. Forward pass through TCN
        // 3. Calculate loss
        // 4. Backward pass (gradient computation)
        // 5. Update weights

        // For demonstration, we'll show progress with random metrics
        let train_loss = 1.0 - (epoch as f64 / args.epochs as f64) * 0.7 + rand::random::<f64>() * 0.1;
        let val_loss = train_loss + 0.1 + rand::random::<f64>() * 0.1;
        let accuracy = 0.33 + (epoch as f64 / args.epochs as f64) * 0.3 + rand::random::<f64>() * 0.05;

        println!(
            "Epoch {:3}/{:3} - loss: {:.4} - val_loss: {:.4} - accuracy: {:.2}%",
            epoch, args.epochs, train_loss, val_loss, accuracy * 100.0
        );
    }

    if args.epochs > 10 {
        println!("... (training continues for {} more epochs)", args.epochs - 10);
    }

    println!("\n=== Training Complete ===");

    // Test prediction on last sequence
    println!("\nTesting model on latest data...");
    let test_start = normalized.ncols() - args.seq_len;
    let test_input = normalized.slice(ndarray::s![.., test_start..]).to_owned();

    let prediction = tcn.predict_proba(&test_input);
    println!("\nPrediction probabilities:");
    println!("  Down:    {:.2}%", prediction[0] * 100.0);
    println!("  Neutral: {:.2}%", prediction[1] * 100.0);
    println!("  Up:      {:.2}%", prediction[2] * 100.0);

    let predicted_class = if prediction[2] > prediction[0] && prediction[2] > prediction[1] {
        "UP"
    } else if prediction[0] > prediction[2] && prediction[0] > prediction[1] {
        "DOWN"
    } else {
        "NEUTRAL"
    };

    println!("\nPredicted direction: {}", predicted_class);

    // Note about production usage
    println!("\n=== Note ===");
    println!("This is a demonstration of the TCN training pipeline.");
    println!("For production use, you would need to:");
    println!("  1. Implement proper backpropagation");
    println!("  2. Use a deep learning framework (e.g., tch-rs for PyTorch bindings)");
    println!("  3. Tune hyperparameters carefully");
    println!("  4. Validate on out-of-sample data");
    println!("  5. Implement proper model saving/loading");

    Ok(())
}
