//! Train InceptionTime model
//!
//! This binary trains an InceptionTime ensemble on historical data.

use anyhow::Result;
use clap::Parser;

use inception_time_trading::{Config, setup_logging};

#[derive(Parser)]
#[command(name = "train")]
#[command(about = "Train InceptionTime ensemble model")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,

    /// Path to training data CSV
    #[arg(short, long)]
    data: Option<String>,

    /// Number of epochs (overrides config)
    #[arg(short, long)]
    epochs: Option<u32>,

    /// Path to save the trained model
    #[arg(short, long)]
    output: Option<String>,

    /// Resume from checkpoint
    #[arg(long)]
    resume: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    setup_logging("info")?;

    let config = Config::load_or_default(&args.config);

    println!("\nInceptionTime Training");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("[CONFIG] Configuration loaded from: {}", args.config);
    println!("[CONFIG] Symbol: {}", config.data.symbol);
    println!("[CONFIG] Interval: {}m", config.data.interval);
    println!("[CONFIG] Window size: {}", config.data.window_size);
    println!();

    println!("[MODEL] InceptionTime Ensemble");
    println!("[MODEL] Ensemble size: {}", config.model.ensemble_size);
    println!("[MODEL] Depth: {}", config.model.depth);
    println!("[MODEL] Filters: {}", config.model.num_filters);
    println!("[MODEL] Kernel sizes: {:?}", config.model.kernel_sizes);
    println!("[MODEL] Dropout: {}", config.model.dropout);
    println!();

    let epochs = args.epochs.unwrap_or(config.training.epochs);
    println!("[TRAINING] Epochs: {}", epochs);
    println!("[TRAINING] Batch size: {}", config.training.batch_size);
    println!("[TRAINING] Learning rate: {}", config.training.learning_rate);
    println!("[TRAINING] Early stopping patience: {}", config.training.early_stopping_patience);
    println!();

    if let Some(data_path) = &args.data {
        println!("[DATA] Loading from: {}", data_path);
    } else {
        println!("[DATA] No data path specified, would fetch from Bybit");
    }

    if let Some(resume_path) = &args.resume {
        println!("[RESUME] Resuming from checkpoint: {}", resume_path);
    }

    if let Some(output_path) = &args.output {
        println!("[OUTPUT] Model will be saved to: {}", output_path);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("                     TRAINING NOTES");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("To run actual training, you need:");
    println!("  1. Install libtorch (PyTorch C++ library)");
    println!("  2. Set LIBTORCH environment variable");
    println!("  3. Have sufficient training data (CSV or fetch from Bybit)");
    println!();
    println!("Example with libtorch:");
    println!("  export LIBTORCH=/path/to/libtorch");
    println!("  export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH");
    println!("  cargo run --release --bin train -- --data data/btcusdt_15_90d.csv");
    println!();
    println!("Training workflow:");
    println!("  1. Load and preprocess data");
    println!("  2. Generate features (RSI, MACD, BB, etc.)");
    println!("  3. Create windowed sequences");
    println!("  4. Train {} InceptionTime models", config.model.ensemble_size);
    println!("  5. Evaluate on test set");
    println!("  6. Save best ensemble");
    println!();
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}
