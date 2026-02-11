//! Train Binary
//!
//! Trains a Neural ODE model on cryptocurrency data for trading.
//!
//! Usage:
//!   cargo run --bin train -- --symbol BTCUSDT --epochs 50 --hidden-dim 32
//!   cargo run --bin train -- --symbol ETHUSDT --epochs 100 --solver rk4

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use neural_ode_trading::data::{create_windows, generate_synthetic_klines, BybitClient};
use neural_ode_trading::model::{NeuralODE, SimpleTrainer};

#[derive(Parser, Debug)]
#[command(name = "train")]
#[command(about = "Train Neural ODE model for crypto trading")]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 50)]
    epochs: usize,

    /// Hidden dimension of the ODE
    #[arg(long, default_value_t = 32)]
    hidden_dim: usize,

    /// ODE solver step size
    #[arg(long, default_value_t = 0.05)]
    dt: f64,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f64,

    /// Lookback window size
    #[arg(long, default_value_t = 10)]
    window_size: usize,

    /// Number of data points
    #[arg(long, default_value_t = 500)]
    n_data: usize,

    /// Use synthetic data
    #[arg(long)]
    synthetic: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("================================================");
    println!("  Neural ODE Trading - Training");
    println!("================================================");
    println!("  Symbol:      {}", args.symbol);
    println!("  Hidden dim:  {}", args.hidden_dim);
    println!("  Window:      {}", args.window_size);
    println!("  Epochs:      {}", args.epochs);
    println!("  LR:          {}", args.lr);
    println!("  ODE dt:      {}", args.dt);
    println!("================================================\n");

    // Load data
    println!("Loading data...");
    let data = if args.synthetic {
        generate_synthetic_klines(&args.symbol, args.n_data)
    } else {
        let client = BybitClient::new();
        match client.fetch_klines(&args.symbol, "1", args.n_data) {
            Ok(data) => {
                println!("Fetched {} candles from Bybit", data.len());
                data
            }
            Err(e) => {
                println!("API error: {}. Using synthetic data.", e);
                generate_synthetic_klines(&args.symbol, args.n_data)
            }
        }
    };

    println!("Data points: {}", data.len());

    // Create sliding windows
    let (inputs, targets) = create_windows(&data, args.window_size, 1);
    println!("Training samples: {}", inputs.len());

    if inputs.is_empty() {
        eprintln!("Not enough data for training. Need at least {} data points.", args.window_size + 2);
        std::process::exit(1);
    }

    // Split train/test (80/20)
    let split = (inputs.len() as f64 * 0.8) as usize;
    let train_inputs = &inputs[..split];
    let train_targets = &targets[..split];
    let test_inputs = &inputs[split..];
    let test_targets = &targets[split..];

    println!("Train: {}, Test: {}", train_inputs.len(), test_inputs.len());

    // Input dim = window_size * 3 features (close, range, log_volume)
    let input_dim = args.window_size * 3;

    // Create model
    let mut model = NeuralODE::new(input_dim, args.hidden_dim, 1, args.dt);
    println!("Model parameters: {}", model.num_parameters());

    // Create trainer
    let trainer = SimpleTrainer::new(args.lr);

    // Training loop
    println!("\nTraining...");
    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} | Loss: {msg}")
            .unwrap(),
    );

    let mut best_test_loss = f64::MAX;

    for epoch in 0..args.epochs {
        // Train epoch (use subset for speed in demo)
        let batch_size = train_inputs.len().min(32);
        let batch_inputs = &train_inputs[..batch_size];
        let batch_targets = &train_targets[..batch_size];

        let train_loss = trainer.train_epoch(&mut model, batch_inputs, batch_targets);

        // Test loss
        let test_batch = test_inputs.len().min(16);
        let test_loss =
            SimpleTrainer::compute_loss(&model, &test_inputs[..test_batch], &test_targets[..test_batch]);

        if test_loss < best_test_loss {
            best_test_loss = test_loss;
        }

        pb.set_message(format!("Train={:.6}, Test={:.6}", train_loss, test_loss));
        pb.inc(1);

        if (epoch + 1) % 10 == 0 {
            pb.println(format!(
                "  Epoch {:4}/{} | Train: {:.6} | Test: {:.6} | Best: {:.6}",
                epoch + 1,
                args.epochs,
                train_loss,
                test_loss,
                best_test_loss,
            ));
        }
    }

    pb.finish_with_message("Training complete!");

    println!("\n================================================");
    println!("  Training Complete");
    println!("================================================");
    println!("  Best test loss: {:.6}", best_test_loss);
    println!("  Model params:   {}", model.num_parameters());
    println!("================================================");

    // Quick evaluation
    println!("\nSample predictions (test set):");
    println!("{:>10} {:>10} {:>10}", "Actual", "Predicted", "Error");
    for i in 0..test_inputs.len().min(5) {
        let pred = model.forward(&test_inputs[i]);
        let actual = test_targets[i][0];
        let predicted = pred[0];
        let error = (predicted - actual).abs();
        println!("{:>10.4} {:>10.4} {:>10.4}", actual, predicted, error);
    }

    Ok(())
}
