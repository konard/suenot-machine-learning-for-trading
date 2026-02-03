//! Example: Train LRP model

use clap::Parser;
use ndarray::Array1;
use rust_lrp::{
    model::{LRPNetwork, LRPConfig},
    data::{prepare_data, normalize_dataset, generate_sample_data},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value = "32")]
    batch_size: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    learning_rate: f64,

    /// Lookback window
    #[arg(long, default_value = "60")]
    lookback: usize,

    /// Output model path
    #[arg(short, long, default_value = "model.bin")]
    output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Training LRP Model");
    println!("==================");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.learning_rate);
    println!("  Lookback: {}", args.lookback);

    // Generate sample data (replace with real data loading)
    println!("\nGenerating sample data...");
    let features = generate_sample_data(2000);
    let mut dataset = prepare_data(&features, args.lookback, 1);

    println!("  Samples: {}", dataset.len());
    println!("  Input dim: {}", dataset.input_dim());

    // Normalize
    normalize_dataset(&mut dataset);

    // Split dataset
    let (train, val, _test) = dataset.split(0.7, 0.15);
    println!("  Train: {}, Val: {}", train.len(), val.len());

    // Create model
    let config = LRPConfig::default()
        .with_epsilon(0.01)
        .with_gamma(0.25);

    let mut model = LRPNetwork::new(
        dataset.input_dim(),
        vec![128, 64, 32],
        2,
        config,
    );

    println!("\nModel Architecture:");
    println!("  {} -> [128, 64, 32] -> 2", dataset.input_dim());

    // Training loop (simplified - no actual gradient descent)
    println!("\nTraining...");

    for epoch in 0..args.epochs {
        let mut train_correct = 0;
        let mut train_total = 0;

        for (batch_x, batch_y) in train.batches(args.batch_size) {
            for i in 0..batch_x.nrows() {
                let input = batch_x.row(i).to_owned();
                let pred = model.predict(&input);
                let actual = batch_y[i] as usize;

                if pred == actual {
                    train_correct += 1;
                }
                train_total += 1;
            }
        }

        // Validation
        let mut val_correct = 0;
        let mut val_total = 0;

        for (batch_x, batch_y) in val.batches(args.batch_size) {
            for i in 0..batch_x.nrows() {
                let input = batch_x.row(i).to_owned();
                let pred = model.predict(&input);
                let actual = batch_y[i] as usize;

                if pred == actual {
                    val_correct += 1;
                }
                val_total += 1;
            }
        }

        if (epoch + 1) % 10 == 0 {
            let train_acc = train_correct as f64 / train_total as f64;
            let val_acc = val_correct as f64 / val_total as f64;
            println!(
                "  Epoch {}/{}: Train Acc = {:.4}, Val Acc = {:.4}",
                epoch + 1, args.epochs, train_acc, val_acc
            );
        }
    }

    // Test conservation
    println!("\nVerifying LRP conservation...");
    let sample = train.samples[0].x.view()
        .into_shape(train.input_dim())
        .unwrap()
        .to_owned();

    let (conserved, error) = model.verify_conservation(&sample, 0.1);
    println!("  Conservation: {} (error: {:.4})", conserved, error);

    println!("\nTraining complete!");
    println!("Note: This example uses random initialization without actual training.");
    println!("For production, implement gradient descent optimization.");

    Ok(())
}
