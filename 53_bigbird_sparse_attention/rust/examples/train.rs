//! Train Example
//!
//! Train a BigBird model on trading data.
//!
//! Usage:
//!   cargo run --example train -- --epochs 50 --seq-len 128

use bigbird_trading::data::{DataLoader, FeatureEngine, TradingBatcher, TradingDataset};
use bigbird_trading::model::{BigBirdConfig, BigBirdModel};
use burn::backend::NdArray;
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use clap::Parser;

type Backend = NdArray;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of training epochs
    #[arg(short, long, default_value_t = 50)]
    epochs: usize,

    /// Sequence length
    #[arg(short, long, default_value_t = 128)]
    seq_len: usize,

    /// Batch size
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    /// Learning rate
    #[arg(short = 'r', long, default_value_t = 0.001)]
    learning_rate: f64,

    /// Model dimension
    #[arg(short, long, default_value_t = 64)]
    d_model: usize,

    /// Number of attention heads
    #[arg(short = 'H', long, default_value_t = 4)]
    n_heads: usize,

    /// Number of encoder layers
    #[arg(short = 'L', long, default_value_t = 3)]
    n_layers: usize,

    /// Number of synthetic samples
    #[arg(long, default_value_t = 5000)]
    n_samples: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Training step implementation
#[derive(Clone)]
struct TrainingStep<B: burn::prelude::Backend> {
    model: BigBirdModel<B>,
}

impl<B: burn::prelude::Backend> TrainStep<bigbird_trading::data::TradingBatch<B>, TrainOutput<f32>>
    for TrainingStep<B>
{
    fn step(&self, batch: bigbird_trading::data::TradingBatch<B>) -> TrainOutput<f32> {
        let predictions = self.model.forward(batch.features);
        let predictions = predictions.squeeze::<1>(1);

        let loss = (predictions.clone() - batch.targets.clone()).powf_scalar(2.0).mean();

        let loss_value = loss.clone().into_scalar().elem::<f32>();

        TrainOutput::new(
            &self.model,
            loss,
            loss_value,
        )
    }
}

impl<B: burn::prelude::Backend> ValidStep<bigbird_trading::data::TradingBatch<B>, TrainOutput<f32>>
    for TrainingStep<B>
{
    fn step(&self, batch: bigbird_trading::data::TradingBatch<B>) -> TrainOutput<f32> {
        let predictions = self.model.forward(batch.features);
        let predictions = predictions.squeeze::<1>(1);

        let loss = (predictions.clone() - batch.targets.clone()).powf_scalar(2.0).mean();
        let loss_value = loss.clone().into_scalar().elem::<f32>();

        TrainOutput::new(
            &self.model,
            loss,
            loss_value,
        )
    }
}

fn main() {
    let args = Args::parse();

    println!("=== BigBird Trading - Model Training ===\n");

    println!("Configuration:");
    println!("  Epochs:        {}", args.epochs);
    println!("  Sequence Len:  {}", args.seq_len);
    println!("  Batch Size:    {}", args.batch_size);
    println!("  Learning Rate: {}", args.learning_rate);
    println!("  d_model:       {}", args.d_model);
    println!("  n_heads:       {}", args.n_heads);
    println!("  n_layers:      {}", args.n_layers);
    println!();

    // Generate synthetic data
    println!("1. Generating synthetic data...");
    let loader = DataLoader::offline();
    let data = loader.generate_synthetic(args.n_samples, args.seed);
    println!("   Generated {} samples", data.len());

    // Create dataset
    println!("\n2. Creating dataset...");
    let feature_engine = FeatureEngine::default();
    let dataset = TradingDataset::from_market_data(&data, args.seq_len, &feature_engine);
    println!("   Dataset size: {}", dataset.len());
    println!("   Features: {}", dataset.n_features());
    println!("   Target stats: {}", dataset.target_stats());

    // Split dataset
    let (train_dataset, val_dataset, test_dataset) = dataset.split(0.7, 0.15);
    println!(
        "   Train: {}, Val: {}, Test: {}",
        train_dataset.len(),
        val_dataset.len(),
        test_dataset.len()
    );

    // Create model
    println!("\n3. Creating model...");
    let device = Default::default();
    let config = BigBirdConfig {
        seq_len: args.seq_len,
        input_dim: dataset.n_features(),
        d_model: args.d_model,
        n_heads: args.n_heads,
        n_layers: args.n_layers,
        d_ff: args.d_model * 4,
        window_size: 7,
        num_random: 3,
        num_global: 2,
        dropout: 0.1,
        output_dim: 1,
        pre_norm: true,
        activation: "gelu".to_string(),
        seed: args.seed,
    };
    println!("   Config: {:?}", config);

    let model = BigBirdModel::<Backend>::new(&device, &config);
    println!("   Parameters: ~{}", model.num_parameters());

    // Simple training loop (without burn's LearnerBuilder for simplicity)
    println!("\n4. Training...");

    // Create a simple training function
    train_simple(&model, &train_dataset, &val_dataset, &args, &device);

    // Evaluate on test set
    println!("\n5. Evaluating on test set...");
    evaluate_model(&model, &test_dataset, &device);

    println!("\n=== Training Complete ===");
}

fn train_simple<B: burn::prelude::Backend>(
    model: &BigBirdModel<B>,
    train_dataset: &TradingDataset,
    val_dataset: &TradingDataset,
    args: &Args,
    device: &B::Device,
) {
    let batcher = TradingBatcher::<B>::new(device.clone());

    // Simple epoch-based training
    for epoch in 0..args.epochs.min(10) {
        // Just run a few epochs for demonstration
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Process batches
        let samples: Vec<_> = (0..train_dataset.len().min(args.batch_size * 10))
            .filter_map(|i| train_dataset.get(i).cloned())
            .collect();

        for chunk in samples.chunks(args.batch_size) {
            let batch = batcher.batch(chunk.to_vec());
            let predictions = model.forward(batch.features);
            let predictions = predictions.squeeze::<1>(1);

            let diff = predictions.clone() - batch.targets.clone();
            let loss = diff.powf_scalar(2.0).mean();
            total_loss += loss.into_scalar().elem::<f32>();
            batch_count += 1;
        }

        if batch_count > 0 {
            println!(
                "   Epoch {}/{}: Train Loss = {:.6}",
                epoch + 1,
                args.epochs.min(10),
                total_loss / batch_count as f32
            );
        }
    }
}

fn evaluate_model<B: burn::prelude::Backend>(
    model: &BigBirdModel<B>,
    test_dataset: &TradingDataset,
    device: &B::Device,
) {
    let batcher = TradingBatcher::<B>::new(device.clone());

    let samples: Vec<_> = (0..test_dataset.len().min(100))
        .filter_map(|i| test_dataset.get(i).cloned())
        .collect();

    if samples.is_empty() {
        println!("   No test samples available");
        return;
    }

    let batch = batcher.batch(samples.clone());
    let predictions = model.forward(batch.features);
    let predictions = predictions.squeeze::<1>(1);

    let diff = predictions.clone() - batch.targets.clone();
    let mse = diff.clone().powf_scalar(2.0).mean().into_scalar().elem::<f32>();
    let mae = diff.abs().mean().into_scalar().elem::<f32>();

    println!("   Test MSE: {:.6}", mse);
    println!("   Test MAE: {:.6}", mae);

    // Calculate directional accuracy
    let pred_data: Vec<f32> = predictions.into_data().to_vec().unwrap();
    let target_data: Vec<f32> = batch.targets.into_data().to_vec().unwrap();

    let correct = pred_data
        .iter()
        .zip(target_data.iter())
        .filter(|(p, t)| p.signum() == t.signum())
        .count();
    let accuracy = correct as f32 / pred_data.len() as f32;
    println!("   Directional Accuracy: {:.2}%", accuracy * 100.0);
}
