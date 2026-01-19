//! Backtest Example
//!
//! Run backtesting on synthetic or real market data.
//!
//! Usage:
//!   cargo run --example backtest

use bigbird_trading::data::{DataLoader, FeatureEngine, TradingBatcher, TradingDataset};
use bigbird_trading::model::{BigBirdConfig, BigBirdModel};
use bigbird_trading::strategy::{BacktestConfig, Backtester, SignalConfig, SignalGenerator};
use burn::backend::NdArray;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use clap::Parser;

type Backend = NdArray;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Sequence length
    #[arg(short, long, default_value_t = 64)]
    seq_len: usize,

    /// Initial capital
    #[arg(short, long, default_value_t = 100000.0)]
    initial_capital: f64,

    /// Position size (fraction)
    #[arg(short, long, default_value_t = 0.1)]
    position_size: f64,

    /// Long threshold
    #[arg(long, default_value_t = 0.001)]
    long_threshold: f64,

    /// Short threshold
    #[arg(long, default_value_t = -0.001)]
    short_threshold: f64,

    /// Number of synthetic samples
    #[arg(long, default_value_t = 2000)]
    n_samples: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    println!("=== BigBird Trading - Backtesting ===\n");

    println!("Configuration:");
    println!("  Sequence Len:    {}", args.seq_len);
    println!("  Initial Capital: ${:.2}", args.initial_capital);
    println!("  Position Size:   {:.1}%", args.position_size * 100.0);
    println!("  Long Threshold:  {:.2}%", args.long_threshold * 100.0);
    println!("  Short Threshold: {:.2}%", args.short_threshold * 100.0);
    println!();

    // Generate synthetic data
    println!("1. Generating synthetic data...");
    let loader = DataLoader::offline();
    let data = loader.generate_synthetic(args.n_samples, args.seed);
    println!("   Generated {} price points", data.len());

    // Create dataset
    println!("\n2. Creating dataset...");
    let feature_engine = FeatureEngine::default();
    let dataset = TradingDataset::from_market_data(&data, args.seq_len, &feature_engine);
    println!("   Dataset size: {}", dataset.len());

    // Split into train and test (we only use test for backtest)
    let (_, _, test_dataset) = dataset.split(0.7, 0.15);
    println!("   Test set size: {}", test_dataset.len());

    // Create model (using random weights for demonstration)
    println!("\n3. Creating model...");
    let device = Default::default();
    let config = BigBirdConfig {
        seq_len: args.seq_len,
        input_dim: dataset.n_features(),
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 256,
        window_size: 7,
        num_random: 3,
        num_global: 2,
        dropout: 0.0, // No dropout for inference
        output_dim: 1,
        pre_norm: true,
        activation: "gelu".to_string(),
        seed: args.seed,
    };
    let model = BigBirdModel::<Backend>::new(&device, &config);
    println!("   Model created with {} parameters", model.num_parameters());

    // Generate predictions
    println!("\n4. Generating predictions...");
    let batcher = TradingBatcher::<Backend>::new(device.clone());

    let samples: Vec<_> = (0..test_dataset.len())
        .filter_map(|i| test_dataset.get(i).cloned())
        .collect();

    if samples.is_empty() {
        println!("   No test samples available!");
        return;
    }

    let batch = batcher.batch(samples);
    let predictions = model.forward(batch.features);
    let pred_data: Vec<f32> = predictions.squeeze::<1>(1).into_data().to_vec().unwrap();
    let predictions: Vec<f64> = pred_data.iter().map(|&x| x as f64).collect();

    println!("   Generated {} predictions", predictions.len());

    // Get corresponding prices (we need to align with the test set)
    // For synthetic data, we'll use the closes directly
    let test_start_idx = (args.n_samples as f64 * 0.85) as usize; // After train+val
    let prices: Vec<f64> = data.klines[test_start_idx..]
        .iter()
        .take(predictions.len())
        .map(|k| k.close)
        .collect();

    if prices.len() != predictions.len() {
        println!(
            "   Warning: Price/prediction mismatch ({} vs {})",
            prices.len(),
            predictions.len()
        );
    }

    let min_len = prices.len().min(predictions.len());
    let predictions = &predictions[..min_len];
    let prices = &prices[..min_len];

    // Print prediction statistics
    let pred_mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
    let pred_std = (predictions
        .iter()
        .map(|p| (p - pred_mean).powi(2))
        .sum::<f64>()
        / predictions.len() as f64)
        .sqrt();
    let positive_preds = predictions.iter().filter(|&&p| p > 0.0).count();

    println!("\n   Prediction Statistics:");
    println!("     Mean:     {:.6}", pred_mean);
    println!("     Std:      {:.6}", pred_std);
    println!("     Positive: {} ({:.1}%)", positive_preds, positive_preds as f64 / predictions.len() as f64 * 100.0);

    // Generate signals
    println!("\n5. Generating trading signals...");
    let signal_config = SignalConfig {
        long_threshold: args.long_threshold,
        short_threshold: args.short_threshold,
        min_confidence: 0.0,
        allow_short: true,
    };
    let signal_generator = SignalGenerator::new(signal_config);
    let signals = signal_generator.generate_batch(predictions, None);

    let signal_stats = signal_generator.signal_stats(&signals);
    println!("   {}", signal_stats);

    // Run backtest
    println!("\n6. Running backtest...");
    let backtest_config = BacktestConfig {
        initial_capital: args.initial_capital,
        position_size: args.position_size,
        transaction_cost: 0.001,
        slippage: 0.0005,
        risk_free_rate: 0.02,
        periods_per_year: 252 * 24,
        max_position: 1.0,
        stop_loss: Some(0.05),
        take_profit: Some(0.10),
    };

    let backtester = Backtester::new(backtest_config);
    let result = backtester.run(&signals, prices);

    // Print results
    println!("\n{}", result);

    // Print some trades
    if !result.trades.is_empty() {
        println!("\nSample Trades (first 5):");
        for (i, trade) in result.trades.iter().take(5).enumerate() {
            println!(
                "  {}. {} | Entry: {:.2} @ idx {} | Exit: {:.2} @ idx {} | PnL: {:.2} ({:.2}%)",
                i + 1,
                trade.signal_type,
                trade.entry_price,
                trade.entry_idx,
                trade.exit_price,
                trade.exit_idx,
                trade.pnl,
                trade.return_pct * 100.0
            );
        }
    }

    // Calculate buy-and-hold comparison
    if !prices.is_empty() {
        let buy_hold_return = (prices.last().unwrap() - prices[0]) / prices[0];
        println!("\nBuy & Hold Comparison:");
        println!("  Strategy Return:  {:.2}%", result.total_return * 100.0);
        println!("  Buy & Hold Return: {:.2}%", buy_hold_return * 100.0);
        println!(
            "  Outperformance:    {:.2}%",
            (result.total_return - buy_hold_return) * 100.0
        );
    }

    println!("\n=== Backtest Complete ===");
}
