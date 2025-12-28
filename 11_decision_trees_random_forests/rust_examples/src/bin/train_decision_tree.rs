//! Train a Decision Tree model on crypto data
//!
//! Usage: cargo run --bin train_decision_tree -- --symbol BTCUSDT --days 90

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use crypto_ml::api::{BybitClient, Interval, Symbol};
use crypto_ml::features::FeatureEngine;
use crypto_ml::models::{DecisionTree, TaskType, TreeConfig};
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train Decision Tree on crypto data")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of days of data
    #[arg(short, long, default_value = "90")]
    days: i64,

    /// Max tree depth
    #[arg(long, default_value = "5")]
    max_depth: usize,

    /// Min samples to split
    #[arg(long, default_value = "10")]
    min_samples_split: usize,

    /// Classification mode (vs regression)
    #[arg(short, long)]
    classification: bool,

    /// Test set ratio
    #[arg(long, default_value = "0.2")]
    test_ratio: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("crypto_ml=info")
        .init();

    let args = Args::parse();

    println!("===========================================");
    println!("  Decision Tree Training - Crypto ML");
    println!("===========================================\n");

    // Fetch data
    info!("Fetching {} data for {} days...", args.symbol, args.days);

    let client = BybitClient::new();
    let symbol = Symbol::new(&args.symbol);

    let end = Utc::now();
    let start = end - Duration::days(args.days);

    let candles = client
        .get_historical_klines(&symbol, Interval::Hour1, start, end)
        .await?;

    println!("Fetched {} candles\n", candles.len());

    // Generate features
    info!("Generating features...");
    let engine = FeatureEngine::new().with_horizon(1);
    let mut dataset = engine.generate(&candles);

    println!("Dataset: {} samples, {} features", dataset.n_samples(), dataset.n_features());

    // Convert to classification if needed
    if args.classification {
        dataset.to_binary_classification();
        println!("Converted to binary classification (up/down)");
    }

    // Split data
    let split = dataset.train_test_split(args.test_ratio);
    println!(
        "\nTrain set: {} samples",
        split.train.n_samples()
    );
    println!("Test set:  {} samples\n", split.test.n_samples());

    // Configure and train tree
    let config = TreeConfig {
        max_depth: args.max_depth,
        min_samples_split: args.min_samples_split,
        min_samples_leaf: 5,
        max_features: None,
        seed: 42,
        task: if args.classification {
            TaskType::Classification
        } else {
            TaskType::Regression
        },
    };

    info!("Training decision tree...");
    let mut tree = DecisionTree::new(config);
    tree.fit(&split.train);

    // Evaluate
    println!("\n=== Model Evaluation ===\n");

    if args.classification {
        let train_acc = tree.accuracy(&split.train);
        let test_acc = tree.accuracy(&split.test);

        println!("Training Accuracy: {:.2}%", train_acc * 100.0);
        println!("Test Accuracy:     {:.2}%", test_acc * 100.0);
    } else {
        let train_r2 = tree.r2_score(&split.train);
        let test_r2 = tree.r2_score(&split.test);
        let train_mse = tree.mse_score(&split.train);
        let test_mse = tree.mse_score(&split.test);

        println!("Training R²:  {:.4}", train_r2);
        println!("Test R²:      {:.4}", test_r2);
        println!("Training MSE: {:.6}", train_mse);
        println!("Test MSE:     {:.6}", test_mse);
    }

    // Feature importance
    println!("\n=== Top 10 Feature Importances ===\n");

    let mut importance = tree.feature_importance_map();
    importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (name, imp)) in importance.iter().take(10).enumerate() {
        let bar = "█".repeat((imp * 50.0) as usize);
        println!("{:2}. {:25} {:.4} {}", i + 1, name, imp, bar);
    }

    // Print tree structure (limited depth)
    println!("\n=== Tree Structure (first 3 levels) ===\n");
    tree.print_tree();

    // Sample predictions
    println!("\n=== Sample Predictions (last 10) ===\n");

    let predictions = tree.predict(&split.test);
    let n = predictions.len().min(10);

    println!("{:>12} {:>12} {:>12}",
        "Actual", "Predicted", "Error");
    println!("{}", "-".repeat(40));

    for i in (predictions.len() - n)..predictions.len() {
        let actual = split.test.labels[i];
        let pred = predictions[i];
        let error = actual - pred;

        if args.classification {
            println!(
                "{:>12} {:>12} {:>12}",
                if actual > 0.5 { "UP" } else { "DOWN" },
                if pred > 0.5 { "UP" } else { "DOWN" },
                if (actual > 0.5) == (pred > 0.5) { "✓" } else { "✗" }
            );
        } else {
            println!(
                "{:>12.4}% {:>12.4}% {:>12.4}%",
                actual * 100.0,
                pred * 100.0,
                error * 100.0
            );
        }
    }

    println!("\nTraining complete!");

    Ok(())
}
