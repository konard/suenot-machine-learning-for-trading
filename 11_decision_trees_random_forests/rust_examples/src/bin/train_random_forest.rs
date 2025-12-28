//! Train a Random Forest model on crypto data
//!
//! Usage: cargo run --bin train_random_forest -- --symbol BTCUSDT --days 180 --trees 100

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use crypto_ml::api::{BybitClient, Interval, Symbol};
use crypto_ml::features::FeatureEngine;
use crypto_ml::models::{RandomForest, TaskType};
use crypto_ml::models::random_forest::ForestConfig;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train Random Forest on crypto data")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of days of data
    #[arg(short, long, default_value = "180")]
    days: i64,

    /// Number of trees
    #[arg(short, long, default_value = "50")]
    trees: usize,

    /// Max tree depth
    #[arg(long, default_value = "8")]
    max_depth: usize,

    /// Classification mode
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
    println!("  Random Forest Training - Crypto ML");
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

    println!(
        "Dataset: {} samples, {} features",
        dataset.n_samples(),
        dataset.n_features()
    );

    // Convert to classification if needed
    if args.classification {
        dataset.to_binary_classification();
        println!("Converted to binary classification (up/down)");
    }

    // Split data
    let split = dataset.train_test_split(args.test_ratio);
    println!("\nTrain set: {} samples", split.train.n_samples());
    println!("Test set:  {} samples\n", split.test.n_samples());

    // Configure and train forest
    let config = ForestConfig {
        n_trees: args.trees,
        max_depth: args.max_depth,
        min_samples_split: 10,
        min_samples_leaf: 5,
        max_features: None, // Will use sqrt(n_features) for classification
        bootstrap: true,
        seed: 42,
        task: if args.classification {
            TaskType::Classification
        } else {
            TaskType::Regression
        },
        oob_score: true,
    };

    println!("Training Random Forest with {} trees...", args.trees);
    println!("(This may take a moment)\n");

    let start_time = std::time::Instant::now();
    let mut forest = RandomForest::new(config);
    forest.fit(&split.train);
    let training_time = start_time.elapsed();

    println!("Training completed in {:.2}s\n", training_time.as_secs_f64());

    // Evaluate
    println!("=== Model Evaluation ===\n");

    if args.classification {
        let train_acc = forest.accuracy(&split.train);
        let test_acc = forest.accuracy(&split.test);

        println!("Training Accuracy: {:.2}%", train_acc * 100.0);
        println!("Test Accuracy:     {:.2}%", test_acc * 100.0);

        if let Some(oob) = forest.oob_score() {
            println!("OOB Score:         {:.2}%", oob * 100.0);
        }
    } else {
        let train_r2 = forest.r2_score(&split.train);
        let test_r2 = forest.r2_score(&split.test);
        let train_mse = forest.mse(&split.train);
        let test_mse = forest.mse(&split.test);

        println!("Training R²:  {:.4}", train_r2);
        println!("Test R²:      {:.4}", test_r2);
        println!("Training MSE: {:.6}", train_mse);
        println!("Test MSE:     {:.6}", test_mse);

        if let Some(oob) = forest.oob_score() {
            println!("OOB R²:       {:.4}", oob);
        }
    }

    // Feature importance
    println!("\n=== Feature Importance Ranking ===\n");

    let ranking = forest.feature_importance_ranking();

    for (i, (name, imp)) in ranking.iter().take(15).enumerate() {
        let bar = "█".repeat((imp * 40.0) as usize);
        println!("{:2}. {:25} {:.4} {}", i + 1, name, imp, bar);
    }

    // Prediction distribution
    println!("\n=== Prediction Distribution ===\n");

    let predictions = forest.predict(&split.test);

    if args.classification {
        let up_preds = predictions.iter().filter(|&&p| p > 0.5).count();
        let down_preds = predictions.len() - up_preds;

        println!("UP predictions:   {} ({:.1}%)",
            up_preds,
            up_preds as f64 / predictions.len() as f64 * 100.0);
        println!("DOWN predictions: {} ({:.1}%)",
            down_preds,
            down_preds as f64 / predictions.len() as f64 * 100.0);
    } else {
        let mean_pred = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let std_pred = (predictions.iter().map(|p| (p - mean_pred).powi(2)).sum::<f64>()
            / predictions.len() as f64)
            .sqrt();
        let min_pred = predictions.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_pred = predictions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("Mean prediction: {:.4}%", mean_pred * 100.0);
        println!("Std deviation:   {:.4}%", std_pred * 100.0);
        println!("Min prediction:  {:.4}%", min_pred * 100.0);
        println!("Max prediction:  {:.4}%", max_pred * 100.0);
    }

    // Model summary
    println!();
    forest.summary();

    // Compare single tree vs forest
    println!("\n=== Single Tree vs Random Forest ===\n");

    use crypto_ml::models::{DecisionTree, TreeConfig};

    let tree_config = TreeConfig {
        max_depth: args.max_depth,
        min_samples_split: 10,
        min_samples_leaf: 5,
        max_features: None,
        seed: 42,
        task: if args.classification {
            TaskType::Classification
        } else {
            TaskType::Regression
        },
    };

    let mut single_tree = DecisionTree::new(tree_config);
    single_tree.fit(&split.train);

    if args.classification {
        let tree_acc = single_tree.accuracy(&split.test);
        let forest_acc = forest.accuracy(&split.test);

        println!("Single Tree Test Accuracy:  {:.2}%", tree_acc * 100.0);
        println!("Random Forest Test Accuracy: {:.2}%", forest_acc * 100.0);
        println!(
            "Improvement: {:.2}%",
            (forest_acc - tree_acc) * 100.0
        );
    } else {
        let tree_r2 = single_tree.r2_score(&split.test);
        let forest_r2 = forest.r2_score(&split.test);

        println!("Single Tree Test R²:   {:.4}", tree_r2);
        println!("Random Forest Test R²: {:.4}", forest_r2);
        println!("Improvement: {:.4}", forest_r2 - tree_r2);
    }

    println!("\nTraining complete!");

    Ok(())
}
