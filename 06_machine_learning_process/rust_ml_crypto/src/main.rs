//! ML Crypto - Machine Learning for Cryptocurrency Trading
//!
//! This is the main entry point that demonstrates the ML workflow.
//!
//! # Examples
//!
//! Run examples with:
//! ```bash
//! cargo run --example fetch_data
//! cargo run --example knn_workflow
//! cargo run --example mutual_information
//! cargo run --example bias_variance
//! cargo run --example cross_validation
//! ```

use clap::{Parser, Subcommand};
use ml_crypto::api::BybitClient;
use ml_crypto::data::DataLoader;
use ml_crypto::features::FeatureEngine;
use ml_crypto::ml::{CrossValidator, KNNClassifier, Metrics};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "ml_crypto")]
#[command(about = "Machine Learning for Cryptocurrency Trading")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch market data from Bybit
    Fetch {
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Kline interval (e.g., 1h, 4h, 1d)
        #[arg(short, long, default_value = "1h")]
        interval: String,

        /// Number of candles to fetch
        #[arg(short, long, default_value = "500")]
        limit: usize,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run ML workflow demonstration
    Workflow {
        /// Path to candle data CSV
        #[arg(short, long)]
        data: Option<PathBuf>,

        /// Number of neighbors for KNN
        #[arg(short, long, default_value = "5")]
        k: usize,
    },

    /// Analyze feature importance using mutual information
    Features {
        /// Path to candle data CSV
        #[arg(short, long)]
        data: Option<PathBuf>,

        /// Number of bins for discretization
        #[arg(short, long, default_value = "20")]
        bins: usize,
    },

    /// Demonstrate bias-variance tradeoff
    BiasVariance {
        /// Maximum polynomial degree to test
        #[arg(short, long, default_value = "10")]
        max_degree: usize,

        /// Number of experiments
        #[arg(short, long, default_value = "50")]
        experiments: usize,
    },

    /// Cross-validation demonstration
    CrossVal {
        /// Path to candle data CSV
        #[arg(short, long)]
        data: Option<PathBuf>,

        /// Number of folds
        #[arg(short, long, default_value = "5")]
        folds: usize,

        /// Use time series split instead of k-fold
        #[arg(short, long)]
        time_series: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch {
            symbol,
            interval,
            limit,
            output,
        } => {
            info!("Fetching {} {} candles for {}", limit, interval, symbol);

            let client = BybitClient::new();
            let candles = client.get_klines(&symbol, &interval, limit).await?;

            info!("Fetched {} candles", candles.len());

            if let Some(path) = output {
                DataLoader::save_candles(&candles, &path)?;
                info!("Saved to {:?}", path);
            } else {
                // Print summary
                if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
                    println!("\n{} {} Data Summary", symbol, interval);
                    println!("================");
                    println!("First candle: {}", first.datetime());
                    println!("Last candle:  {}", last.datetime());
                    println!("Total candles: {}", candles.len());
                    println!(
                        "Price range: {:.2} - {:.2}",
                        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
                        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
                    );
                    println!(
                        "Avg volume: {:.2}",
                        candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64
                    );
                }
            }
        }

        Commands::Workflow { data, k } => {
            info!("Running ML workflow with k={}", k);

            let candles = if let Some(path) = data {
                DataLoader::load_candles(&path)?
            } else {
                info!("No data file provided, fetching from Bybit...");
                let client = BybitClient::new();
                client.get_klines("BTCUSDT", "1h", 500).await?
            };

            info!("Loaded {} candles", candles.len());

            // Generate features
            let engine = FeatureEngine::new();
            let dataset = engine
                .generate_features(&candles)
                .expect("Failed to generate features");

            info!(
                "Generated {} features for {} samples",
                dataset.n_features(),
                dataset.n_samples()
            );

            // Split data
            let (train, test) = dataset.train_test_split(0.2);
            info!("Train: {}, Test: {}", train.n_samples(), test.n_samples());

            // Train KNN
            let mut knn = KNNClassifier::new(k);
            knn.fit(&train.x, &train.y);

            // Predict
            let predictions = knn.predict(&test.x);

            // Evaluate
            let accuracy = Metrics::accuracy(&test.y, &predictions);
            let precision = Metrics::precision(&test.y, &predictions, 1.0);
            let recall = Metrics::recall(&test.y, &predictions, 1.0);
            let f1 = Metrics::f1_score(&test.y, &predictions, 1.0);

            println!("\nModel Performance");
            println!("=================");
            println!("Accuracy:  {:.4}", accuracy);
            println!("Precision: {:.4}", precision);
            println!("Recall:    {:.4}", recall);
            println!("F1 Score:  {:.4}", f1);
        }

        Commands::Features { data, bins } => {
            info!("Analyzing feature importance");

            let candles = if let Some(path) = data {
                DataLoader::load_candles(&path)?
            } else {
                info!("No data file provided, fetching from Bybit...");
                let client = BybitClient::new();
                client.get_klines("BTCUSDT", "1h", 500).await?
            };

            let engine = FeatureEngine::new();
            let dataset = engine
                .generate_features(&candles)
                .expect("Failed to generate features");

            // Calculate mutual information
            use ml_crypto::features::MutualInformation;
            let mi_scores = MutualInformation::feature_mutual_info(&dataset.x, &dataset.y, bins);

            // Sort by MI score
            let mut indexed_scores: Vec<(usize, f64)> =
                mi_scores.into_iter().enumerate().collect();
            indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("\nFeature Importance (Mutual Information)");
            println!("========================================");
            for (idx, score) in indexed_scores.iter().take(10) {
                println!("{:30} {:.4}", dataset.feature_names[*idx], score);
            }
        }

        Commands::BiasVariance {
            max_degree,
            experiments,
        } => {
            info!("Analyzing bias-variance tradeoff");

            use ml_crypto::ml::BiasVarianceAnalyzer;

            let true_fn = |x: f64| (x * std::f64::consts::PI).cos();
            let degrees: Vec<usize> = (1..=max_degree).collect();

            let results = BiasVarianceAnalyzer::analyze_bias_variance(
                true_fn,
                &degrees,
                experiments,
                30,
                50,
                0.3,
            );

            println!("\nBias-Variance Analysis");
            println!("======================");
            println!("{:>6} {:>12} {:>12} {:>12}", "Degree", "Bias²", "Variance", "Total");
            println!("{:-<48}", "");

            for (degree, bias_sq, variance, total) in &results {
                println!(
                    "{:>6} {:>12.4} {:>12.4} {:>12.4}",
                    degree, bias_sq, variance, total
                );
            }

            // Find optimal degree
            let (best_degree, _, _, min_error) = results
                .iter()
                .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
                .unwrap();

            println!("\nOptimal degree: {} (error: {:.4})", best_degree, min_error);
        }

        Commands::CrossVal {
            data,
            folds,
            time_series,
        } => {
            info!("Running cross-validation with {} folds", folds);

            let candles = if let Some(path) = data {
                DataLoader::load_candles(&path)?
            } else {
                info!("No data file provided, fetching from Bybit...");
                let client = BybitClient::new();
                client.get_klines("BTCUSDT", "1h", 500).await?
            };

            let engine = FeatureEngine::new();
            let mut dataset = engine
                .generate_features(&candles)
                .expect("Failed to generate features");

            dataset.standardize();

            let n_samples = dataset.n_samples();
            info!("Dataset: {} samples, {} features", n_samples, dataset.n_features());

            // Create splits
            let splits = if time_series {
                info!("Using time series split");
                CrossValidator::time_series_split(n_samples, folds, None)
            } else {
                info!("Using k-fold split");
                CrossValidator::k_fold(n_samples, folds, false)
            };

            // Evaluate with different k values
            println!("\nCross-Validation Results");
            println!("========================");
            println!("{:>4} {:>12} {:>12}", "K", "Mean Acc", "Std Acc");
            println!("{:-<32}", "");

            for k in [3, 5, 7, 9, 11] {
                let mut scores = Vec::new();

                for split in &splits {
                    let x_train = dataset.x.select(ndarray::Axis(0), &split.train_indices);
                    let y_train = ndarray::Array1::from_vec(
                        split.train_indices.iter().map(|&i| dataset.y[i]).collect(),
                    );
                    let x_test = dataset.x.select(ndarray::Axis(0), &split.test_indices);
                    let y_test = ndarray::Array1::from_vec(
                        split.test_indices.iter().map(|&i| dataset.y[i]).collect(),
                    );

                    let mut knn = KNNClassifier::new(k);
                    knn.fit(&x_train, &y_train);
                    let predictions = knn.predict(&x_test);

                    let accuracy = Metrics::accuracy(&y_test, &predictions);
                    scores.push(accuracy);
                }

                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                let variance =
                    scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
                let std = variance.sqrt();

                println!("{:>4} {:>12.4} {:>12.4}", k, mean, std);
            }
        }
    }

    Ok(())
}
