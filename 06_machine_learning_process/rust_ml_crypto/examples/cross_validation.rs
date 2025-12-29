//! Example: Cross-Validation for Model Selection
//!
//! Demonstrates various cross-validation strategies:
//! 1. K-Fold CV (standard)
//! 2. Time Series Split (for temporal data)
//! 3. Purged K-Fold (for financial data with leakage prevention)
//!
//! Run with: cargo run --example cross_validation

use ml_crypto::api::BybitClient;
use ml_crypto::features::FeatureEngine;
use ml_crypto::ml::{CrossValidator, CVScores, KNNClassifier, Metrics};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Cross-Validation Example ===\n");

    // Fetch data
    println!("Fetching data from Bybit...");
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", "1h", 500).await?;
    println!("Fetched {} candles\n", candles.len());

    // Generate features
    let engine = FeatureEngine::new();
    let mut dataset = engine
        .generate_features(&candles)
        .expect("Failed to generate features");

    // Standardize features for better KNN performance
    dataset.standardize();

    let n_samples = dataset.n_samples();
    println!("Dataset: {} samples, {} features\n", n_samples, dataset.n_features());

    // ===================
    // 1. Standard K-Fold CV
    // ===================
    println!("=== 1. K-Fold Cross-Validation ===\n");

    let k_folds = 5;
    let splits = CrossValidator::k_fold(n_samples, k_folds, true);

    println!("Split sizes (shuffled):");
    for (i, split) in splits.iter().enumerate() {
        println!("  Fold {}: train={}, test={}",
                 i + 1, split.train_indices.len(), split.test_indices.len());
    }

    // Evaluate KNN with k=5
    let k = 5;
    let scores = evaluate_knn(&dataset, &splits, k);
    let cv_scores = CVScores::from_scores(scores);

    println!("\nK-Fold CV Results (k={}):", k);
    println!("  Fold scores: {:?}", cv_scores.scores.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    println!("  {}", cv_scores.summary());

    // ===================
    // 2. Time Series Split
    // ===================
    println!("\n=== 2. Time Series Split ===\n");

    let ts_splits = CrossValidator::time_series_split(n_samples, 5, None);

    println!("Split sizes (expanding window):");
    for (i, split) in ts_splits.iter().enumerate() {
        println!("  Fold {}: train={} ({}..{}), test={} ({}..{})",
                 i + 1,
                 split.train_indices.len(),
                 split.train_indices.first().unwrap_or(&0),
                 split.train_indices.last().unwrap_or(&0),
                 split.test_indices.len(),
                 split.test_indices.first().unwrap_or(&0),
                 split.test_indices.last().unwrap_or(&0));
    }

    let ts_scores = evaluate_knn(&dataset, &ts_splits, k);
    let ts_cv_scores = CVScores::from_scores(ts_scores);

    println!("\nTime Series CV Results (k={}):", k);
    println!("  {}", ts_cv_scores.summary());

    // ===================
    // 3. Purged K-Fold CV
    // ===================
    println!("\n=== 3. Purged K-Fold (Financial Data) ===\n");

    let purge_gap = 5;  // Remove 5 samples before test
    let embargo_gap = 3; // Remove 3 samples after test

    let purged_splits = CrossValidator::purged_k_fold(n_samples, 5, purge_gap, embargo_gap);

    println!("Purged split sizes (purge={}, embargo={}):", purge_gap, embargo_gap);
    for (i, split) in purged_splits.iter().enumerate() {
        println!("  Fold {}: train={}, test={}",
                 i + 1, split.train_indices.len(), split.test_indices.len());
    }

    let purged_scores = evaluate_knn(&dataset, &purged_splits, k);
    let purged_cv_scores = CVScores::from_scores(purged_scores);

    println!("\nPurged K-Fold CV Results (k={}):", k);
    println!("  {}", purged_cv_scores.summary());

    // ===================
    // 4. Comparison
    // ===================
    println!("\n=== CV Method Comparison ===\n");

    println!("{:>20} {:>12} {:>12} {:>12}",
             "Method", "Mean Acc", "Std", "CI Width");
    println!("{:-<58}", "");

    println!("{:>20} {:>12.4} {:>12.4} {:>12.4}",
             "K-Fold",
             cv_scores.mean,
             cv_scores.std,
             cv_scores.std * 1.96 * 2.0);

    println!("{:>20} {:>12.4} {:>12.4} {:>12.4}",
             "Time Series",
             ts_cv_scores.mean,
             ts_cv_scores.std,
             ts_cv_scores.std * 1.96 * 2.0);

    println!("{:>20} {:>12.4} {:>12.4} {:>12.4}",
             "Purged K-Fold",
             purged_cv_scores.mean,
             purged_cv_scores.std,
             purged_cv_scores.std * 1.96 * 2.0);

    // ===================
    // 5. Hyperparameter Tuning with CV
    // ===================
    println!("\n=== Hyperparameter Tuning with CV ===\n");

    println!("Testing different k values for KNN:\n");
    println!("{:>4} {:>12} {:>12} {:>12}",
             "K", "Mean", "Std", "95% CI");
    println!("{:-<44}", "");

    let mut best_k = 1;
    let mut best_score = 0.0;

    for k_neighbors in [1, 3, 5, 7, 9, 11, 15, 21, 31] {
        let scores = evaluate_knn(&dataset, &ts_splits, k_neighbors);
        let cv = CVScores::from_scores(scores);

        let ci_lower = cv.mean - 1.96 * cv.std;
        let ci_upper = cv.mean + 1.96 * cv.std;

        println!("{:>4} {:>12.4} {:>12.4} [{:.4}, {:.4}]",
                 k_neighbors, cv.mean, cv.std, ci_lower, ci_upper);

        if cv.mean > best_score {
            best_score = cv.mean;
            best_k = k_neighbors;
        }
    }

    println!("\nBest k: {} (accuracy: {:.4})", best_k, best_score);

    // ===================
    // Summary
    // ===================
    println!("\n=== Summary ===");
    println!();
    println!("Cross-validation is essential for model selection because:");
    println!("  1. It provides unbiased estimates of generalization error");
    println!("  2. It helps detect overfitting before deployment");
    println!("  3. It enables fair comparison between models");
    println!();
    println!("For financial data:");
    println!("  - K-Fold can leak future information (data leakage)");
    println!("  - Time Series Split respects temporal ordering");
    println!("  - Purged K-Fold prevents label overlap issues");
    println!();
    println!("Best practice: Use Time Series or Purged CV for trading strategies!");

    Ok(())
}

/// Helper function to evaluate KNN with given CV splits
fn evaluate_knn(
    dataset: &ml_crypto::data::Dataset,
    splits: &[ml_crypto::ml::cross_validation::CVSplit],
    k: usize,
) -> Vec<f64> {
    splits
        .iter()
        .map(|split| {
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

            Metrics::accuracy(&y_test, &predictions)
        })
        .collect()
}
