//! Example: Feature selection using Mutual Information
//!
//! Demonstrates how to use mutual information to:
//! 1. Measure feature importance
//! 2. Select the most predictive features
//! 3. Compare model performance with all vs selected features
//!
//! Run with: cargo run --example mutual_information

use ml_crypto::api::BybitClient;
use ml_crypto::features::{FeatureEngine, MutualInformation};
use ml_crypto::ml::{KNNClassifier, Metrics};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Mutual Information Example ===\n");

    // Fetch data
    println!("Fetching data from Bybit...");
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", "1h", 500).await?;
    println!("Fetched {} candles\n", candles.len());

    // Generate features
    println!("Generating features...");
    let engine = FeatureEngine::new();
    let dataset = engine
        .generate_features(&candles)
        .expect("Failed to generate features");

    println!("Generated {} features for {} samples\n",
             dataset.n_features(),
             dataset.n_samples());

    // Calculate mutual information
    println!("=== Calculating Mutual Information ===\n");
    let n_bins = 20;
    let mi_scores = MutualInformation::feature_mutual_info(&dataset.x, &dataset.y, n_bins);

    // Sort by MI score
    let mut indexed_scores: Vec<(usize, f64)> = mi_scores.iter().cloned().enumerate().collect();
    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 10 Features by Mutual Information:");
    println!("{:>4} {:>30} {:>12}", "Rank", "Feature", "MI Score");
    println!("{:-<50}", "");

    for (rank, (idx, score)) in indexed_scores.iter().take(10).enumerate() {
        println!("{:>4} {:>30} {:>12.4}",
                 rank + 1,
                 &dataset.feature_names[*idx],
                 score);
    }

    println!("\nBottom 5 Features:");
    for (idx, score) in indexed_scores.iter().rev().take(5) {
        println!("  {:>30} {:>12.4}",
                 &dataset.feature_names[*idx],
                 score);
    }

    // Calculate entropy of target
    println!("\n=== Information Theory Metrics ===");
    let target_entropy = MutualInformation::entropy_continuous(&dataset.y.to_vec(), n_bins);
    println!("Target entropy: {:.4} bits", target_entropy);

    // Normalized MI for top features
    println!("\nNormalized MI (top 5 features):");
    for (idx, _) in indexed_scores.iter().take(5) {
        let feature: Vec<f64> = dataset.x.column(*idx).to_vec();
        let nmi = MutualInformation::normalized_mutual_info(&feature, &dataset.y.to_vec(), n_bins);
        println!("  {:>30}: {:.4}", &dataset.feature_names[*idx], nmi);
    }

    // Feature selection experiment
    println!("\n=== Feature Selection Experiment ===\n");

    let (train, test) = dataset.train_test_split(0.2);
    let k = 5;

    // Train with all features
    let mut knn_all = KNNClassifier::new(k);
    knn_all.fit(&train.x, &train.y);
    let pred_all = knn_all.predict(&test.x);
    let acc_all = Metrics::accuracy(&test.y, &pred_all);

    println!("All {} features: Accuracy = {:.4}", dataset.n_features(), acc_all);

    // Train with top N features
    for n_features in [3, 5, 10, 15] {
        let top_indices: Vec<usize> = indexed_scores.iter().take(n_features).map(|(i, _)| *i).collect();

        let train_selected = train.select_features(&top_indices);
        let test_selected = test.select_features(&top_indices);

        let mut knn = KNNClassifier::new(k);
        knn.fit(&train_selected.x, &train_selected.y);
        let predictions = knn.predict(&test_selected.x);
        let accuracy = Metrics::accuracy(&test_selected.y, &predictions);

        let improvement = (accuracy - acc_all) / acc_all * 100.0;
        println!("Top {:>2} features: Accuracy = {:.4} ({:+.2}%)",
                 n_features, accuracy, improvement);
    }

    // Show feature correlations (simplified)
    println!("\n=== Feature Statistics ===");
    println!("\nFeature means:");
    for (idx, _) in indexed_scores.iter().take(5) {
        let feature: Vec<f64> = dataset.x.column(*idx).to_vec();
        let mean: f64 = feature.iter().sum::<f64>() / feature.len() as f64;
        let variance: f64 = feature.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / feature.len() as f64;
        let std = variance.sqrt();
        println!("  {:>30}: mean={:>8.4}, std={:>8.4}",
                 &dataset.feature_names[*idx], mean, std);
    }

    Ok(())
}
