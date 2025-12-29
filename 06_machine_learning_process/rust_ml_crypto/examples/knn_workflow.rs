//! Example: Complete ML workflow with KNN classifier
//!
//! Demonstrates the full machine learning workflow:
//! 1. Fetch data from Bybit
//! 2. Generate features
//! 3. Split into train/test
//! 4. Train KNN classifier
//! 5. Evaluate performance
//!
//! Run with: cargo run --example knn_workflow

use ml_crypto::api::BybitClient;
use ml_crypto::features::FeatureEngine;
use ml_crypto::ml::{KNNClassifier, Metrics};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== KNN Workflow Example ===\n");

    // Step 1: Fetch data
    println!("Step 1: Fetching data from Bybit...");
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", "1h", 500).await?;
    println!("  Fetched {} candles\n", candles.len());

    // Step 2: Feature engineering
    println!("Step 2: Generating features...");
    let engine = FeatureEngine::new();
    let dataset = engine
        .generate_features(&candles)
        .expect("Failed to generate features");

    println!("  Generated {} features for {} samples",
             dataset.n_features(),
             dataset.n_samples());
    println!("  Features: {:?}\n", &dataset.feature_names[..5]);

    // Step 3: Train-test split
    println!("Step 3: Splitting data...");
    let (train, test) = dataset.train_test_split(0.2);
    println!("  Training samples: {}", train.n_samples());
    println!("  Testing samples: {}\n", test.n_samples());

    // Step 4: Train and evaluate with different k values
    println!("Step 4: Training KNN classifiers...\n");

    println!("{:>4} {:>10} {:>10} {:>10} {:>10}",
             "K", "Accuracy", "Precision", "Recall", "F1");
    println!("{:-<48}", "");

    let mut best_k = 1;
    let mut best_accuracy = 0.0;

    for k in [1, 3, 5, 7, 9, 11, 15, 21] {
        let mut knn = KNNClassifier::new(k);
        knn.fit(&train.x, &train.y);

        let predictions = knn.predict(&test.x);

        let accuracy = Metrics::accuracy(&test.y, &predictions);
        let precision = Metrics::precision(&test.y, &predictions, 1.0);
        let recall = Metrics::recall(&test.y, &predictions, 1.0);
        let f1 = Metrics::f1_score(&test.y, &predictions, 1.0);

        println!("{:>4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                 k, accuracy, precision, recall, f1);

        if accuracy > best_accuracy {
            best_accuracy = accuracy;
            best_k = k;
        }
    }

    println!("\nBest k={} with accuracy={:.4}", best_k, best_accuracy);

    // Step 5: Detailed analysis with best k
    println!("\n=== Detailed Analysis with k={} ===\n", best_k);

    let mut knn = KNNClassifier::new(best_k);
    knn.fit(&train.x, &train.y);
    let predictions = knn.predict(&test.x);

    // Classification report
    let report = Metrics::classification_report(&test.y, &predictions);
    println!("Classification Report:");
    println!("{:>8} {:>10} {:>10} {:>10} {:>10}",
             "Class", "Precision", "Recall", "F1", "Support");
    println!("{:-<52}", "");

    for (class, precision, recall, f1, support) in report {
        println!("{:>8.0} {:>10.4} {:>10.4} {:>10.4} {:>10}",
                 class, precision, recall, f1, support);
    }

    // Confusion matrix
    println!("\nConfusion Matrix:");
    let cm = Metrics::confusion_matrix(&test.y, &predictions);
    let mut classes: Vec<i64> = cm.keys().map(|(t, _)| *t).collect();
    classes.sort();
    classes.dedup();

    print!("      ");
    for &p in &classes {
        print!("{:>8}", format!("P={}", p));
    }
    println!();

    for &t in &classes {
        print!("A={:<3}", t);
        for &p in &classes {
            let count = cm.get(&(t, p)).unwrap_or(&0);
            print!("{:>8}", count);
        }
        println!();
    }

    // Trading metrics
    println!("\n=== Trading Metrics ===");
    let hit_ratio = Metrics::hit_ratio(&predictions);
    println!("Hit ratio (positive predictions): {:.4}", hit_ratio);

    Ok(())
}
