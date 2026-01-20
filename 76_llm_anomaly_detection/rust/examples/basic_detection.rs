//! Basic anomaly detection example.
//!
//! Demonstrates loading data from Bybit and running anomaly detection.

use llm_anomaly_detection::{
    data_loader::{BybitLoader, FeatureCalculator},
    detector::{AnomalyDetector, StatisticalDetector},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=".repeat(60));
    println!("LLM Anomaly Detection - Basic Example");
    println!("=".repeat(60));

    // Load data from Bybit
    println!("\nLoading BTCUSDT data from Bybit...");
    let loader = BybitLoader::new();
    let candles = loader.get_klines("BTCUSDT", "1h", 500).await?;
    println!("Loaded {} candles", candles.len());

    // Calculate features
    println!("Calculating features...");
    let calculator = FeatureCalculator::new(20);
    let features = calculator.calculate_features(&candles);
    println!("Calculated {} feature vectors", features.len());

    // Create and train detector
    println!("\nTraining anomaly detector...");
    let mut detector = StatisticalDetector::new(2.5);

    // Train on first 400 observations
    let train_size = 400.min(features.len());
    detector.fit(&features[..train_size])?;
    println!("Trained on {} observations", train_size);

    // Detect anomalies in remaining data
    println!("\nAnalyzing last {} observations...", features.len() - train_size);
    let mut anomalies_found = 0;

    for i in train_size..features.len() {
        let result = detector.detect(&features[i])?;

        if result.is_anomaly {
            anomalies_found += 1;
            let candle = &candles[i];

            println!("\nANOMALY at {}:", candle.timestamp);
            println!("  Type: {}", result.anomaly_type);
            println!("  Score: {:.3}", result.score);
            println!("  Confidence: {:.3}", result.confidence);
            println!("  Explanation: {}", result.explanation);
            println!("  BTC Price: ${:.2}", candle.close);

            // Show details if available
            if let Some(zscore) = result.details.get("max_zscore") {
                println!("  Max Z-score: {:.2}", zscore);
            }
        }
    }

    println!("\n{}", "-".repeat(60));
    println!(
        "Total anomalies found: {} / {}",
        anomalies_found,
        features.len() - train_size
    );
    println!(
        "Anomaly rate: {:.1}%",
        anomalies_found as f64 / (features.len() - train_size) as f64 * 100.0
    );
    println!("{}", "=".repeat(60));

    Ok(())
}
