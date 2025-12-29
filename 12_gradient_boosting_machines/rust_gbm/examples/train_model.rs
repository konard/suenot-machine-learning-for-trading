//! Example: Training a Gradient Boosting Model
//!
//! Run with: cargo run --example train_model

use anyhow::Result;
use rust_gbm::{
    data::{BybitClient, Interval},
    features::FeatureEngineer,
    models::{GbmParams, GbmRegressor},
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("GBM Training Example");
    println!("{}", "=".repeat(40));

    // 1. Fetch data
    println!("\n📥 Fetching data from Bybit...");
    let client = BybitClient::new();
    let candles = client
        .get_klines("BTCUSDT", Interval::Hour1, Some(1000), None, None)
        .await?;
    println!("   Fetched {} candles", candles.len());

    // 2. Engineer features
    println!("\n🔧 Engineering features...");
    let engineer = FeatureEngineer::new();
    let dataset = engineer.build_clean_features(&candles);
    println!("   Created {} samples with {} features",
        dataset.len(), dataset.num_features());

    // 3. Split data
    let (train, test) = dataset.train_test_split(0.8);
    println!("\n📊 Data Split:");
    println!("   Train: {} samples", train.len());
    println!("   Test:  {} samples", test.len());

    // 4. Train model with different configurations
    let configs = vec![
        ("Conservative", GbmParams {
            n_estimators: 50,
            max_depth: 3,
            learning_rate: 0.05,
            ..Default::default()
        }),
        ("Balanced", GbmParams {
            n_estimators: 100,
            max_depth: 4,
            learning_rate: 0.1,
            ..Default::default()
        }),
        ("Aggressive", GbmParams {
            n_estimators: 200,
            max_depth: 6,
            learning_rate: 0.15,
            ..Default::default()
        }),
    ];

    println!("\n🤖 Training models...\n");
    println!("{:<15} {:>10} {:>10} {:>12}", "Config", "RMSE", "R²", "Dir. Acc.");
    println!("{}", "-".repeat(50));

    for (name, params) in configs {
        let mut model = GbmRegressor::with_params(params);
        model.fit(&train)?;

        let metrics = model.evaluate(&test)?;

        println!("{:<15} {:>10.4} {:>10.4} {:>11.2}%",
            name,
            metrics.rmse.unwrap_or(0.0),
            metrics.r2.unwrap_or(0.0),
            metrics.directional_accuracy.unwrap_or(0.0));
    }

    Ok(())
}
