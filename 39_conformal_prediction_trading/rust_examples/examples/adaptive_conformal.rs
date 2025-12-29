//! Example: Adaptive Conformal Inference for Time Series
//!
//! This example demonstrates how to use adaptive conformal prediction
//! that adjusts coverage levels dynamically based on recent performance.
//!
//! Run with: cargo run --example adaptive_conformal

use conformal_prediction_trading::{
    api::bybit::{BybitClient, Interval},
    conformal::{adaptive::AdaptiveConformalPredictor, model::LinearModel},
    data::{features::FeatureEngineering, processor::DataProcessor},
};

fn main() -> anyhow::Result<()> {
    println!("=== Adaptive Conformal Inference Example ===\n");

    // Fetch data from Bybit
    println!("Fetching data from Bybit...");
    let client = BybitClient::new();
    let klines = client.get_klines("ETHUSDT", Interval::Hour1, Some(500), None, None)?;
    println!("Received {} candles\n", klines.len());

    // Generate features
    let (features, _feature_names) = FeatureEngineering::generate_features(&klines);
    let targets = FeatureEngineering::create_returns(&klines, 1);

    // Skip warm-up period
    let valid_start = 30;
    let features = features.slice(ndarray::s![valid_start.., ..]).to_owned();
    let targets: Vec<f64> = targets[valid_start..].to_vec();

    // Split data
    let train_size = (features.nrows() as f64 * 0.5) as usize;
    let calib_size = (features.nrows() as f64 * 0.2) as usize;

    let x_train = features.slice(ndarray::s![..train_size, ..]).to_owned();
    let y_train: Vec<f64> = targets[..train_size].to_vec();

    let x_calib = features
        .slice(ndarray::s![train_size..train_size + calib_size, ..])
        .to_owned();
    let y_calib: Vec<f64> = targets[train_size..train_size + calib_size].to_vec();

    println!(
        "Data split: {} train, {} calib, {} test\n",
        train_size,
        calib_size,
        features.nrows() - train_size - calib_size
    );

    // Create adaptive conformal predictor
    let model = LinearModel::new(true);
    let mut acp = AdaptiveConformalPredictor::new(model, 0.9, 0.05);

    // Initial training
    println!("Initial training...");
    acp.fit(&x_train, &y_train, &x_calib, &y_calib);
    println!(
        "Initial alpha: {:.4} (coverage target: {:.1}%)\n",
        acp.current_alpha(),
        acp.target_coverage() * 100.0
    );

    // Online prediction and update
    println!("Running online prediction with adaptation...\n");
    println!("{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Step", "Actual%", "Pred%", "Width%", "Covered", "Alpha", "RecentCov%");

    let test_start = train_size + calib_size;
    let mut covered_count = 0;
    let mut total_count = 0;

    for i in test_start..(test_start + 50).min(features.nrows()) {
        let x: Vec<f64> = features.row(i).iter().copied().collect();
        let actual = targets[i];

        // Make prediction
        let interval = acp.predict_one(&x);

        // Check coverage
        let covered = interval.covers(actual);
        if covered {
            covered_count += 1;
        }
        total_count += 1;

        // Update (adapt alpha based on coverage)
        acp.update(actual, &interval);

        // Print every 5 steps
        if (i - test_start) % 5 == 0 {
            let recent_cov = acp.recent_coverage(20).unwrap_or(0.0);
            println!(
                "{:>5} {:>10.4} {:>10.4} {:>10.4} {:>10} {:>10.4} {:>10.1}",
                i - test_start,
                actual * 100.0,
                interval.prediction * 100.0,
                interval.width * 100.0,
                if covered { "Yes" } else { "No" },
                acp.current_alpha(),
                recent_cov * 100.0
            );
        }
    }

    println!("\n--- Summary ---");
    println!(
        "Final alpha: {:.4} (started at {:.4})",
        acp.current_alpha(),
        1.0 - acp.target_coverage()
    );
    println!(
        "Empirical coverage: {:.1}% ({}/{})",
        covered_count as f64 / total_count as f64 * 100.0,
        covered_count,
        total_count
    );
    println!("Target coverage: {:.1}%", acp.target_coverage() * 100.0);
    println!(
        "Final interval width: {:.4}%",
        acp.interval_width() * 100.0
    );

    // Compare with non-adaptive
    println!("\n--- Comparison with Non-Adaptive ---");
    println!("Adaptive CP adjusts alpha based on recent coverage:");
    println!("- If coverage > target: alpha increases (narrower intervals)");
    println!("- If coverage < target: alpha decreases (wider intervals)");
    println!("\nThis helps maintain coverage during distribution shifts.");

    println!("\n=== Done! ===");

    Ok(())
}
