//! Example: Split Conformal Prediction for Return Forecasting
//!
//! This example demonstrates how to use split conformal prediction
//! to create calibrated prediction intervals for cryptocurrency returns.
//!
//! Run with: cargo run --example split_conformal

use conformal_prediction_trading::{
    api::bybit::{BybitClient, Interval},
    conformal::{model::LinearModel, split::SplitConformalPredictor},
    data::{features::FeatureEngineering, processor::DataProcessor},
    metrics::coverage::CoverageMetrics,
};

fn main() -> anyhow::Result<()> {
    println!("=== Split Conformal Prediction Example ===\n");

    // Fetch data from Bybit
    println!("Fetching data from Bybit...");
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", Interval::Hour4, Some(500), None, None)?;
    println!("Received {} candles\n", klines.len());

    // Generate features
    println!("Generating features...");
    let (features, feature_names) = FeatureEngineering::generate_features(&klines);
    println!("Created {} features: {:?}\n", feature_names.len(), &feature_names[..5]);

    // Create targets (forward returns)
    let targets = FeatureEngineering::create_returns(&klines, 1);

    // Remove first rows that may have NaN values
    let valid_start = 30; // Skip warm-up period for indicators
    let features = features.slice(ndarray::s![valid_start.., ..]).to_owned();
    let targets: Vec<f64> = targets[valid_start..].to_vec();

    println!("Data shape: {} samples x {} features", features.nrows(), features.ncols());

    // Split data: 60% train, 20% calibration, 20% test
    let ((x_train, y_train), (x_calib, y_calib), (x_test, y_test)) =
        DataProcessor::train_calib_test_split(&features, &targets, 0.6, 0.2);

    println!(
        "Split: {} train, {} calib, {} test\n",
        x_train.nrows(),
        x_calib.nrows(),
        x_test.nrows()
    );

    // Create and train conformal predictor
    println!("Training conformal predictor (90% coverage target)...");
    let model = LinearModel::new(true);
    let mut cp = SplitConformalPredictor::new(model, 0.1); // 90% coverage
    cp.fit(&x_train, &y_train, &x_calib, &y_calib);

    println!("Calibration complete!");
    println!(
        "Interval quantile (q_hat): {:.6}",
        cp.quantile().unwrap_or(0.0)
    );
    println!("Interval width: {:.4}%\n", cp.interval_width() * 100.0);

    // Make predictions on test set
    println!("Making predictions on test set...");
    let intervals = cp.predict(&x_test);

    // Calculate coverage metrics
    let metrics = CoverageMetrics::calculate(&intervals, &y_test, 0.1);
    println!("\n{}", metrics.report());

    // Show some example predictions
    println!("\n--- Example Predictions (first 10) ---");
    println!("{:>10} {:>10} {:>10} {:>10} {:>10}", "Actual", "Lower", "Pred", "Upper", "Covered");
    for i in 0..10.min(intervals.len()) {
        let interval = &intervals[i];
        let actual = y_test[i];
        let covered = if interval.covers(actual) { "Yes" } else { "No" };
        println!(
            "{:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10}",
            actual * 100.0,
            interval.lower * 100.0,
            interval.prediction * 100.0,
            interval.upper * 100.0,
            covered
        );
    }

    // Analyze coverage by interval width
    println!("\n--- Coverage by Interval Width Quantile ---");
    let coverage_by_width =
        CoverageMetrics::coverage_by_width_quantile(&intervals, &y_test, 4);
    println!("{:>15} {:>15} {:>10}", "Avg Width (%)", "Coverage (%)", "N");
    for (width, cov, n) in coverage_by_width {
        println!("{:>15.4} {:>15.1} {:>10}", width * 100.0, cov * 100.0, n);
    }

    println!("\n=== Done! ===");

    Ok(())
}
