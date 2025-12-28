//! Example: Linear Regression for Cryptocurrency Return Prediction
//!
//! This example demonstrates how to use linear regression to predict
//! cryptocurrency returns using technical indicators as features.

use chrono::{Duration, Utc};
use linear_models_crypto::{
    api::bybit::{BybitClient, Interval},
    data::{
        features::FeatureEngineering,
        processor::{train_test_split, DataProcessor},
    },
    metrics::regression::RegressionMetrics,
    models::linear::{LinearRegression, LinearRegressionGD},
};
use ndarray::Array1;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("================================================");
    println!("  Linear Regression for Crypto Price Prediction");
    println!("================================================\n");

    // Fetch data from Bybit
    println!("Fetching BTCUSDT hourly data...");
    let client = BybitClient::new();

    let end_time = Utc::now().timestamp_millis();
    let start_time = (Utc::now() - Duration::days(30)).timestamp_millis();

    let klines = client.get_klines_history("BTCUSDT", Interval::Hour1, start_time, end_time)?;
    println!("Fetched {} candles\n", klines.len());

    // Generate features
    println!("Generating technical features...");
    let (features, feature_names) = FeatureEngineering::generate_features(&klines);
    println!("Generated {} features:", feature_names.len());
    for (i, name) in feature_names.iter().enumerate() {
        println!("  {:2}. {}", i + 1, name);
    }

    // Create target (1-period forward returns)
    println!("\nCreating target variable (1-hour forward returns)...");
    let target = FeatureEngineering::create_target(&klines, 1);

    // Clean data (remove NaN values)
    let (x_clean, y_clean) = DataProcessor::dropna(&features, &target);
    println!(
        "After removing NaN: {} samples, {} features",
        x_clean.nrows(),
        x_clean.ncols()
    );

    // Standardize features
    let mut processor = DataProcessor::new();
    processor.fit_standard_scaler(&x_clean);
    let x_scaled = processor.transform_standard(&x_clean);

    // Split into train/test
    let test_ratio = 0.2;
    let (x_train, x_test, y_train, y_test) = train_test_split(&x_scaled, &y_clean, test_ratio);
    println!(
        "\nTrain set: {} samples, Test set: {} samples",
        x_train.nrows(),
        x_test.nrows()
    );

    // =====================
    // OLS Linear Regression
    // =====================
    println!("\n" + "=".repeat(50).as_str());
    println!("Ordinary Least Squares (OLS) Regression");
    println!("{}", "=".repeat(50));

    let mut ols_model = LinearRegression::new(true).with_feature_names(feature_names.clone());

    ols_model.fit(&x_train, &y_train)?;

    // Print model summary
    println!("\n{}", ols_model.summary());

    // Make predictions
    let train_pred = ols_model.predict(&x_train)?;
    let test_pred = ols_model.predict(&x_test)?;

    // Calculate metrics
    let train_metrics = RegressionMetrics::calculate(&y_train, &train_pred);
    let test_metrics = RegressionMetrics::calculate(&y_test, &test_pred);

    println!("Training Performance:");
    println!("  R²:   {:.6}", train_metrics.r2);
    println!("  RMSE: {:.6}", train_metrics.rmse);
    println!("  IC:   {:.6}", train_metrics.ic);

    println!("\nTest Performance:");
    println!("  R²:   {:.6}", test_metrics.r2);
    println!("  RMSE: {:.6}", test_metrics.rmse);
    println!("  IC:   {:.6}", test_metrics.ic);

    // Hit rate (direction accuracy)
    let hit_rate = RegressionMetrics::hit_rate(&y_test, &test_pred);
    println!("  Hit Rate: {:.2}%", hit_rate * 100.0);

    // =========================
    // Gradient Descent Approach
    // =========================
    println!("\n" + "=".repeat(50).as_str());
    println!("Gradient Descent Linear Regression");
    println!("{}", "=".repeat(50));

    let mut gd_model = LinearRegressionGD::new(0.01, 5000, 1e-8, true);
    gd_model.fit(&x_train, &y_train)?;

    println!("\nTraining converged after {} iterations", gd_model.cost_history.len());
    println!(
        "Final cost: {:.8}",
        gd_model.cost_history.last().unwrap_or(&0.0)
    );

    let gd_test_pred = gd_model.predict(&x_test)?;
    let gd_test_metrics = RegressionMetrics::calculate(&y_test, &gd_test_pred);

    println!("\nTest Performance (GD):");
    println!("  R²:   {:.6}", gd_test_metrics.r2);
    println!("  RMSE: {:.6}", gd_test_metrics.rmse);
    println!("  IC:   {:.6}", gd_test_metrics.ic);

    // ==================
    // Feature Importance
    // ==================
    println!("\n" + "=".repeat(50).as_str());
    println!("Feature Importance (by coefficient magnitude)");
    println!("{}", "=".repeat(50));

    if let Some(ref coef) = ols_model.coefficients {
        let mut importance: Vec<(usize, f64)> = coef
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c.abs()))
            .collect();

        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\nTop 10 most important features:");
        for (rank, (idx, importance)) in importance.iter().take(10).enumerate() {
            println!(
                "  {:2}. {:20} ({:+.6})",
                rank + 1,
                feature_names[*idx],
                coef[*idx]
            );
        }
    }

    // =====================
    // Correlation Analysis
    // =====================
    println!("\n" + "=".repeat(50).as_str());
    println!("Correlation with Target");
    println!("{}", "=".repeat(50));

    let corr_matrix = DataProcessor::correlation_matrix(&x_clean);

    // Since target is separate, calculate correlations manually
    let mut correlations: Vec<(String, f64)> = Vec::new();
    for (j, name) in feature_names.iter().enumerate() {
        let col = x_clean.column(j);
        let mean_x = col.mean().unwrap();
        let mean_y = y_clean.mean().unwrap();
        let std_x = col.std(0.0);
        let std_y = y_clean.std(0.0);

        if std_x > 1e-10 && std_y > 1e-10 {
            let cov: f64 = col
                .iter()
                .zip(y_clean.iter())
                .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
                .sum::<f64>()
                / col.len() as f64;

            let corr = cov / (std_x * std_y);
            correlations.push((name.clone(), corr));
        }
    }

    correlations.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    println!("\nTop 10 features most correlated with returns:");
    for (rank, (name, corr)) in correlations.iter().take(10).enumerate() {
        println!("  {:2}. {:20}: {:+.4}", rank + 1, name, corr);
    }

    // =================
    // Prediction Sample
    // =================
    println!("\n" + "=".repeat(50).as_str());
    println!("Sample Predictions vs Actual");
    println!("{}", "=".repeat(50));

    println!("\nLast 10 test samples:");
    println!("{:>12} {:>12} {:>12}", "Actual", "Predicted", "Error");
    println!("{}", "-".repeat(40));

    let n_test = y_test.len();
    for i in (n_test.saturating_sub(10))..n_test {
        let actual = y_test[i] * 100.0; // Convert to percentage
        let pred = test_pred[i] * 100.0;
        let error = actual - pred;
        println!("{:>11.4}% {:>11.4}% {:>11.4}%", actual, pred, error);
    }

    println!("\nLinear regression example completed!");

    Ok(())
}
