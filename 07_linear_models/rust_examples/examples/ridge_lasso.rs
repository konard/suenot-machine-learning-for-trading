//! Example: Ridge and Lasso Regression for Crypto Prediction
//!
//! This example demonstrates regularized regression methods and their
//! effect on model coefficients and prediction performance.

use chrono::{Duration, Utc};
use linear_models_crypto::{
    api::bybit::{BybitClient, Interval},
    data::{
        features::FeatureEngineering,
        processor::{train_test_split, DataProcessor},
    },
    metrics::regression::RegressionMetrics,
    models::{
        linear::LinearRegression,
        regularization::{ElasticNet, LassoRegression, RidgeRegression},
    },
};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("==============================================");
    println!("  Regularized Regression: Ridge & Lasso");
    println!("==============================================\n");

    // Fetch data
    println!("Fetching ETHUSDT 4-hour data...");
    let client = BybitClient::new();

    let end_time = Utc::now().timestamp_millis();
    let start_time = (Utc::now() - Duration::days(60)).timestamp_millis();

    let klines = client.get_klines_history("ETHUSDT", Interval::Hour4, start_time, end_time)?;
    println!("Fetched {} candles\n", klines.len());

    // Generate features
    let (features, feature_names) = FeatureEngineering::generate_features(&klines);
    let target = FeatureEngineering::create_target(&klines, 1);

    // Clean and prepare data
    let (x_clean, y_clean) = DataProcessor::dropna(&features, &target);

    let mut processor = DataProcessor::new();
    processor.fit_standard_scaler(&x_clean);
    let x_scaled = processor.transform_standard(&x_clean);

    let (x_train, x_test, y_train, y_test) = train_test_split(&x_scaled, &y_clean, 0.2);

    println!(
        "Data prepared: {} train samples, {} test samples, {} features\n",
        x_train.nrows(),
        x_test.nrows(),
        x_train.ncols()
    );

    // ================
    // OLS (baseline)
    // ================
    println!("{}", "=".repeat(50));
    println!("Baseline: OLS Linear Regression");
    println!("{}", "=".repeat(50));

    let mut ols = LinearRegression::new(true);
    ols.fit(&x_train, &y_train)?;

    let ols_pred = ols.predict(&x_test)?;
    let ols_metrics = RegressionMetrics::calculate(&y_test, &ols_pred);

    println!("Test R²:    {:.6}", ols_metrics.r2);
    println!("Test RMSE:  {:.6}", ols_metrics.rmse);
    println!("Test IC:    {:.6}", ols_metrics.ic);

    if let Some(ref coef) = ols.coefficients {
        let l2_norm: f64 = coef.iter().map(|c| c.powi(2)).sum::<f64>().sqrt();
        let l1_norm: f64 = coef.iter().map(|c| c.abs()).sum();
        let n_nonzero = coef.iter().filter(|&&c| c.abs() > 1e-10).count();

        println!("Coefficient L2 norm: {:.4}", l2_norm);
        println!("Coefficient L1 norm: {:.4}", l1_norm);
        println!("Non-zero coefficients: {}", n_nonzero);
    }

    // ==================
    // Ridge Regression
    // ==================
    println!("\n{}", "=".repeat(50));
    println!("Ridge Regression (L2 Regularization)");
    println!("{}", "=".repeat(50));

    let alphas = vec![0.01, 0.1, 1.0, 10.0, 100.0];

    println!("\nAlpha sensitivity analysis:");
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>12}",
        "Alpha", "R²", "RMSE", "IC", "L2 Norm"
    );
    println!("{}", "-".repeat(55));

    let mut best_ridge_alpha = 0.0;
    let mut best_ridge_r2 = f64::NEG_INFINITY;

    for &alpha in &alphas {
        let mut ridge = RidgeRegression::new(alpha, true, false);
        ridge.fit(&x_train, &y_train)?;

        let pred = ridge.predict(&x_test)?;
        let metrics = RegressionMetrics::calculate(&y_test, &pred);

        let l2_norm = ridge
            .coefficients
            .as_ref()
            .map(|c| c.iter().map(|v| v.powi(2)).sum::<f64>().sqrt())
            .unwrap_or(0.0);

        println!(
            "{:>10.2} {:>10.6} {:>10.6} {:>10.6} {:>12.4}",
            alpha, metrics.r2, metrics.rmse, metrics.ic, l2_norm
        );

        if metrics.r2 > best_ridge_r2 {
            best_ridge_r2 = metrics.r2;
            best_ridge_alpha = alpha;
        }
    }

    println!("\nBest Ridge alpha: {} (R² = {:.6})", best_ridge_alpha, best_ridge_r2);

    // =================
    // Lasso Regression
    // =================
    println!("\n{}", "=".repeat(50));
    println!("Lasso Regression (L1 Regularization)");
    println!("{}", "=".repeat(50));

    let lasso_alphas = vec![0.0001, 0.001, 0.01, 0.1, 1.0];

    println!("\nAlpha sensitivity analysis:");
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>12}",
        "Alpha", "R²", "RMSE", "IC", "Non-zero"
    );
    println!("{}", "-".repeat(55));

    let mut best_lasso_alpha = 0.0;
    let mut best_lasso_r2 = f64::NEG_INFINITY;

    for &alpha in &lasso_alphas {
        let mut lasso = LassoRegression::new(alpha, true, 2000, 1e-6);
        lasso.fit(&x_train, &y_train)?;

        let pred = lasso.predict(&x_test)?;
        let metrics = RegressionMetrics::calculate(&y_test, &pred);

        let n_nonzero = lasso.n_nonzero();

        println!(
            "{:>10.4} {:>10.6} {:>10.6} {:>10.6} {:>12}",
            alpha, metrics.r2, metrics.rmse, metrics.ic, n_nonzero
        );

        if metrics.r2 > best_lasso_r2 {
            best_lasso_r2 = metrics.r2;
            best_lasso_alpha = alpha;
        }
    }

    println!("\nBest Lasso alpha: {} (R² = {:.6})", best_lasso_alpha, best_lasso_r2);

    // Feature selection with Lasso
    println!("\nFeature Selection with Lasso (alpha = 0.01):");
    let mut lasso_selection = LassoRegression::new(0.01, true, 2000, 1e-6);
    lasso_selection.fit(&x_train, &y_train)?;

    let selected = lasso_selection.selected_features();
    println!("Selected {} out of {} features:", selected.len(), feature_names.len());

    if let Some(ref coef) = lasso_selection.coefficients {
        let mut selected_with_coef: Vec<(usize, f64)> = selected
            .iter()
            .map(|&i| (i, coef[i]))
            .collect();

        selected_with_coef.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        for (idx, c) in selected_with_coef.iter().take(10) {
            println!("  {:20}: {:+.6}", feature_names[*idx], c);
        }
    }

    // ==============
    // Elastic Net
    // ==============
    println!("\n{}", "=".repeat(50));
    println!("Elastic Net (L1 + L2 Regularization)");
    println!("{}", "=".repeat(50));

    let l1_ratios = vec![0.1, 0.3, 0.5, 0.7, 0.9];

    println!("\nL1 ratio sensitivity (alpha = 0.1):");
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>12}",
        "L1 Ratio", "R²", "RMSE", "IC", "Non-zero"
    );
    println!("{}", "-".repeat(55));

    for &l1_ratio in &l1_ratios {
        let mut enet = ElasticNet::new(0.1, l1_ratio, true, 2000, 1e-6);
        enet.fit(&x_train, &y_train)?;

        let pred = enet.predict(&x_test)?;
        let metrics = RegressionMetrics::calculate(&y_test, &pred);

        let n_nonzero = enet
            .coefficients
            .as_ref()
            .map(|c| c.iter().filter(|&&v| v.abs() > 1e-10).count())
            .unwrap_or(0);

        println!(
            "{:>10.1} {:>10.6} {:>10.6} {:>10.6} {:>12}",
            l1_ratio, metrics.r2, metrics.rmse, metrics.ic, n_nonzero
        );
    }

    // =======================
    // Regularization Paths
    // =======================
    println!("\n{}", "=".repeat(50));
    println!("Regularization Path Visualization");
    println!("{}", "=".repeat(50));

    let path_alphas: Vec<f64> = (0..20).map(|i| 0.001 * (1.5f64).powi(i)).collect();

    println!("\nRidge coefficient paths (first 5 features):");
    let ridge_path = RidgeRegression::regularization_path(&x_train, &y_train, &path_alphas)?;

    println!("{:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
             "Alpha", &feature_names[0][..8.min(feature_names[0].len())],
             &feature_names[1][..8.min(feature_names[1].len())],
             &feature_names[2][..8.min(feature_names[2].len())],
             &feature_names[3][..8.min(feature_names[3].len())],
             &feature_names[4][..8.min(feature_names[4].len())]);
    println!("{}", "-".repeat(65));

    for (alpha, coef) in ridge_path.iter().take(10) {
        println!(
            "{:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            alpha, coef[0], coef[1], coef[2], coef[3], coef[4]
        );
    }

    println!("\nLasso coefficient sparsity path:");
    let lasso_path = LassoRegression::regularization_path(&x_train, &y_train, &path_alphas)?;

    println!("{:>10} {:>12}", "Alpha", "Non-zero");
    println!("{}", "-".repeat(25));

    for (alpha, _, n_nonzero) in lasso_path.iter() {
        println!("{:>10.4} {:>12}", alpha, n_nonzero);
    }

    // =================
    // Model Comparison
    // =================
    println!("\n{}", "=".repeat(50));
    println!("Final Model Comparison");
    println!("{}", "=".repeat(50));

    // Train best models
    let mut best_ridge = RidgeRegression::new(best_ridge_alpha, true, false);
    best_ridge.fit(&x_train, &y_train)?;
    let ridge_pred = best_ridge.predict(&x_test)?;
    let ridge_metrics = RegressionMetrics::calculate(&y_test, &ridge_pred);

    let mut best_lasso = LassoRegression::new(best_lasso_alpha, true, 2000, 1e-6);
    best_lasso.fit(&x_train, &y_train)?;
    let lasso_pred = best_lasso.predict(&x_test)?;
    let lasso_metrics = RegressionMetrics::calculate(&y_test, &lasso_pred);

    println!(
        "\n{:>15} {:>10} {:>10} {:>10} {:>10}",
        "Model", "R²", "RMSE", "IC", "Hit Rate"
    );
    println!("{}", "-".repeat(55));

    let ols_hr = RegressionMetrics::hit_rate(&y_test, &ols_pred);
    let ridge_hr = RegressionMetrics::hit_rate(&y_test, &ridge_pred);
    let lasso_hr = RegressionMetrics::hit_rate(&y_test, &lasso_pred);

    println!(
        "{:>15} {:>10.6} {:>10.6} {:>10.6} {:>9.2}%",
        "OLS", ols_metrics.r2, ols_metrics.rmse, ols_metrics.ic, ols_hr * 100.0
    );
    println!(
        "{:>15} {:>10.6} {:>10.6} {:>10.6} {:>9.2}%",
        format!("Ridge({})", best_ridge_alpha),
        ridge_metrics.r2,
        ridge_metrics.rmse,
        ridge_metrics.ic,
        ridge_hr * 100.0
    );
    println!(
        "{:>15} {:>10.6} {:>10.6} {:>10.6} {:>9.2}%",
        format!("Lasso({})", best_lasso_alpha),
        lasso_metrics.r2,
        lasso_metrics.rmse,
        lasso_metrics.ic,
        lasso_hr * 100.0
    );

    println!("\nRegularized regression example completed!");

    Ok(())
}
