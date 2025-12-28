//! Example: Logistic Regression for Price Direction Prediction
//!
//! This example demonstrates how to use logistic regression to predict
//! whether cryptocurrency prices will go up or down.

use chrono::{Duration, Utc};
use linear_models_crypto::{
    api::bybit::{BybitClient, Interval},
    data::{
        features::FeatureEngineering,
        processor::{train_test_split, DataProcessor},
    },
    metrics::classification::{ClassificationMetrics, roc_curve},
    models::logistic::{LogisticRegression, Regularization},
};
use ndarray::Array1;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("================================================");
    println!("  Logistic Regression: Price Direction Prediction");
    println!("================================================\n");

    // Fetch data
    println!("Fetching BTCUSDT daily data...");
    let client = BybitClient::new();

    let end_time = Utc::now().timestamp_millis();
    let start_time = (Utc::now() - Duration::days(365)).timestamp_millis();

    let klines = client.get_klines_history("BTCUSDT", Interval::Day1, start_time, end_time)?;
    println!("Fetched {} daily candles\n", klines.len());

    // Generate features
    println!("Generating features...");
    let (features, feature_names) = FeatureEngineering::generate_features(&klines);

    // Create binary target (1 = price up, 0 = price down)
    let target = FeatureEngineering::create_binary_target(&klines, 1);

    // Clean data
    let (x_clean, y_clean) = DataProcessor::dropna(&features, &target);
    println!(
        "Dataset: {} samples, {} features",
        x_clean.nrows(),
        x_clean.ncols()
    );

    // Check class balance
    let n_positive = y_clean.iter().filter(|&&y| y >= 0.5).count();
    let n_negative = y_clean.len() - n_positive;
    println!(
        "Class distribution: {} up ({:.1}%), {} down ({:.1}%)\n",
        n_positive,
        n_positive as f64 / y_clean.len() as f64 * 100.0,
        n_negative,
        n_negative as f64 / y_clean.len() as f64 * 100.0
    );

    // Standardize features
    let mut processor = DataProcessor::new();
    processor.fit_standard_scaler(&x_clean);
    let x_scaled = processor.transform_standard(&x_clean);

    // Split data
    let (x_train, x_test, y_train, y_test) = train_test_split(&x_scaled, &y_clean, 0.2);
    println!(
        "Train: {} samples, Test: {} samples",
        x_train.nrows(),
        x_test.nrows()
    );

    // ============================
    // Basic Logistic Regression
    // ============================
    println!("\n{}", "=".repeat(50));
    println!("Logistic Regression (No Regularization)");
    println!("{}", "=".repeat(50));

    let mut lr = LogisticRegression::new(0.1, 2000, 1e-6, true, Regularization::None);
    lr.fit(&x_train, &y_train)?;

    println!("\nTraining completed in {} iterations", lr.cost_history.len());
    println!(
        "Final cost: {:.6}",
        lr.cost_history.last().unwrap_or(&0.0)
    );

    // Predictions
    let train_proba = lr.predict_proba(&x_train)?;
    let test_proba = lr.predict_proba(&x_test)?;
    let train_pred = lr.predict(&x_train)?;
    let test_pred = lr.predict(&x_test)?;

    // Metrics
    let train_metrics =
        ClassificationMetrics::calculate_with_proba(&y_train, &train_pred, Some(&train_proba));
    let test_metrics =
        ClassificationMetrics::calculate_with_proba(&y_test, &test_pred, Some(&test_proba));

    println!("\nTraining Metrics:");
    println!("  Accuracy:  {:.4}", train_metrics.accuracy);
    println!("  Precision: {:.4}", train_metrics.precision);
    println!("  Recall:    {:.4}", train_metrics.recall);
    println!("  F1 Score:  {:.4}", train_metrics.f1);
    if let Some(auc) = train_metrics.auc_roc {
        println!("  AUC-ROC:   {:.4}", auc);
    }

    println!("\nTest Metrics:");
    println!("{}", test_metrics.report());

    // ============================
    // L2 Regularized (Ridge)
    // ============================
    println!("\n{}", "=".repeat(50));
    println!("Logistic Regression with L2 Regularization");
    println!("{}", "=".repeat(50));

    let c_values = vec![0.01, 0.1, 1.0, 10.0, 100.0];

    println!(
        "\n{:>10} {:>10} {:>10} {:>10} {:>10}",
        "C", "Accuracy", "Precision", "Recall", "AUC"
    );
    println!("{}", "-".repeat(55));

    let mut best_c = 1.0;
    let mut best_auc = 0.0;

    for &c in &c_values {
        let mut lr_l2 = LogisticRegression::with_l2(c);
        lr_l2.fit(&x_train, &y_train)?;

        let proba = lr_l2.predict_proba(&x_test)?;
        let pred = lr_l2.predict(&x_test)?;
        let metrics = ClassificationMetrics::calculate_with_proba(&y_test, &pred, Some(&proba));

        let auc = metrics.auc_roc.unwrap_or(0.5);
        println!(
            "{:>10.2} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            c, metrics.accuracy, metrics.precision, metrics.recall, auc
        );

        if auc > best_auc {
            best_auc = auc;
            best_c = c;
        }
    }

    println!("\nBest C: {} (AUC = {:.4})", best_c, best_auc);

    // ============================
    // L1 Regularized (Lasso)
    // ============================
    println!("\n{}", "=".repeat(50));
    println!("Logistic Regression with L1 Regularization");
    println!("{}", "=".repeat(50));

    println!(
        "\n{:>10} {:>10} {:>10} {:>10} {:>12}",
        "C", "Accuracy", "F1", "AUC", "Non-zero"
    );
    println!("{}", "-".repeat(55));

    for &c in &c_values {
        let mut lr_l1 = LogisticRegression::with_l1(c);
        lr_l1.fit(&x_train, &y_train)?;

        let proba = lr_l1.predict_proba(&x_test)?;
        let pred = lr_l1.predict(&x_test)?;
        let metrics = ClassificationMetrics::calculate_with_proba(&y_test, &pred, Some(&proba));

        let n_nonzero = lr_l1
            .coefficients
            .as_ref()
            .map(|c| c.iter().filter(|&&v| v.abs() > 1e-6).count())
            .unwrap_or(0);

        println!(
            "{:>10.2} {:>10.4} {:>10.4} {:>10.4} {:>12}",
            c,
            metrics.accuracy,
            metrics.f1,
            metrics.auc_roc.unwrap_or(0.5),
            n_nonzero
        );
    }

    // ============================
    // Threshold Analysis
    // ============================
    println!("\n{}", "=".repeat(50));
    println!("Classification Threshold Analysis");
    println!("{}", "=".repeat(50));

    let mut best_lr = LogisticRegression::with_l2(best_c);
    best_lr.fit(&x_train, &y_train)?;
    let proba = best_lr.predict_proba(&x_test)?;

    println!(
        "\n{:>10} {:>10} {:>10} {:>10} {:>10}",
        "Threshold", "Accuracy", "Precision", "Recall", "F1"
    );
    println!("{}", "-".repeat(55));

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7] {
        let pred = best_lr.predict_with_threshold(&x_test, threshold)?;
        let metrics = ClassificationMetrics::calculate(&y_test, &pred);

        println!(
            "{:>10.2} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            threshold, metrics.accuracy, metrics.precision, metrics.recall, metrics.f1
        );
    }

    // ============================
    // Feature Importance
    // ============================
    println!("\n{}", "=".repeat(50));
    println!("Feature Importance (Odds Ratios)");
    println!("{}", "=".repeat(50));

    println!("{}", best_lr.summary(Some(&feature_names)));

    if let Some(ref coef) = best_lr.coefficients {
        let mut importance: Vec<(usize, f64, f64)> = coef
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c, c.exp()))
            .collect();

        importance.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        println!("\nTop 10 predictive features:");
        println!(
            "{:>25} {:>12} {:>12}",
            "Feature", "Coefficient", "Odds Ratio"
        );
        println!("{}", "-".repeat(52));

        for (idx, coef, or) in importance.iter().take(10) {
            let direction = if *coef > 0.0 { "↑" } else { "↓" };
            println!(
                "{:>25} {:>11.4} {:>11.4} {}",
                feature_names[*idx], coef, or, direction
            );
        }
    }

    // ============================
    // ROC Curve Points
    // ============================
    println!("\n{}", "=".repeat(50));
    println!("ROC Curve Data Points");
    println!("{}", "=".repeat(50));

    let (fprs, tprs, thresholds) = roc_curve(&y_test, &proba);

    println!("\n{:>12} {:>12} {:>12}", "Threshold", "FPR", "TPR");
    println!("{}", "-".repeat(40));

    // Sample some points
    let step = thresholds.len() / 10;
    for i in (0..thresholds.len()).step_by(step.max(1)) {
        println!(
            "{:>12.4} {:>12.4} {:>12.4}",
            thresholds[i], fprs[i], tprs[i]
        );
    }

    // ============================
    // Prediction Examples
    // ============================
    println!("\n{}", "=".repeat(50));
    println!("Sample Predictions");
    println!("{}", "=".repeat(50));

    println!(
        "\n{:>8} {:>12} {:>12} {:>10}",
        "Actual", "Probability", "Predicted", "Correct"
    );
    println!("{}", "-".repeat(45));

    let pred_class = best_lr.predict(&x_test)?;

    for i in 0..10.min(y_test.len()) {
        let actual = if y_test[i] >= 0.5 { "UP" } else { "DOWN" };
        let predicted = if pred_class[i] >= 0.5 { "UP" } else { "DOWN" };
        let correct = if actual == predicted { "✓" } else { "✗" };

        println!(
            "{:>8} {:>12.4} {:>12} {:>10}",
            actual, proba[i], predicted, correct
        );
    }

    // Calculate final trading performance
    println!("\n{}", "=".repeat(50));
    println!("Simulated Trading Performance");
    println!("{}", "=".repeat(50));

    // Get actual returns for test period
    let test_start = (x_clean.nrows() as f64 * 0.8) as usize;
    let actual_returns: Vec<f64> = klines
        .windows(2)
        .skip(test_start)
        .take(y_test.len())
        .map(|w| (w[1].close / w[0].close) - 1.0)
        .collect();

    // Strategy returns: go long when predict up, flat when predict down
    let strategy_returns: Vec<f64> = actual_returns
        .iter()
        .zip(pred_class.iter())
        .map(|(&ret, &pred)| if pred >= 0.5 { ret } else { 0.0 })
        .collect();

    let buy_hold_return: f64 = actual_returns.iter().map(|r| r + 1.0).product::<f64>() - 1.0;
    let strategy_return: f64 = strategy_returns.iter().map(|r| r + 1.0).product::<f64>() - 1.0;

    let n_trades = pred_class.iter().filter(|&&p| p >= 0.5).count();
    let winning_trades = strategy_returns.iter().filter(|&&r| r > 0.0).count();

    println!("\nBuy & Hold Return:  {:+.2}%", buy_hold_return * 100.0);
    println!("Strategy Return:    {:+.2}%", strategy_return * 100.0);
    println!("Number of trades:   {}", n_trades);
    println!(
        "Win rate:           {:.1}%",
        if n_trades > 0 {
            winning_trades as f64 / n_trades as f64 * 100.0
        } else {
            0.0
        }
    );

    println!("\nLogistic regression example completed!");

    Ok(())
}
