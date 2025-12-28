//! Example: Full ML Pipeline for Crypto Trading
//!
//! This example demonstrates a complete machine learning pipeline
//! for cryptocurrency price prediction, including:
//! - Data fetching from Bybit
//! - Feature engineering
//! - Model training and selection
//! - Backtesting simulation

use chrono::{Duration, Utc};
use linear_models_crypto::{
    api::bybit::{BybitClient, Interval, Kline},
    data::{
        features::FeatureEngineering,
        processor::{train_test_split, DataProcessor},
    },
    metrics::{
        classification::ClassificationMetrics,
        regression::{RegressionMetrics, TimeSeriesCV},
    },
    models::{
        linear::LinearRegression,
        logistic::{LogisticRegression, Regularization},
        regularization::{LassoRegression, RidgeRegression},
    },
};
use ndarray::{Array1, Array2};

/// Simple backtest result
struct BacktestResult {
    total_return: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
    n_trades: usize,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║     Cryptocurrency ML Trading Pipeline               ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // ================
    // 1. Data Fetching
    // ================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Step 1: Fetching Market Data");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let client = BybitClient::new();
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    let end_time = Utc::now().timestamp_millis();
    let start_time = (Utc::now() - Duration::days(90)).timestamp_millis();

    println!("Fetching 90 days of 4-hour data for multiple assets...\n");

    let mut all_data: Vec<(&str, Vec<Kline>)> = Vec::new();

    for symbol in &symbols {
        match client.get_klines_history(symbol, Interval::Hour4, start_time, end_time) {
            Ok(klines) => {
                println!("  {} - {} candles fetched", symbol, klines.len());
                all_data.push((symbol, klines));
            }
            Err(e) => {
                println!("  {} - Error: {}", symbol, e);
            }
        }
    }

    // Use BTC as primary asset
    let (primary_symbol, klines) = all_data
        .iter()
        .find(|(s, _)| *s == "BTCUSDT")
        .expect("BTC data required");

    println!("\nPrimary asset: {} ({} candles)", primary_symbol, klines.len());

    // ====================
    // 2. Feature Engineering
    // ====================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Step 2: Feature Engineering");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let (features, feature_names) = FeatureEngineering::generate_features(klines);
    let target_returns = FeatureEngineering::create_target(klines, 1);
    let target_direction = FeatureEngineering::create_binary_target(klines, 1);

    println!("Generated {} features:", feature_names.len());
    for (i, name) in feature_names.iter().enumerate().take(10) {
        println!("  {:2}. {}", i + 1, name);
    }
    if feature_names.len() > 10 {
        println!("  ... and {} more", feature_names.len() - 10);
    }

    // Clean data
    let (x_clean, y_returns) = DataProcessor::dropna(&features, &target_returns);
    let (_, y_direction) = DataProcessor::dropna(&features, &target_direction);

    println!("\nAfter cleaning: {} samples", x_clean.nrows());

    // =================
    // 3. Data Splitting
    // =================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Step 3: Data Splitting & Preprocessing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Time series split (no future leakage)
    let (x_train, x_test, y_train_ret, y_test_ret) = train_test_split(&x_clean, &y_returns, 0.2);
    let (_, _, y_train_dir, y_test_dir) = train_test_split(&x_clean, &y_direction, 0.2);

    println!("Training set: {} samples", x_train.nrows());
    println!("Test set:     {} samples", x_test.nrows());

    // Standardize (fit on train only)
    let mut processor = DataProcessor::new();
    processor.fit_standard_scaler(&x_train);
    let x_train_scaled = processor.transform_standard(&x_train);
    let x_test_scaled = processor.transform_standard(&x_test);

    println!("Features standardized (zero mean, unit variance)");

    // =====================
    // 4. Model Training
    // =====================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Step 4: Model Training & Selection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Cross-validation setup
    let cv = TimeSeriesCV::new(5, 50, 20, 1);
    let splits = cv.split(x_train_scaled.nrows());
    println!("Using {}-fold time series cross-validation\n", splits.len());

    // ---- Regression Models ----
    println!("A. Regression Models (Predicting Returns)\n");

    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "Model", "CV R²", "CV IC", "CV RMSE"
    );
    println!("{}", "-".repeat(55));

    // OLS
    let (ols_r2, ols_ic, ols_rmse) = cross_validate_regression(&x_train_scaled, &y_train_ret, &splits, "ols")?;
    println!("{:<20} {:>10.4} {:>10.4} {:>10.6}", "OLS", ols_r2, ols_ic, ols_rmse);

    // Ridge
    let (ridge_r2, ridge_ic, ridge_rmse) = cross_validate_regression(&x_train_scaled, &y_train_ret, &splits, "ridge")?;
    println!("{:<20} {:>10.4} {:>10.4} {:>10.6}", "Ridge (α=1.0)", ridge_r2, ridge_ic, ridge_rmse);

    // Lasso
    let (lasso_r2, lasso_ic, lasso_rmse) = cross_validate_regression(&x_train_scaled, &y_train_ret, &splits, "lasso")?;
    println!("{:<20} {:>10.4} {:>10.4} {:>10.6}", "Lasso (α=0.01)", lasso_r2, lasso_ic, lasso_rmse);

    // ---- Classification Models ----
    println!("\nB. Classification Models (Predicting Direction)\n");

    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "Model", "CV Acc", "CV F1", "CV AUC"
    );
    println!("{}", "-".repeat(55));

    // Logistic Regression
    let (lr_acc, lr_f1, lr_auc) = cross_validate_classification(&x_train_scaled, &y_train_dir, &splits, "none")?;
    println!("{:<20} {:>10.4} {:>10.4} {:>10.4}", "LogReg", lr_acc, lr_f1, lr_auc);

    // L2 Regularized
    let (lr_l2_acc, lr_l2_f1, lr_l2_auc) = cross_validate_classification(&x_train_scaled, &y_train_dir, &splits, "l2")?;
    println!("{:<20} {:>10.4} {:>10.4} {:>10.4}", "LogReg-L2", lr_l2_acc, lr_l2_f1, lr_l2_auc);

    // L1 Regularized
    let (lr_l1_acc, lr_l1_f1, lr_l1_auc) = cross_validate_classification(&x_train_scaled, &y_train_dir, &splits, "l1")?;
    println!("{:<20} {:>10.4} {:>10.4} {:>10.4}", "LogReg-L1", lr_l1_acc, lr_l1_f1, lr_l1_auc);

    // ===================
    // 5. Final Evaluation
    // ===================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Step 5: Test Set Evaluation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Train best regression model (Ridge)
    let mut ridge = RidgeRegression::new(1.0, true, false);
    ridge.fit(&x_train_scaled, &y_train_ret)?;
    let ridge_pred = ridge.predict(&x_test_scaled)?;
    let ridge_metrics = RegressionMetrics::calculate(&y_test_ret, &ridge_pred);

    println!("Best Regression Model: Ridge");
    println!("  Test R²:   {:.4}", ridge_metrics.r2);
    println!("  Test RMSE: {:.6}", ridge_metrics.rmse);
    println!("  Test IC:   {:.4}", ridge_metrics.ic);

    // Train best classification model
    let mut logreg = LogisticRegression::with_l2(1.0);
    logreg.fit(&x_train_scaled, &y_train_dir)?;
    let logreg_proba = logreg.predict_proba(&x_test_scaled)?;
    let logreg_pred = logreg.predict(&x_test_scaled)?;
    let logreg_metrics = ClassificationMetrics::calculate_with_proba(&y_test_dir, &logreg_pred, Some(&logreg_proba));

    println!("\nBest Classification Model: Logistic Regression L2");
    println!("  Test Accuracy:  {:.4}", logreg_metrics.accuracy);
    println!("  Test Precision: {:.4}", logreg_metrics.precision);
    println!("  Test Recall:    {:.4}", logreg_metrics.recall);
    println!("  Test F1:        {:.4}", logreg_metrics.f1);
    println!("  Test AUC:       {:.4}", logreg_metrics.auc_roc.unwrap_or(0.5));

    // ==============
    // 6. Backtesting
    // ==============
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Step 6: Backtesting Simulation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Get actual returns for test period
    let test_start = x_train_scaled.nrows();
    let actual_returns: Vec<f64> = klines
        .windows(2)
        .skip(test_start)
        .take(y_test_ret.len())
        .map(|w| (w[1].close / w[0].close) - 1.0)
        .collect();

    // Strategy 1: Regression-based (use predictions as position size)
    let regression_result = backtest_regression(&actual_returns, &ridge_pred);

    // Strategy 2: Classification-based (binary positions)
    let classification_result = backtest_classification(&actual_returns, &logreg_pred);

    // Strategy 3: Buy and Hold
    let buy_hold_result = backtest_buy_hold(&actual_returns);

    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>10} {:>8}",
        "Strategy", "Return", "Sharpe", "Max DD", "Win Rate", "Trades"
    );
    println!("{}", "-".repeat(80));

    println!(
        "{:<20} {:>11.2}% {:>12.2} {:>11.2}% {:>9.1}% {:>8}",
        "Buy & Hold",
        buy_hold_result.total_return * 100.0,
        buy_hold_result.sharpe_ratio,
        buy_hold_result.max_drawdown * 100.0,
        buy_hold_result.win_rate * 100.0,
        buy_hold_result.n_trades
    );

    println!(
        "{:<20} {:>11.2}% {:>12.2} {:>11.2}% {:>9.1}% {:>8}",
        "Regression Signal",
        regression_result.total_return * 100.0,
        regression_result.sharpe_ratio,
        regression_result.max_drawdown * 100.0,
        regression_result.win_rate * 100.0,
        regression_result.n_trades
    );

    println!(
        "{:<20} {:>11.2}% {:>12.2} {:>11.2}% {:>9.1}% {:>8}",
        "Classification",
        classification_result.total_return * 100.0,
        classification_result.sharpe_ratio,
        classification_result.max_drawdown * 100.0,
        classification_result.win_rate * 100.0,
        classification_result.n_trades
    );

    // ==============
    // 7. Summary
    // ==============
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Summary & Recommendations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Key Findings:");
    println!("  • Ridge regression provides best regression fit (R² = {:.4})", ridge_metrics.r2);
    println!("  • Logistic regression achieves {:.1}% directional accuracy", logreg_metrics.accuracy * 100.0);
    println!(
        "  • Information Coefficient (IC) of {:.4} shows predictive power",
        ridge_metrics.ic
    );

    let best_strategy = if classification_result.sharpe_ratio > regression_result.sharpe_ratio {
        ("Classification", classification_result.sharpe_ratio)
    } else {
        ("Regression", regression_result.sharpe_ratio)
    };

    println!(
        "\nBest performing strategy: {} (Sharpe: {:.2})",
        best_strategy.0, best_strategy.1
    );

    if best_strategy.1 > 1.0 {
        println!("  ✓ Strategy shows promising risk-adjusted returns");
    } else if best_strategy.1 > 0.0 {
        println!("  △ Strategy is profitable but with moderate risk-adjusted returns");
    } else {
        println!("  ✗ Strategy underperforms, consider refinement");
    }

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║     Pipeline Completed Successfully!                 ║");
    println!("╚══════════════════════════════════════════════════════╝");

    Ok(())
}

// Helper functions

fn cross_validate_regression(
    x: &Array2<f64>,
    y: &Array1<f64>,
    splits: &[(Vec<usize>, Vec<usize>)],
    model_type: &str,
) -> anyhow::Result<(f64, f64, f64)> {
    let mut r2_scores = Vec::new();
    let mut ic_scores = Vec::new();
    let mut rmse_scores = Vec::new();

    for (train_idx, test_idx) in splits {
        let x_train = get_rows(x, train_idx);
        let y_train = get_elements(y, train_idx);
        let x_test = get_rows(x, test_idx);
        let y_test = get_elements(y, test_idx);

        let pred = match model_type {
            "ols" => {
                let mut model = LinearRegression::new(true);
                model.fit(&x_train, &y_train)?;
                model.predict(&x_test)?
            }
            "ridge" => {
                let mut model = RidgeRegression::new(1.0, true, false);
                model.fit(&x_train, &y_train)?;
                model.predict(&x_test)?
            }
            "lasso" => {
                let mut model = LassoRegression::new(0.01, true, 1000, 1e-6);
                model.fit(&x_train, &y_train)?;
                model.predict(&x_test)?
            }
            _ => panic!("Unknown model type"),
        };

        let metrics = RegressionMetrics::calculate(&y_test, &pred);
        r2_scores.push(metrics.r2);
        ic_scores.push(metrics.ic);
        rmse_scores.push(metrics.rmse);
    }

    Ok((
        r2_scores.iter().sum::<f64>() / r2_scores.len() as f64,
        ic_scores.iter().sum::<f64>() / ic_scores.len() as f64,
        rmse_scores.iter().sum::<f64>() / rmse_scores.len() as f64,
    ))
}

fn cross_validate_classification(
    x: &Array2<f64>,
    y: &Array1<f64>,
    splits: &[(Vec<usize>, Vec<usize>)],
    reg_type: &str,
) -> anyhow::Result<(f64, f64, f64)> {
    let mut acc_scores = Vec::new();
    let mut f1_scores = Vec::new();
    let mut auc_scores = Vec::new();

    for (train_idx, test_idx) in splits {
        let x_train = get_rows(x, train_idx);
        let y_train = get_elements(y, train_idx);
        let x_test = get_rows(x, test_idx);
        let y_test = get_elements(y, test_idx);

        let mut model = match reg_type {
            "none" => LogisticRegression::new(0.1, 1000, 1e-6, true, Regularization::None),
            "l2" => LogisticRegression::with_l2(1.0),
            "l1" => LogisticRegression::with_l1(1.0),
            _ => panic!("Unknown regularization"),
        };

        model.fit(&x_train, &y_train)?;
        let proba = model.predict_proba(&x_test)?;
        let pred = model.predict(&x_test)?;

        let metrics = ClassificationMetrics::calculate_with_proba(&y_test, &pred, Some(&proba));
        acc_scores.push(metrics.accuracy);
        f1_scores.push(metrics.f1);
        auc_scores.push(metrics.auc_roc.unwrap_or(0.5));
    }

    Ok((
        acc_scores.iter().sum::<f64>() / acc_scores.len() as f64,
        f1_scores.iter().sum::<f64>() / f1_scores.len() as f64,
        auc_scores.iter().sum::<f64>() / auc_scores.len() as f64,
    ))
}

fn get_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_cols = x.ncols();
    let data: Vec<f64> = indices.iter().flat_map(|&i| x.row(i).to_vec()).collect();
    Array2::from_shape_vec((indices.len(), n_cols), data).unwrap()
}

fn get_elements(y: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

fn backtest_regression(actual_returns: &[f64], predictions: &Array1<f64>) -> BacktestResult {
    let strategy_returns: Vec<f64> = actual_returns
        .iter()
        .zip(predictions.iter())
        .map(|(&ret, &pred)| ret * pred.signum())
        .collect();

    calculate_backtest_metrics(&strategy_returns)
}

fn backtest_classification(actual_returns: &[f64], predictions: &Array1<f64>) -> BacktestResult {
    let strategy_returns: Vec<f64> = actual_returns
        .iter()
        .zip(predictions.iter())
        .map(|(&ret, &pred)| if pred >= 0.5 { ret } else { 0.0 })
        .collect();

    calculate_backtest_metrics(&strategy_returns)
}

fn backtest_buy_hold(actual_returns: &[f64]) -> BacktestResult {
    calculate_backtest_metrics(actual_returns)
}

fn calculate_backtest_metrics(returns: &[f64]) -> BacktestResult {
    let total_return = returns.iter().map(|r| r + 1.0).product::<f64>() - 1.0;

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();

    // Annualized Sharpe (assuming 4-hour data = 6 periods per day)
    let sharpe_ratio = if std_dev > 1e-10 {
        (mean_return / std_dev) * (365.0 * 6.0_f64).sqrt()
    } else {
        0.0
    };

    // Max drawdown
    let mut peak = 1.0;
    let mut max_dd = 0.0;
    let mut equity = 1.0;

    for &ret in returns {
        equity *= 1.0 + ret;
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    // Win rate
    let n_trades = returns.iter().filter(|&&r| r != 0.0).count();
    let winning = returns.iter().filter(|&&r| r > 0.0).count();
    let win_rate = if n_trades > 0 {
        winning as f64 / n_trades as f64
    } else {
        0.0
    };

    BacktestResult {
        total_return,
        sharpe_ratio,
        max_drawdown: max_dd,
        win_rate,
        n_trades,
    }
}
