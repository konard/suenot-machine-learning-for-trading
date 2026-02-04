//! Example: Train ensemble models with uncertainty

use ensemble_uncertainty_trading::prelude::*;
use ensemble_uncertainty_trading::ensemble::traits::EnsembleModel;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;

fn main() -> anyhow::Result<()> {
    println!("=".repeat(60));
    println!("Training Ensemble Models with Uncertainty");
    println!("=".repeat(60));

    // Generate synthetic data
    println!("\n--- Generating Synthetic Data ---");
    let n_samples = 500;
    let n_features = 10;

    let x = Array2::random((n_samples, n_features), Uniform::new(-1.0, 1.0));

    // Target: linear combination + noise
    let y: Array1<f64> = x.column(0).to_owned() * 2.0
        + x.column(1).to_owned() * 1.5
        + x.column(2).to_owned() * 0.5
        + Array1::random(n_samples, Uniform::new(-0.3, 0.3));

    // Split data
    let train_size = (n_samples as f64 * 0.8) as usize;
    let x_train = x.slice(ndarray::s![..train_size, ..]).to_owned();
    let y_train = y.slice(ndarray::s![..train_size]).to_owned();
    let x_test = x.slice(ndarray::s![train_size.., ..]).to_owned();
    let y_test = y.slice(ndarray::s![train_size..]).to_owned();

    println!("Training samples: {}", x_train.nrows());
    println!("Test samples: {}", x_test.nrows());
    println!("Features: {}", x_train.ncols());

    // Train Random Forest
    println!("\n--- Training Random Forest ---");
    let mut rf = RandomForestEnsemble::new(50, Some(10));
    rf.fit(&x_train, &y_train)?;
    println!("Random Forest trained with {} trees", rf.n_estimators());

    // Get predictions with uncertainty
    let (rf_preds, rf_unc) = rf.predict_with_uncertainty(&x_test)?;
    println!("Mean prediction: {:.4}", rf_preds.mean().unwrap_or(0.0));
    println!("Mean uncertainty: {:.4}", rf_unc.mean().unwrap_or(0.0));

    // Train Gradient Boosting
    println!("\n--- Training Gradient Boosting ---");
    let mut gb = GradientBoostingEnsemble::new(50, true);
    gb.fit(&x_train, &y_train)?;
    println!("Gradient Boosting trained");

    let (gb_preds, gb_unc) = gb.predict_with_uncertainty(&x_test)?;
    println!("Mean prediction: {:.4}", gb_preds.mean().unwrap_or(0.0));
    println!("Mean uncertainty: {:.4}", gb_unc.mean().unwrap_or(0.0));

    // Train Stacking Ensemble
    println!("\n--- Training Stacking Ensemble ---");
    let mut stack = StackingEnsemble::new(40);
    stack.fit(&x_train, &y_train)?;
    println!("Stacking Ensemble trained");

    let (stack_preds, stack_unc) = stack.predict_with_uncertainty(&x_test)?;
    println!("Mean prediction: {:.4}", stack_preds.mean().unwrap_or(0.0));
    println!("Mean uncertainty: {:.4}", stack_unc.mean().unwrap_or(0.0));

    // Compare predictions
    println!("\n--- Model Comparison ---");
    println!("{:<20} {:>12} {:>12} {:>12}", "Model", "MSE", "Mean Unc.", "Corr w/ Err");

    for (name, preds, unc) in [
        ("Random Forest", &rf_preds, &rf_unc),
        ("Gradient Boosting", &gb_preds, &gb_unc),
        ("Stacking", &stack_preds, &stack_unc),
    ] {
        let errors: Array1<f64> = &y_test - preds;
        let mse: f64 = errors.mapv(|e| e.powi(2)).mean().unwrap_or(0.0);
        let mean_unc: f64 = unc.mean().unwrap_or(0.0);

        // Correlation between uncertainty and absolute error
        let abs_errors: Array1<f64> = errors.mapv(|e| e.abs());
        let corr = correlation(&unc, &abs_errors);

        println!("{:<20} {:>12.4} {:>12.4} {:>12.4}", name, mse, mean_unc, corr);
    }

    // Analyze uncertainty quality
    println!("\n--- Uncertainty Analysis ---");
    let quantifier = UncertaintyQuantifier::new();

    // Create predictions matrix (models x samples)
    let all_preds = Array2::from_shape_fn((3, x_test.nrows()), |(i, j)| {
        match i {
            0 => rf_preds[j],
            1 => gb_preds[j],
            _ => stack_preds[j],
        }
    });

    let metrics = quantifier.compute_all_metrics(&all_preds);
    println!("Combined ensemble metrics:");
    println!("  Mean std across models: {:.4}", metrics.std.mean().unwrap_or(0.0));
    println!("  Mean IQR: {:.4}", metrics.iqr.mean().unwrap_or(0.0));
    println!("  Mean CI width (95%): {:.4}", metrics.ci_width().mean().unwrap_or(0.0));

    // Calibration check
    println!("\n--- Calibration Check ---");
    let mut calibrator = Calibrator::new(10);

    // Use Random Forest uncertainty for calibration
    let calib_result = calibrator.regression_calibration_error(&y_test, &rf_preds, &rf_unc);
    println!("Random Forest calibration:");
    println!("  Mean calibration error: {:.4}", calib_result.mean_calibration_error);
    println!("  Max calibration error: {:.4}", calib_result.max_calibration_error);

    // Calibrate uncertainty
    let scale = calibrator.calibrate_uncertainty(&y_test, &rf_preds, &rf_unc);
    println!("  Calibration scale factor: {:.4}", scale);

    println!("\n{}", "=".repeat(60));
    println!("Training complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}

/// Calculate Pearson correlation coefficient
fn correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}
