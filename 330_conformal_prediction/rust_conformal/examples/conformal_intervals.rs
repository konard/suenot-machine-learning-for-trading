//! Example: Conformal prediction intervals

use conformal_trading::conformal::predictor::SplitConformalPredictor;
use conformal_trading::conformal::adaptive::AdaptiveConformalPredictor;
use conformal_trading::conformal::scores::NonConformityScore;
use ndarray::Array1;
use rand::Rng;

fn main() {
    println!("Conformal Prediction - Interval Demo");
    println!("{}", "=".repeat(50));

    // Generate synthetic data
    println!("\nGenerating synthetic regression data...");
    let mut rng = rand::thread_rng();

    let n_cal = 200;
    let n_test = 100;

    // Calibration data
    let x_cal: Vec<f64> = (0..n_cal).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let y_cal: Vec<f64> = x_cal.iter()
        .map(|&x| 2.0 * x + 1.0 + rng.gen_range(-1.0..1.0))
        .collect();
    let y_pred_cal: Vec<f64> = x_cal.iter()
        .map(|&x| 2.0 * x + 1.0) // Perfect prediction (for demo)
        .collect();

    // Test data
    let x_test: Vec<f64> = (0..n_test).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let y_test: Vec<f64> = x_test.iter()
        .map(|&x| 2.0 * x + 1.0 + rng.gen_range(-1.0..1.0))
        .collect();
    let y_pred_test: Vec<f64> = x_test.iter()
        .map(|&x| 2.0 * x + 1.0)
        .collect();

    // Create arrays
    let y_cal_arr = Array1::from_vec(y_cal);
    let y_pred_cal_arr = Array1::from_vec(y_pred_cal);
    let y_test_arr = Array1::from_vec(y_test);
    let y_pred_test_arr = Array1::from_vec(y_pred_test);

    // ==========================================================
    // Split Conformal Prediction
    // ==========================================================
    println!("\n{}", "-".repeat(50));
    println!("Split Conformal Prediction (90% coverage)");
    println!("{}", "-".repeat(50));

    let mut cp = SplitConformalPredictor::new(0.1); // 10% miscoverage = 90% coverage
    cp.calibrate(&y_cal_arr, &y_pred_cal_arr);

    println!("Calibration complete!");
    println!("  Quantile: {:.4}", cp.quantile().unwrap());

    // Evaluate on test set
    let metrics = cp.evaluate_coverage(&y_test_arr, &y_pred_test_arr).unwrap();
    println!("\nTest Set Evaluation:");
    println!("  {}", metrics);

    // Show some example predictions
    println!("\nSample Predictions:");
    println!("{:<10} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "Index", "Prediction", "Lower", "Upper", "Actual", "Covered");
    println!("{}", "-".repeat(70));

    for i in 0..10 {
        let interval = cp.predict_interval(y_pred_test_arr[i]).unwrap();
        let covered = interval.covers(y_test_arr[i]);
        println!("{:<10} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>8}",
            i,
            interval.point,
            interval.lower,
            interval.upper,
            y_test_arr[i],
            if covered { "Yes" } else { "No" }
        );
    }

    // ==========================================================
    // Different Alpha Values
    // ==========================================================
    println!("\n{}", "-".repeat(50));
    println!("Coverage vs. Interval Width Trade-off");
    println!("{}", "-".repeat(50));

    for alpha in [0.05, 0.10, 0.20, 0.30] {
        let mut cp_alpha = SplitConformalPredictor::new(alpha);
        cp_alpha.calibrate(&y_cal_arr, &y_pred_cal_arr);

        let metrics = cp_alpha.evaluate_coverage(&y_test_arr, &y_pred_test_arr).unwrap();

        println!("\nAlpha = {:.2} (Target coverage: {:.0}%)", alpha, (1.0 - alpha) * 100.0);
        println!("  Empirical coverage: {:.1}%", metrics.coverage * 100.0);
        println!("  Mean interval width: {:.4}", metrics.mean_width);
    }

    // ==========================================================
    // Different Score Functions
    // ==========================================================
    println!("\n{}", "-".repeat(50));
    println!("Different Non-conformity Scores");
    println!("{}", "-".repeat(50));

    for score_type in [NonConformityScore::Absolute, NonConformityScore::Normalized] {
        let mut cp_score = SplitConformalPredictor::new(0.1)
            .with_score_type(score_type);
        cp_score.calibrate(&y_cal_arr, &y_pred_cal_arr);

        let metrics = cp_score.evaluate_coverage(&y_test_arr, &y_pred_test_arr).unwrap();

        println!("\nScore type: {:?}", score_type);
        println!("  Coverage: {:.1}%", metrics.coverage * 100.0);
        println!("  Width: {:.4}", metrics.mean_width);
    }

    // ==========================================================
    // Adaptive Conformal Prediction
    // ==========================================================
    println!("\n{}", "-".repeat(50));
    println!("Adaptive Conformal Inference");
    println!("{}", "-".repeat(50));

    let mut acp = AdaptiveConformalPredictor::new(0.1, 0.01, 100);

    // Calibrate
    let scores: Vec<f64> = y_cal_arr.iter()
        .zip(y_pred_cal_arr.iter())
        .map(|(&y, &yp)| (y - yp).abs())
        .collect();
    acp.calibrate(scores);

    // Simulate online predictions with distribution shift
    println!("\nSimulating online predictions with distribution shift...");

    let mut cumulative_coverage = Vec::new();

    for (i, (&y_true, &y_pred)) in y_test_arr.iter().zip(y_pred_test_arr.iter()).enumerate() {
        // Add shift after midpoint
        let shifted_pred = if i > n_test / 2 {
            y_pred + 0.5 // Systematic bias after midpoint
        } else {
            y_pred
        };

        if let Some(interval) = acp.predict_interval(shifted_pred) {
            acp.update(y_true, &interval);
            cumulative_coverage.push(acp.rolling_coverage(20));
        }
    }

    let stats = acp.compute_stats();
    println!("\nAdaptive Stats:");
    println!("  Mean alpha: {:.4}", stats.mean_alpha);
    println!("  Std alpha: {:.4}", stats.std_alpha);
    println!("  Overall coverage: {:.1}%", stats.overall_coverage * 100.0);
    println!("  Updates: {}", stats.num_updates);

    println!("\nRolling coverage (last 20):");
    println!("  Before shift (sample): {:.1}%", cumulative_coverage[n_test / 4] * 100.0);
    println!("  After shift (sample): {:.1}%", cumulative_coverage.last().unwrap_or(&0.0) * 100.0);

    println!("\nDemo complete!");
}
