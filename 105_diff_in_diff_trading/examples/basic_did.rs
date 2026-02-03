//! Basic Difference-in-Differences example.
//!
//! Demonstrates how to:
//! - Generate synthetic panel data
//! - Estimate DiD treatment effect
//! - Test parallel trends assumption
//! - Generate trading signals

use diff_in_diff_trading::model::{generate_synthetic_panel, DiDEstimator};
use diff_in_diff_trading::trading::SignalGenerator;

fn main() {
    println!("=== Difference-in-Differences: Basic Example ===\n");

    // Generate synthetic panel data with a known treatment effect
    let treatment_effect = 0.05; // 5% treatment effect
    let data = generate_synthetic_panel(
        20,                // 20 treated units
        20,                // 20 control units
        10,                // 10 pre-treatment periods
        10,                // 10 post-treatment periods
        treatment_effect,  // True treatment effect
        0.002,             // Small positive trend
    );

    println!("Generated {} observations", data.len());
    println!(
        "  - Treated units: {}",
        data.iter().filter(|d| d.treated).count() / 20
    );
    println!(
        "  - Control units: {}",
        data.iter().filter(|d| !d.treated).count() / 20
    );
    println!("  - Time periods: 20 (10 pre, 10 post)");
    println!("  - True treatment effect: {:.2}%\n", treatment_effect * 100.0);

    // Create DiD estimator
    let estimator = DiDEstimator::new()
        .with_bootstrap(500)
        .with_confidence(0.95);

    // Estimate treatment effect
    println!("Estimating treatment effect...\n");
    let results = estimator.fit(&data);
    println!("{}\n", results);

    // Check if estimate is close to true value
    let estimate_error = (results.did_estimate - treatment_effect).abs();
    println!("Estimation accuracy:");
    println!("  - True effect:      {:.4}", treatment_effect);
    println!("  - Estimated effect: {:.4}", results.did_estimate);
    println!("  - Absolute error:   {:.4}", estimate_error);

    if results.ci_lower < treatment_effect && results.ci_upper > treatment_effect {
        println!("  - True effect is within 95% confidence interval: YES");
    } else {
        println!("  - True effect is within 95% confidence interval: NO");
    }
    println!();

    // Test parallel trends assumption
    println!("Testing parallel trends assumption...");
    let pre_trend_coefficients = estimator.test_parallel_trends(&data, 5);

    let max_pre_coef = pre_trend_coefficients
        .values()
        .map(|v| v.abs())
        .fold(0.0, f64::max);

    println!("Pre-treatment coefficients (should be close to 0):");
    for (t, coef) in &pre_trend_coefficients {
        let stars = if coef.abs() > 0.02 { "(!)" } else { "" };
        println!("  t={}: {:.4} {}", t, coef, stars);
    }
    println!();

    if max_pre_coef < 0.03 {
        println!("Parallel trends assumption: SATISFIED (max |coef| = {:.4})", max_pre_coef);
    } else {
        println!("Parallel trends assumption: QUESTIONABLE (max |coef| = {:.4})", max_pre_coef);
    }
    println!();

    // Generate trading signal
    println!("Generating trading signal...");
    let signal_generator = SignalGenerator::new()
        .with_threshold(0.02)
        .with_significance(0.05);

    let signal = signal_generator.generate(&results);
    let position_size = signal_generator.compute_position_size(&results, 0.15, 100000.0);

    println!("  - Signal: {:?}", signal);
    println!("  - Recommended position size: ${:.2}", position_size);
    println!("  - Stop-loss (2x SE): {:.2}%", signal_generator.compute_stop_loss(&results, 2.0) * 100.0);
    println!("  - Take-profit (3x effect): {:.2}%", signal_generator.compute_take_profit(&results, 3.0) * 100.0);
    println!();

    // Statistical significance
    if results.is_significant(0.05) {
        println!("Result is statistically significant at 5% level.");
    } else {
        println!("Result is NOT statistically significant at 5% level.");
    }

    if results.is_significant(0.01) {
        println!("Result is statistically significant at 1% level.");
    }
}
