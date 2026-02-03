//! Basic Synthetic Control Method example.
//!
//! Demonstrates fitting an SCM model on simple synthetic data.

use synthetic_control_trading::SyntheticControlModel;

fn main() {
    println!("Synthetic Control Method - Basic Example");
    println!("=========================================\n");

    // Generate sample pre-treatment data
    // Treated unit follows a trend
    let treated_pre: Vec<f64> = (0..60)
        .map(|i| 0.01 * (i as f64 / 10.0).sin() + 0.002 * (i as f64 - 30.0))
        .collect();

    // Donors follow similar patterns with noise
    let donors_pre: Vec<Vec<f64>> = (0..5)
        .map(|j| {
            treated_pre
                .iter()
                .enumerate()
                .map(|(i, &t)| t + 0.005 * ((j * i) as f64).sin() + (j as f64 - 2.0) * 0.001)
                .collect()
        })
        .collect();

    // Fit the model
    let mut model = SyntheticControlModel::new();
    model
        .fit_with_labels(
            &treated_pre,
            &donors_pre,
            vec![
                "MSFT".to_string(),
                "GOOGL".to_string(),
                "META".to_string(),
                "AMZN".to_string(),
                "NVDA".to_string(),
            ],
        )
        .expect("Failed to fit model");

    println!("Model fitted successfully!");
    println!(
        "Pre-treatment RMSE: {:.6}",
        model.pre_treatment_rmse().unwrap()
    );

    // Print weights
    println!("\nDonor weights:");
    if let Some(weights_map) = model.weights_map() {
        for (label, weight) in weights_map {
            if weight > 0.01 {
                println!("  {}: {:.4}", label, weight);
            }
        }
    }

    // Generate post-treatment data with treatment effect
    let treatment_effect = 0.03; // 3% cumulative effect
    let treated_post: Vec<f64> = (60..80)
        .map(|i| {
            0.01 * (i as f64 / 10.0).sin()
                + 0.002 * (i as f64 - 30.0)
                + treatment_effect / 20.0 // Spread effect over 20 days
        })
        .collect();

    // Donors continue without treatment effect
    let donors_post: Vec<Vec<f64>> = (0..5)
        .map(|j| {
            (60..80)
                .map(|i| {
                    0.01 * (i as f64 / 10.0).sin()
                        + 0.002 * (i as f64 - 30.0)
                        + 0.005 * ((j * i) as f64).sin()
                        + (j as f64 - 2.0) * 0.001
                })
                .collect()
        })
        .collect();

    // Estimate treatment effects
    let effects = model
        .estimate_effects(&treated_post, &donors_post)
        .expect("Failed to estimate effects");

    println!("\nTreatment Effects:");
    println!("  Cumulative effect: {:.4}", effects.cumulative);
    println!("  CAR: {:.4}", effects.car);

    // Run placebo tests
    println!("\nRunning placebo tests...");
    let placebo_results = model
        .run_placebo_tests(&treated_post, &donors_post)
        .expect("Failed to run placebo tests");

    println!(
        "  P-value: {:.4} (rank {}/{})",
        placebo_results.p_value, placebo_results.rank, placebo_results.n_placebos + 1
    );

    if placebo_results.p_value < 0.1 {
        println!("  Treatment effect is statistically significant at 10% level!");
    }

    println!("\nExample completed successfully!");
}
