//! Counterfactual Trading Demo
//!
//! This example demonstrates the counterfactual estimation framework
//! for analyzing trading decisions.

use counterfactual_trading::{
    compute_max_drawdown, compute_sharpe_ratio, compute_win_rate, ATEResult,
    CounterfactualTradingStrategy, DoublyRobustEstimator, RegretMetrics, StrategyAttribution,
};
use ndarray::{Array1, Array2};

fn main() {
    println!("Counterfactual Trading Demo");
    println!("===========================\n");

    // Generate synthetic trading data
    let n_samples = 200;
    let n_features = 4;

    // Generate features (market indicators)
    let mut features_vec = Vec::with_capacity(n_samples * n_features);
    let mut treatment_vec = Vec::with_capacity(n_samples);
    let mut outcome_vec = Vec::with_capacity(n_samples);

    // Simple LCG random number generator
    let mut seed: u64 = 42;
    let lcg = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(1103515245).wrapping_add(12345);
        (*s as f64 / u64::MAX as f64) * 2.0 - 1.0
    };

    for i in 0..n_samples {
        // Features: returns, volatility, momentum, volume ratio
        let ret_1d = lcg(&mut seed) * 0.02;
        let ret_5d = lcg(&mut seed) * 0.05;
        let volatility = 0.01 + lcg(&mut seed).abs() * 0.02;
        let momentum = lcg(&mut seed) * 0.1;

        features_vec.push(ret_1d);
        features_vec.push(ret_5d);
        features_vec.push(volatility);
        features_vec.push(momentum);

        // Treatment: momentum-based trading (1 = trade, 0 = no trade)
        let treatment = if momentum > 0.0 { 1.0 } else { 0.0 };
        treatment_vec.push(treatment);

        // Outcome: return depends on treatment and some noise
        let base_return = ret_1d * 0.5 + momentum * 0.2;
        let treatment_effect = if treatment > 0.5 { 0.005 } else { 0.0 };
        let noise = lcg(&mut seed) * 0.01;
        let outcome = if treatment > 0.5 {
            base_return + treatment_effect + noise
        } else {
            0.0
        };
        outcome_vec.push(outcome);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), features_vec).unwrap();
    let treatment = Array1::from_vec(treatment_vec);
    let outcome = Array1::from_vec(outcome_vec);

    // Split into train and test
    let train_size = 150;

    let x_train = x.slice(ndarray::s![..train_size, ..]).to_owned();
    let treatment_train = treatment.slice(ndarray::s![..train_size]).to_owned();
    let outcome_train = outcome.slice(ndarray::s![..train_size]).to_owned();

    let x_test = x.slice(ndarray::s![train_size.., ..]).to_owned();
    let treatment_test = treatment.slice(ndarray::s![train_size..]).to_owned();
    let outcome_test = outcome.slice(ndarray::s![train_size..]).to_owned();

    println!("Data prepared:");
    println!("  Training samples: {}", train_size);
    println!("  Test samples: {}\n", n_samples - train_size);

    // Fit doubly robust estimator
    println!("Fitting doubly robust estimator...\n");
    let mut estimator = DoublyRobustEstimator::new();
    estimator.fit(&x_train, &treatment_train, &outcome_train);

    // Estimate Average Treatment Effect
    let ate = estimator.estimate_ate(&x_test, &treatment_test, &outcome_test);
    println!("Average Treatment Effect:");
    println!("  ATE: {:.6}", ate.ate);
    println!("  Standard Error: {:.6}", ate.se);
    println!("  95% CI: [{:.6}, {:.6}]\n", ate.ci_low, ate.ci_high);

    // Analyze individual counterfactuals
    println!("Sample Counterfactual Analysis:");
    for i in 0..5 {
        let cf = estimator.estimate_counterfactual(
            &x_test.row(i).to_owned(),
            treatment_test[i],
            outcome_test[i],
        );
        println!(
            "  Decision {}: Observed={:.6}, Counterfactual={:.6}, Effect={:.6}",
            i + 1,
            cf.observed_outcome,
            cf.counterfactual_outcome,
            cf.treatment_effect
        );
    }

    // Create trading strategy for evaluation
    println!("\nEvaluating trading strategy...\n");
    let mut strategy = CounterfactualTradingStrategy::new(estimator, 100);

    // Evaluate all test decisions
    for i in 0..x_test.nrows() {
        let action = if treatment_test[i] > 0.5 { 1 } else { 0 };
        strategy.evaluate_decision(&x_test.row(i).to_owned(), action, outcome_test[i], i as i64);
    }

    // Compute attribution
    let attribution = strategy.compute_attribution();
    println!("Strategy Attribution:");
    println!("  Total Return: {:.6}", attribution.total_return);
    println!("  Market Component: {:.6}", attribution.market_component);
    println!("  Strategy Alpha: {:.6}", attribution.strategy_alpha);
    println!(
        "  Alpha Contribution: {:.2}%\n",
        attribution.alpha_contribution_pct
    );

    // Compute regret
    let regret = strategy.compute_regret();
    println!("Regret Analysis:");
    println!("  Total Regret: {:.6}", regret.total_regret);
    println!("  Mean Regret: {:.6}", regret.mean_regret);
    println!("  Max Regret: {:.6}", regret.max_regret);
    println!("  Regret Frequency: {:.2}%\n", regret.regret_frequency * 100.0);

    // Compute performance metrics
    let returns: Vec<f64> = outcome_test.iter().cloned().collect();
    let sharpe = compute_sharpe_ratio(&returns, 252.0);
    let max_dd = compute_max_drawdown(&returns);
    let win_rate = compute_win_rate(&returns);

    println!("Performance Metrics:");
    println!("  Sharpe Ratio: {:.2}", sharpe);
    println!("  Max Drawdown: {:.2}%", max_dd * 100.0);
    println!("  Win Rate: {:.2}%\n", win_rate * 100.0);

    println!("Demo completed successfully!");
}
