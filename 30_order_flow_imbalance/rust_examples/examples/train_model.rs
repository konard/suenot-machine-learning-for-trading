//! # Train Model Example
//!
//! Demonstrates training a gradient boosting model on synthetic data.
//!
//! Run with: `cargo run --example train_model`

use anyhow::Result;
use chrono::Utc;
use order_flow_imbalance::data::snapshot::FeatureVector;
use order_flow_imbalance::models::gradient_boosting::GradientBoostingModel;
use order_flow_imbalance::metrics::classification::{ClassificationMetrics, auc_roc};
use rand::Rng;

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              Model Training Example                        ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Generate synthetic training data
    println!("Generating synthetic trading data...");
    let (train_data, test_data) = generate_synthetic_data(1000, 200);
    println!("  Training samples: {}", train_data.len());
    println!("  Test samples:     {}", test_data.len());
    println!();

    // Train model
    println!("Training gradient boosting model...");
    println!("  Trees:         100");
    println!("  Learning rate: 0.1");
    println!();

    let mut model = GradientBoostingModel::new(0.1);
    model.train(&train_data, 100, 1);

    println!("Training complete!");
    println!("  Trees trained: {}", model.n_trees());
    println!();

    // Evaluate on test set
    println!("═══════════════════════════════════════════════════════════");
    println!("              MODEL EVALUATION                              ");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let predictions: Vec<i32> = test_data.iter().map(|f| model.predict(f)).collect();
    let probabilities: Vec<f64> = test_data.iter().map(|f| model.predict_proba(f)).collect();
    let labels: Vec<i32> = test_data
        .iter()
        .map(|f| if f.label.unwrap_or(0.0) > 0.5 { 1 } else { 0 })
        .collect();

    let metrics = ClassificationMetrics::from_predictions(&predictions, &labels);
    let auc = auc_roc(&probabilities, &labels);

    println!("Classification Metrics:");
    println!("───────────────────────────────────────────────────────────");
    println!("  Accuracy:    {:.2}%", metrics.accuracy() * 100.0);
    println!("  Precision:   {:.2}%", metrics.precision() * 100.0);
    println!("  Recall:      {:.2}%", metrics.recall() * 100.0);
    println!("  F1 Score:    {:.2}%", metrics.f1_score() * 100.0);
    println!("  Specificity: {:.2}%", metrics.specificity() * 100.0);
    println!("  AUC-ROC:     {:.4}", auc);
    println!();

    // Confusion matrix
    println!("Confusion Matrix:");
    println!("───────────────────────────────────────────────────────────");
    println!("                    Predicted");
    println!("                   0        1");
    println!("  Actual  0      {:>4}     {:>4}", metrics.tn, metrics.fp);
    println!("          1      {:>4}     {:>4}", metrics.fn_, metrics.tp);
    println!();

    // Feature importance
    println!("Feature Importance:");
    println!("───────────────────────────────────────────────────────────");
    let top_features = model.top_features(10);
    for (i, (name, importance)) in top_features.iter().enumerate() {
        let bar_len = (importance * 50.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  {:>2}. {:20} {:.4} {}", i + 1, name, importance, bar);
    }
    println!();

    // Save model
    println!("Saving model to JSON...");
    let json = model.to_json()?;
    println!("  Model size: {} bytes", json.len());
    println!();

    // Test single prediction
    println!("═══════════════════════════════════════════════════════════");
    println!("              SAMPLE PREDICTION                             ");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let mut sample = FeatureVector::new(Utc::now());
    sample.add("ofi_zscore", 2.5);
    sample.add("depth_imbalance", 0.3);
    sample.add("spread_bps", 5.0);
    sample.add("trade_imbalance", 0.4);
    sample.add("momentum_1min", 0.1);

    let prob = model.predict_proba(&sample);
    let pred = model.predict(&sample);

    println!("  Input Features:");
    for (name, value) in sample.names.iter().zip(sample.values.iter()) {
        println!("    {:20} = {:>8.4}", name, value);
    }
    println!();
    println!("  Prediction:");
    println!("    Probability(Up): {:.4}", prob);
    println!("    Class:           {} ({})",
        pred,
        if pred == 1 { "PRICE UP" } else { "PRICE DOWN" }
    );
    println!();

    Ok(())
}

/// Generate synthetic trading data for training
fn generate_synthetic_data(train_size: usize, test_size: usize) -> (Vec<FeatureVector>, Vec<FeatureVector>) {
    let mut rng = rand::thread_rng();

    let generate_sample = || {
        let mut fv = FeatureVector::new(Utc::now());

        // Generate features with some correlation to label
        let ofi_zscore: f64 = rng.gen_range(-3.0..3.0);
        let depth_imbalance: f64 = rng.gen_range(-1.0..1.0);
        let spread_bps: f64 = rng.gen_range(1.0..20.0);
        let trade_imbalance: f64 = rng.gen_range(-1.0..1.0);
        let momentum: f64 = rng.gen_range(-1.0..1.0);
        let vpin: f64 = rng.gen_range(0.0..1.0);

        // Generate label with correlation to features
        let signal_strength = 0.3 * ofi_zscore
            + 0.2 * depth_imbalance
            + 0.2 * trade_imbalance
            + 0.15 * momentum
            - 0.1 * spread_bps / 10.0
            + rng.gen_range(-1.0..1.0); // noise

        let label = if signal_strength > 0.0 { 1.0 } else { 0.0 };

        fv.add("ofi_zscore", ofi_zscore);
        fv.add("depth_imbalance", depth_imbalance);
        fv.add("spread_bps", spread_bps);
        fv.add("trade_imbalance", trade_imbalance);
        fv.add("momentum_1min", momentum);
        fv.add("vpin", vpin);
        fv.set_label(label);

        fv
    };

    let train: Vec<_> = (0..train_size).map(|_| generate_sample()).collect();
    let test: Vec<_> = (0..test_size).map(|_| generate_sample()).collect();

    (train, test)
}
