//! Domain Adaptation Example
//!
//! Demonstrates different domain adaptation methods:
//! - MMD (Maximum Mean Discrepancy)
//! - CORAL (Correlation Alignment)
//! - Adversarial domain adaptation
//! - No adaptation (baseline)
//!
//! Compares their effectiveness on synthetic financial data
//! with different domain shifts.

use transfer_learning_trading::network::{
    TransferNetwork, DomainAdapter, AdaptationMethod, FineTuneStrategy,
};
use transfer_learning_trading::data::{StockDataLoader, DataFeatureExtractor};
use transfer_learning_trading::training::{TransferTrainer, TrainingConfig, AdaptationConfig};
use transfer_learning_trading::utils::ConfusionMatrix;
use ndarray::Axis;

fn main() {
    println!("=== Domain Adaptation Comparison ===\n");

    // Generate source and target data with a distribution shift
    let loader = StockDataLoader::new();
    let source_bars = loader.generate_synthetic(400, 100.0, 0.015);
    let target_bars = loader.generate_synthetic(100, 200.0, 0.035);

    let feat_ext = DataFeatureExtractor::new(15);
    let source_features = feat_ext.extract_features(&source_bars);
    let source_labels = feat_ext.generate_labels(&source_bars, 3, 0.005);
    let target_features = feat_ext.extract_features(&target_bars);
    let target_labels = feat_ext.generate_labels(&target_bars, 3, 0.005);

    let n_source = source_features.nrows().min(source_labels.len());
    let n_target = target_features.nrows().min(target_labels.len());
    let source_labels = &source_labels[..n_source];
    let target_labels = &target_labels[..n_target];
    let source_features = source_features.slice(ndarray::s![..n_source, ..]).to_owned();
    let target_features = target_features.slice(ndarray::s![..n_target, ..]).to_owned();

    if n_source == 0 || n_target == 0 {
        println!("Not enough data. Increase bar count.");
        return;
    }

    println!("Source: {} samples, Target: {} samples", n_source, n_target);
    println!("Feature dimension: {}\n", feat_ext.feature_dim());

    // Compute initial domain statistics
    let source_mean = source_features.mean_axis(Axis(0)).unwrap();
    let target_mean = target_features.mean_axis(Axis(0)).unwrap();
    let mean_diff: f64 = (&source_mean - &target_mean).iter().map(|x| x * x).sum::<f64>().sqrt();
    println!("Initial domain distance (L2 of means): {:.4}\n", mean_diff);

    // Test different adaptation methods
    let methods = vec![
        ("No Adaptation", AdaptationMethod::None),
        ("MMD", AdaptationMethod::MMD),
        ("CORAL", AdaptationMethod::CORAL),
        ("Adversarial", AdaptationMethod::Adversarial),
    ];

    println!("{:<20} {:<15} {:<15} {:<15}",
        "Method", "Source Acc", "Target Acc", "Adapt Loss");
    println!("{}", "-".repeat(65));

    for (name, method) in &methods {
        let network = TransferNetwork::new(feat_ext.feature_dim(), 64, 32, *method != AdaptationMethod::None)
            .with_adaptation_method(*method);

        let mut trainer = TransferTrainer::new(network)
            .with_pretrain_config(TrainingConfig {
                epochs: 20,
                learning_rate: 0.01,
                ..Default::default()
            })
            .with_adapt_config(AdaptationConfig {
                epochs: 10,
                learning_rate: 0.001,
                method: *method,
                strategy: FineTuneStrategy::PartialFineTune,
                ..Default::default()
            });

        // Pre-train
        trainer.pretrain(&source_features, source_labels);

        // Adapt
        let adapt_result = trainer.adapt(
            &source_features,
            &target_features,
            Some(target_labels),
        );

        // Evaluate
        let source_preds = trainer.network().predict(&source_features);
        let target_preds = trainer.network().predict(&target_features);
        let source_cm = ConfusionMatrix::new(&source_preds, source_labels, 3);
        let target_cm = ConfusionMatrix::new(&target_preds, target_labels, 3);

        println!(
            "{:<20} {:<15.2}% {:<15.2}% {:<15.6}",
            name,
            source_cm.accuracy() * 100.0,
            target_cm.accuracy() * 100.0,
            adapt_result.final_loss,
        );
    }

    // Detailed comparison: MMD vs CORAL domain alignment
    println!("\n--- Domain Alignment Analysis ---\n");

    let mmd_adapter = DomainAdapter::new(AdaptationMethod::MMD);
    let coral_adapter = DomainAdapter::new(AdaptationMethod::CORAL);

    let mmd_loss = mmd_adapter.compute_loss(&source_features, &target_features);
    let coral_loss = coral_adapter.compute_loss(&source_features, &target_features);
    let mmd_sim = mmd_adapter.domain_similarity(&source_features, &target_features);

    println!("MMD loss (mean discrepancy):    {:.6}", mmd_loss);
    println!("CORAL loss (covariance align):  {:.6}", coral_loss);
    println!("Domain similarity (MMD-based):  {:.4}", mmd_sim);

    // Test fine-tuning strategies
    println!("\n--- Fine-Tuning Strategy Comparison ---\n");

    let strategies = vec![
        ("Feature Extraction", FineTuneStrategy::FeatureExtraction),
        ("Partial Fine-Tune", FineTuneStrategy::PartialFineTune),
        ("Full Fine-Tune", FineTuneStrategy::FullFineTune),
    ];

    println!("{:<25} {:<15}", "Strategy", "Target Accuracy");
    println!("{}", "-".repeat(40));

    for (name, strategy) in &strategies {
        let network = TransferNetwork::new(feat_ext.feature_dim(), 64, 32, true);
        let mut trainer = TransferTrainer::new(network)
            .with_pretrain_config(TrainingConfig {
                epochs: 20,
                learning_rate: 0.01,
                ..Default::default()
            })
            .with_adapt_config(AdaptationConfig {
                epochs: 10,
                learning_rate: 0.001,
                strategy: *strategy,
                ..Default::default()
            });

        trainer.pretrain(&source_features, source_labels);
        trainer.adapt(&source_features, &target_features, Some(target_labels));

        let target_preds = trainer.network().predict(&target_features);
        let target_cm = ConfusionMatrix::new(&target_preds, target_labels, 3);

        println!("{:<25} {:.2}%", name, target_cm.accuracy() * 100.0);
    }

    println!("\n=== Domain Adaptation Example Complete ===");
}
