//! Basic Transfer Learning Example
//!
//! Demonstrates the core transfer learning pipeline:
//! 1. Generate synthetic source and target domain data
//! 2. Pre-train a model on the source domain
//! 3. Fine-tune on the target domain
//! 4. Compare performance with and without transfer

use transfer_learning_trading::{
    TransferNetwork, FineTuneStrategy,
    TransferTradingStrategy, TradingSignal,
};
use transfer_learning_trading::data::{StockDataLoader, DataFeatureExtractor};
use transfer_learning_trading::training::{TransferTrainer, TrainingConfig, AdaptationConfig};
use transfer_learning_trading::utils::{ConfusionMatrix, Metrics};
use transfer_learning_trading::network::AdaptationMethod;

fn main() {
    println!("=== Transfer Learning for Trading - Basic Example ===\n");

    // Step 1: Generate synthetic market data
    println!("Step 1: Generating synthetic market data...");
    let loader = StockDataLoader::new();

    // Source domain: stable market with moderate volatility
    let source_bars = loader.generate_synthetic(500, 100.0, 0.02);
    println!("  Source domain: {} bars (stable market)", source_bars.len());

    // Target domain: volatile market with different characteristics
    let target_bars = loader.generate_synthetic(100, 50.0, 0.04);
    println!("  Target domain: {} bars (volatile market)", target_bars.len());

    // Step 2: Extract features
    println!("\nStep 2: Extracting features...");
    let feature_extractor = DataFeatureExtractor::new(20);

    let source_features = feature_extractor.extract_features(&source_bars);
    let source_labels = feature_extractor.generate_labels(&source_bars, 3, 0.005);
    println!(
        "  Source features: {} samples x {} features",
        source_features.nrows(),
        source_features.ncols()
    );

    let target_features = feature_extractor.extract_features(&target_bars);
    let target_labels = feature_extractor.generate_labels(&target_bars, 3, 0.005);
    println!(
        "  Target features: {} samples x {} features",
        target_features.nrows(),
        target_features.ncols()
    );

    // Ensure labels match feature count
    let n_source = source_features.nrows().min(source_labels.len());
    let n_target = target_features.nrows().min(target_labels.len());
    let source_labels = &source_labels[..n_source];
    let target_labels = &target_labels[..n_target];
    let source_features = source_features.slice(ndarray::s![..n_source, ..]).to_owned();
    let target_features = target_features.slice(ndarray::s![..n_target, ..]).to_owned();

    if n_source == 0 || n_target == 0 {
        println!("Not enough data to train. Please increase bar count.");
        return;
    }

    // Step 3: Baseline - Train from scratch on target
    println!("\nStep 3: Baseline - Training from scratch on target domain...");
    let baseline_network = TransferNetwork::new(
        feature_extractor.feature_dim(), 64, 32, false,
    );
    let mut baseline_trainer = TransferTrainer::new(baseline_network)
        .with_pretrain_config(TrainingConfig {
            epochs: 20,
            learning_rate: 0.01,
            ..Default::default()
        });
    let baseline_result = baseline_trainer.pretrain(&target_features, target_labels);
    let baseline_predictions = baseline_trainer.network().predict(&target_features);
    let baseline_cm = ConfusionMatrix::new(&baseline_predictions, target_labels, 3);
    println!("  Baseline accuracy: {:.2}%", baseline_cm.accuracy() * 100.0);
    println!("  Epochs trained: {}", baseline_result.epochs_trained);

    // Step 4: Transfer Learning
    println!("\nStep 4: Transfer Learning pipeline...");
    let transfer_network = TransferNetwork::new(
        feature_extractor.feature_dim(), 64, 32, true,
    );
    let mut transfer_trainer = TransferTrainer::new(transfer_network)
        .with_pretrain_config(TrainingConfig {
            epochs: 30,
            learning_rate: 0.01,
            ..Default::default()
        })
        .with_adapt_config(AdaptationConfig {
            epochs: 10,
            learning_rate: 0.001,
            method: AdaptationMethod::MMD,
            strategy: FineTuneStrategy::PartialFineTune,
            ..Default::default()
        });

    println!("  Pre-training on source domain...");
    let pretrain_result = transfer_trainer.pretrain(&source_features, source_labels);
    println!(
        "  Pre-training done: {} epochs, final loss: {:.4}",
        pretrain_result.epochs_trained, pretrain_result.final_loss
    );

    println!("  Adapting to target domain...");
    let adapt_result = transfer_trainer.adapt(
        &source_features,
        &target_features,
        Some(target_labels),
    );
    println!(
        "  Adaptation done: {} epochs, final loss: {:.6}",
        adapt_result.epochs_trained, adapt_result.final_loss
    );

    // Step 5: Evaluate
    println!("\nStep 5: Evaluation...");
    let source_predictions = transfer_trainer.network().predict(&source_features);
    let target_predictions = transfer_trainer.network().predict(&target_features);

    let source_cm = ConfusionMatrix::new(&source_predictions, source_labels, 3);
    let target_cm = ConfusionMatrix::new(&target_predictions, target_labels, 3);

    let domain_similarity = transfer_trainer.compute_domain_similarity(&source_features, &target_features);

    let metrics = Metrics::compute(
        &source_predictions,
        source_labels,
        &target_predictions,
        target_labels,
        baseline_cm.accuracy(),
        domain_similarity,
        adapt_result.final_loss,
    );

    println!("{}", source_cm);
    println!("{}", target_cm);
    println!("{}", metrics);

    // Step 6: Trading simulation
    println!("Step 6: Trading simulation...");
    let mut strategy = TransferTradingStrategy::new(100_000.0)
        .with_confidence_threshold(0.5)
        .with_stop_loss(0.02);

    let mut signals_count = [0usize; 3]; // Buy, Sell, Hold

    for i in 0..n_target {
        let proba = transfer_trainer.network().predict_proba(
            &target_features.slice(ndarray::s![i..i+1, ..]).to_owned(),
        );
        let price = target_bars[i + 20].close; // Offset by lookback
        let signal = strategy.generate_signal(&proba.row(0).to_owned(), price, domain_similarity);

        match signal {
            TradingSignal::Buy => signals_count[0] += 1,
            TradingSignal::Sell => signals_count[1] += 1,
            TradingSignal::Hold => signals_count[2] += 1,
        }
    }

    println!("  Signals: Buy={}, Sell={}, Hold={}", signals_count[0], signals_count[1], signals_count[2]);
    println!("  Final portfolio: ${:.2}", strategy.portfolio_value());

    let summary = strategy.performance_summary();
    println!("{}", summary);

    println!("\n=== Example Complete ===");
}
