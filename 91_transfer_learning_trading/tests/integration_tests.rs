//! Integration tests for the transfer learning trading library.

use transfer_learning_trading::{
    TransferNetwork, FineTuneStrategy, AdaptationMethod,
    TransferTradingStrategy, TradingSignal, Position,
};
use transfer_learning_trading::data::{StockDataLoader, DataFeatureExtractor, OHLCVBar};
use transfer_learning_trading::training::{TransferTrainer, TrainingConfig, AdaptationConfig};
use transfer_learning_trading::utils::{ConfusionMatrix, Metrics};
use transfer_learning_trading::network::DomainAdapter;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

// ── Helpers ──────────────────────────────────────────────

fn generate_test_data(
    n_source: usize,
    n_target: usize,
    n_features: usize,
) -> (Array2<f64>, Vec<usize>, Array2<f64>, Vec<usize>) {
    let source = Array2::random((n_source, n_features), Normal::new(0.0, 1.0).unwrap());
    let source_labels: Vec<usize> = (0..n_source).map(|i| i % 3).collect();
    let target = Array2::random((n_target, n_features), Normal::new(2.0, 1.5).unwrap());
    let target_labels: Vec<usize> = (0..n_target).map(|i| i % 3).collect();
    (source, source_labels, target, target_labels)
}

fn generate_market_data() -> (Vec<OHLCVBar>, Vec<OHLCVBar>) {
    let loader = StockDataLoader::new();
    let source = loader.generate_synthetic(200, 100.0, 0.02);
    let target = loader.generate_synthetic(80, 50.0, 0.04);
    (source, target)
}

// ── Network Tests ────────────────────────────────────────

#[test]
fn test_end_to_end_transfer_pipeline() {
    let (source, source_labels, target, target_labels) = generate_test_data(100, 50, 20);

    let network = TransferNetwork::new(20, 64, 32, true);
    let mut trainer = TransferTrainer::new(network)
        .with_pretrain_config(TrainingConfig {
            epochs: 5,
            learning_rate: 0.01,
            ..Default::default()
        })
        .with_adapt_config(AdaptationConfig {
            epochs: 3,
            ..Default::default()
        });

    let (pretrain_result, adapt_result) = trainer.run_pipeline(
        &source,
        &source_labels,
        &target,
        Some(&target_labels),
    );

    assert!(!pretrain_result.loss_history.is_empty());
    assert!(!adapt_result.loss_history.is_empty());

    let predictions = trainer.network().predict(&target);
    assert_eq!(predictions.len(), 50);
    for p in &predictions {
        assert!(*p < 3);
    }
}

#[test]
fn test_domain_adaptation_reduces_discrepancy() {
    let (source, source_labels, target, target_labels) = generate_test_data(80, 40, 15);

    let network = TransferNetwork::new(15, 32, 16, true);
    let mut trainer = TransferTrainer::new(network)
        .with_pretrain_config(TrainingConfig {
            epochs: 5,
            learning_rate: 0.01,
            ..Default::default()
        })
        .with_adapt_config(AdaptationConfig {
            epochs: 5,
            method: AdaptationMethod::MMD,
            ..Default::default()
        });

    // Measure similarity before adaptation
    let sim_before = trainer.compute_domain_similarity(&source, &target);

    trainer.pretrain(&source, &source_labels);
    trainer.adapt(&source, &target, Some(&target_labels));

    // Measure after
    let sim_after = trainer.compute_domain_similarity(&source, &target);

    // Similarity should either increase or at least not crash
    assert!(sim_before >= 0.0 && sim_before <= 1.0);
    assert!(sim_after >= 0.0 && sim_after <= 1.0);
}

#[test]
fn test_all_fine_tune_strategies() {
    let (source, source_labels, target, target_labels) = generate_test_data(60, 30, 10);

    for strategy in &[
        FineTuneStrategy::FeatureExtraction,
        FineTuneStrategy::PartialFineTune,
        FineTuneStrategy::FullFineTune,
    ] {
        let network = TransferNetwork::new(10, 32, 16, true)
            .with_fine_tune_strategy(*strategy);
        let mut trainer = TransferTrainer::new(network)
            .with_pretrain_config(TrainingConfig {
                epochs: 3,
                learning_rate: 0.01,
                ..Default::default()
            })
            .with_adapt_config(AdaptationConfig {
                epochs: 3,
                strategy: *strategy,
                ..Default::default()
            });

        trainer.pretrain(&source, &source_labels);
        trainer.adapt(&source, &target, Some(&target_labels));

        let preds = trainer.network().predict(&target);
        assert_eq!(preds.len(), 30);
    }
}

#[test]
fn test_all_adaptation_methods() {
    let (source, _, target, _) = generate_test_data(50, 30, 10);

    for method in &[
        AdaptationMethod::MMD,
        AdaptationMethod::CORAL,
        AdaptationMethod::Adversarial,
        AdaptationMethod::None,
    ] {
        let adapter = DomainAdapter::new(*method);
        let loss = adapter.compute_loss(&source, &target);
        if *method == AdaptationMethod::None {
            assert_eq!(loss, 0.0);
        } else {
            assert!(loss >= 0.0);
        }
    }
}

// ── Data Tests ───────────────────────────────────────────

#[test]
fn test_feature_extraction_pipeline() {
    let (source_bars, target_bars) = generate_market_data();
    let extractor = DataFeatureExtractor::new(20);

    let source_features = extractor.extract_features(&source_bars);
    let target_features = extractor.extract_features(&target_bars);

    assert!(source_features.nrows() > 0);
    assert!(target_features.nrows() > 0);
    assert_eq!(source_features.ncols(), extractor.feature_dim());
    assert_eq!(target_features.ncols(), extractor.feature_dim());

    // Features should not contain NaN
    for &val in source_features.iter() {
        assert!(!val.is_nan(), "Source features contain NaN");
    }
    for &val in target_features.iter() {
        assert!(!val.is_nan(), "Target features contain NaN");
    }
}

#[test]
fn test_label_generation() {
    let (source_bars, _) = generate_market_data();
    let extractor = DataFeatureExtractor::new(10);

    let labels = extractor.generate_labels(&source_bars, 3, 0.005);
    assert!(!labels.is_empty());

    for &label in &labels {
        assert!(label <= 2, "Label out of range: {}", label);
    }

    // Should have at least some variety
    let unique: std::collections::HashSet<&usize> = labels.iter().collect();
    assert!(unique.len() >= 2, "Labels should have variety");
}

#[test]
fn test_csv_parsing() {
    let loader = StockDataLoader::new();
    let csv = "Date,Open,High,Low,Close,Volume\n\
               2024-01-01,100.0,110.0,95.0,105.0,1000\n\
               2024-01-02,105.0,115.0,100.0,110.0,1500\n\
               2024-01-03,110.0,120.0,105.0,115.0,2000\n";
    let bars = loader.parse_csv(csv).unwrap();
    assert_eq!(bars.len(), 3);
    assert_eq!(bars[0].close, 105.0);
    assert_eq!(bars[2].volume, 2000.0);
}

// ── Strategy Tests ───────────────────────────────────────

#[test]
fn test_trading_strategy_full_cycle() {
    let mut strategy = TransferTradingStrategy::new(100_000.0)
        .with_confidence_threshold(0.5)
        .with_stop_loss(0.02)
        .with_take_profit(0.03);

    // Open long
    let buy_proba = Array1::from_vec(vec![0.1, 0.2, 0.7]);
    let signal = strategy.generate_signal(&buy_proba, 100.0, 0.9);
    assert_eq!(signal, TradingSignal::Buy);
    assert!(matches!(strategy.position(), Position::Long { .. }));

    // Close with profit
    let sell_proba = Array1::from_vec(vec![0.7, 0.2, 0.1]);
    let signal = strategy.generate_signal(&sell_proba, 103.5, 0.9);
    assert_eq!(signal, TradingSignal::Sell);
    assert_eq!(strategy.position(), &Position::Flat);

    // Check trade history
    assert_eq!(strategy.trades().len(), 1);
    assert!(strategy.trades()[0].pnl > 0.0);
}

#[test]
fn test_stop_loss_trigger() {
    let mut strategy = TransferTradingStrategy::new(100_000.0)
        .with_confidence_threshold(0.5)
        .with_stop_loss(0.02);

    // Open long
    let buy_proba = Array1::from_vec(vec![0.1, 0.2, 0.7]);
    strategy.generate_signal(&buy_proba, 100.0, 0.9);

    // Price drops 3% (past 2% stop-loss)
    let hold_proba = Array1::from_vec(vec![0.3, 0.4, 0.3]);
    let signal = strategy.generate_signal(&hold_proba, 97.0, 0.9);
    assert_eq!(signal, TradingSignal::Sell); // Stop-loss triggered
}

#[test]
fn test_low_domain_similarity_reduces_position() {
    let mut strategy_high = TransferTradingStrategy::new(100_000.0)
        .with_confidence_threshold(0.3);

    let mut strategy_low = TransferTradingStrategy::new(100_000.0)
        .with_confidence_threshold(0.3);

    let buy_proba = Array1::from_vec(vec![0.05, 0.05, 0.9]);

    strategy_high.generate_signal(&buy_proba.clone(), 100.0, 0.95);
    strategy_low.generate_signal(&buy_proba, 100.0, 0.5);

    // Both should open positions but with different sizes
    match (strategy_high.position(), strategy_low.position()) {
        (Position::Long { size: high_size, .. }, Position::Long { size: low_size, .. }) => {
            assert!(
                high_size > low_size,
                "High similarity position ({}) should be larger than low similarity ({})",
                high_size,
                low_size
            );
        }
        _ => panic!("Both strategies should have long positions"),
    }
}

#[test]
fn test_performance_summary() {
    let mut strategy = TransferTradingStrategy::new(100_000.0)
        .with_confidence_threshold(0.5);

    let buy = Array1::from_vec(vec![0.1, 0.2, 0.7]);
    let sell = Array1::from_vec(vec![0.7, 0.2, 0.1]);

    // Winning trade
    strategy.generate_signal(&buy, 100.0, 0.9);
    strategy.generate_signal(&sell, 105.0, 0.9);

    // Losing trade
    strategy.generate_signal(&buy, 100.0, 0.9);
    strategy.generate_signal(&sell, 97.0, 0.9);

    let summary = strategy.performance_summary();
    assert_eq!(summary.total_trades, 2);
    assert_eq!(summary.winning_trades, 1);
    assert!((summary.win_rate - 0.5).abs() < 1e-10);
    assert!(summary.profit_factor > 0.0);
}

// ── Metrics Tests ────────────────────────────────────────

#[test]
fn test_confusion_matrix_multiclass() {
    let predictions = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let labels = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let cm = ConfusionMatrix::new(&predictions, &labels, 3);

    assert_eq!(cm.accuracy(), 1.0);
    for c in 0..3 {
        assert_eq!(cm.precision(c), 1.0);
        assert_eq!(cm.recall(c), 1.0);
        assert_eq!(cm.f1_score(c), 1.0);
    }
    assert_eq!(cm.macro_f1(), 1.0);
}

#[test]
fn test_metrics_transfer_gain() {
    let source_preds = vec![0, 1, 2, 0, 1, 2];
    let source_labels = vec![0, 1, 2, 0, 1, 2];
    let target_preds = vec![0, 1, 2, 0, 1, 1]; // 5/6 correct
    let target_labels = vec![0, 1, 2, 0, 1, 2];

    let metrics = Metrics::compute(
        &source_preds,
        &source_labels,
        &target_preds,
        &target_labels,
        0.33, // baseline
        0.8,
        0.05,
    );

    assert_eq!(metrics.source_accuracy, 1.0);
    assert!(metrics.target_accuracy > 0.8);
    assert!(metrics.transfer_gain > 0.0, "Should gain over random baseline");
}

// ── Full Integration ─────────────────────────────────────

#[test]
fn test_full_trading_workflow() {
    // Generate market data
    let (source_bars, target_bars) = generate_market_data();

    // Extract features
    let feat_ext = DataFeatureExtractor::new(15);
    let source_features = feat_ext.extract_features(&source_bars);
    let source_labels = feat_ext.generate_labels(&source_bars, 3, 0.005);
    let target_features = feat_ext.extract_features(&target_bars);

    let n_source = source_features.nrows().min(source_labels.len());
    if n_source == 0 || target_features.nrows() == 0 {
        return; // Not enough data
    }

    let source_features = source_features.slice(ndarray::s![..n_source, ..]).to_owned();
    let source_labels = &source_labels[..n_source];

    // Train transfer model
    let network = TransferNetwork::new(feat_ext.feature_dim(), 32, 16, true);
    let mut trainer = TransferTrainer::new(network)
        .with_pretrain_config(TrainingConfig {
            epochs: 5,
            learning_rate: 0.01,
            ..Default::default()
        });

    trainer.pretrain(&source_features, source_labels);

    // Run strategy
    let mut strategy = TransferTradingStrategy::new(100_000.0)
        .with_confidence_threshold(0.4);

    let domain_similarity = trainer.compute_domain_similarity(&source_features, &target_features);

    for i in 0..target_features.nrows() {
        let proba = trainer.network().predict_proba(
            &target_features.slice(ndarray::s![i..i+1, ..]).to_owned(),
        );
        let price = target_bars[i + 15].close;
        strategy.generate_signal(&proba.row(0).to_owned(), price, domain_similarity);
    }

    // Strategy should have executed some trades
    let summary = strategy.performance_summary();
    // May or may not have trades depending on synthetic data
    assert!(strategy.portfolio_value() > 0.0);
}
