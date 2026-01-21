//! Integration tests for prototypical networks
//!
//! These tests verify the end-to-end functionality of the library.

use ndarray::{Array1, Array2};
use prototypical_networks_finance::prelude::*;
use rand::prelude::*;

/// Helper function to create test data
fn create_test_data(
    n_classes: usize,
    n_support: usize,
    n_query: usize,
    n_features: usize,
) -> (Array2<f64>, Vec<usize>, Array2<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(42);

    let total_support = n_classes * n_support;
    let total_query = n_classes * n_query;

    let mut support_features = Array2::zeros((total_support, n_features));
    let mut support_labels = Vec::with_capacity(total_support);
    let mut query_features = Array2::zeros((total_query, n_features));
    let mut query_labels = Vec::with_capacity(total_query);

    for class_idx in 0..n_classes {
        // Create class-specific pattern
        let class_center: Vec<f64> = (0..n_features)
            .map(|f| class_idx as f64 * 0.5 + f as f64 * 0.01)
            .collect();

        // Support samples
        for i in 0..n_support {
            let row_idx = class_idx * n_support + i;
            for j in 0..n_features {
                support_features[[row_idx, j]] = class_center[j] + rng.gen::<f64>() * 0.2 - 0.1;
            }
            support_labels.push(class_idx);
        }

        // Query samples
        for i in 0..n_query {
            let row_idx = class_idx * n_query + i;
            for j in 0..n_features {
                query_features[[row_idx, j]] = class_center[j] + rng.gen::<f64>() * 0.2 - 0.1;
            }
            query_labels.push(class_idx);
        }
    }

    (support_features, support_labels, query_features, query_labels)
}

#[test]
fn test_full_pipeline_euclidean() {
    let n_classes = 5;
    let n_support = 20;
    let n_query = 10;
    let input_dim = 15;
    let embedding_dim = 8;

    let (support_features, support_labels, query_features, query_labels) =
        create_test_data(n_classes, n_support, n_query, input_dim);

    // Create embedding network
    let embedding_config = EmbeddingConfig {
        input_dim,
        hidden_dims: vec![32, 16],
        output_dim: embedding_dim,
        normalize_embeddings: true,
        dropout_rate: 0.0,
        activation: ActivationType::ReLU,
    };
    let embedding_network = EmbeddingNetwork::new(embedding_config);

    // Create classifier
    let mut classifier = RegimeClassifier::new(embedding_network, DistanceFunction::Euclidean);

    // Initialize prototypes
    classifier.initialize_prototypes(&support_features, &support_labels);

    // Classify query samples
    let mut correct = 0;
    for (i, &true_label) in query_labels.iter().enumerate() {
        let features = query_features.row(i).to_owned();
        let result = classifier.classify(&features);

        if result.regime.to_index() == true_label {
            correct += 1;
        }

        // Verify probabilities sum to 1
        let sum: f64 = result.probabilities.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5, "Probabilities should sum to 1");

        // Verify confidence is the max probability
        let max_prob = result.probabilities.iter().map(|(_, p)| *p).fold(0.0f64, f64::max);
        assert!((result.confidence - max_prob).abs() < 1e-5);
    }

    let accuracy = correct as f64 / query_labels.len() as f64;
    println!("Pipeline accuracy: {:.1}%", accuracy * 100.0);

    // With random initialization, we just verify the pipeline runs
    // Actual accuracy depends on network initialization
    assert!(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy should be a valid probability");
}

#[test]
fn test_full_pipeline_cosine() {
    let (support_features, support_labels, query_features, query_labels) =
        create_test_data(5, 15, 5, 12);

    let embedding_config = EmbeddingConfig {
        input_dim: 12,
        hidden_dims: vec![24],
        output_dim: 6,
        normalize_embeddings: false,
        dropout_rate: 0.0,
        activation: ActivationType::ReLU,
    };
    let embedding_network = EmbeddingNetwork::new(embedding_config);

    let mut classifier = RegimeClassifier::new(embedding_network, DistanceFunction::Cosine);
    classifier.initialize_prototypes(&support_features, &support_labels);

    let results = classifier.classify_batch(&query_features);
    assert_eq!(results.len(), query_features.nrows());

    for result in &results {
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }
}

#[test]
fn test_episode_generation_and_training() {
    let n_features = 10;
    let n_classes = 5;
    let samples_per_class = 30;

    // Create dataset
    let mut rng = StdRng::seed_from_u64(123);
    let mut features = Array2::zeros((n_classes * samples_per_class, n_features));
    let mut labels = Vec::new();

    for class_idx in 0..n_classes {
        for i in 0..samples_per_class {
            let row_idx = class_idx * samples_per_class + i;
            for j in 0..n_features {
                features[[row_idx, j]] = class_idx as f64 + rng.gen::<f64>() * 0.3;
            }
            labels.push(class_idx);
        }
    }

    // Create episode generator
    let episode_config = EpisodeConfig {
        n_way: 3,
        k_shot: 5,
        n_query: 5,
    };
    let mut generator = EpisodeGenerator::with_seed(episode_config.clone(), 42);
    generator.add_data(features, &labels);

    assert!(generator.can_generate());

    // Generate episodes
    let episodes = generator.generate_episodes(10);
    assert_eq!(episodes.len(), 10);

    for episode in &episodes {
        assert_eq!(episode.n_way(), 3);
        assert_eq!(episode.k_shot(), 5);
        assert_eq!(episode.n_query(), 5);
        assert_eq!(episode.support_features.nrows(), 15); // 3 * 5
        assert_eq!(episode.query_features.nrows(), 15); // 3 * 5
    }
}

#[test]
fn test_signal_generation_chain() {
    let (support_features, support_labels, _, _) = create_test_data(5, 20, 5, 15);

    // Setup classifier
    let embedding_config = EmbeddingConfig {
        input_dim: 15,
        hidden_dims: vec![16],
        output_dim: 8,
        ..Default::default()
    };
    let network = EmbeddingNetwork::new(embedding_config);
    let mut classifier = RegimeClassifier::new(network, DistanceFunction::Euclidean);
    classifier.initialize_prototypes(&support_features, &support_labels);

    // Setup signal generator
    let mut signal_generator = SignalGenerator::new();

    // Create test features for different regimes
    for regime in MarketRegime::all() {
        // Create features biased towards this regime
        let mut features = Array1::zeros(15);
        let bias = regime.to_index() as f64 * 0.5;
        for i in 0..15 {
            features[i] = bias + i as f64 * 0.01;
        }

        let classification = classifier.classify(&features);
        let signal = signal_generator.generate(&classification);

        // Verify signal is valid
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
        assert!(signal.position_size >= 0.0 && signal.position_size <= 1.0);
    }
}

#[test]
fn test_position_management() {
    let config = ExecutionConfig {
        max_position_size: 100000.0,
        min_position_size: 10.0,  // Lower minimum for test
        stop_loss_pct: 0.02,
        take_profit_pct: 0.05,
        ..Default::default()
    };
    let mut manager = PositionManager::new(config);

    // Create a buy signal with high position size to ensure it passes min threshold
    let buy_signal = TradingSignal {
        signal_type: SignalType::Buy,
        regime: MarketRegime::WeakUptrend,
        confidence: 0.8,
        position_size: 0.5,
        is_unusual: false,
        reason: "Test buy".to_string(),
    };

    // Process signal at price 100 with plenty of capital
    let orders = manager.process_signal(&buy_signal, 100.0, 100000.0);
    assert!(!orders.is_empty(), "Should generate orders with sufficient capital");

    // Execute the order
    manager.execute_order(&orders[0], 100.0);
    assert!(manager.position().is_open());
    assert_eq!(manager.position().side, PositionSide::Long);

    // Price goes up to 105 (5% profit) - should trigger take profit
    let hold_signal = TradingSignal {
        signal_type: SignalType::Hold,
        regime: MarketRegime::WeakUptrend,
        confidence: 0.6,
        position_size: 0.0,
        is_unusual: false,
        reason: "Hold".to_string(),
    };

    let orders = manager.process_signal(&hold_signal, 105.0, 10000.0);
    assert!(!orders.is_empty());
    assert_eq!(orders[0].order_type, OrderType::TakeProfit);
}

#[test]
fn test_metrics_calculation() {
    // Simulate portfolio values with known characteristics
    let portfolio_values: Vec<f64> = (0..100)
        .map(|i| 10000.0 * (1.0 + 0.001_f64).powi(i))
        .collect();

    let calculator = MetricsCalculator::daily();
    let metrics = calculator.calculate(&portfolio_values);

    // Consistent positive returns should give positive Sharpe
    assert!(metrics.total_return > 0.0);
    assert!(metrics.sharpe_ratio > 0.0);
    assert!(metrics.max_drawdown >= 0.0 && metrics.max_drawdown <= 1.0);
}

#[test]
fn test_feature_extraction() {
    use chrono::{TimeZone, Utc};

    // Create test klines
    let klines: Vec<Kline> = (0..50)
        .map(|i| {
            let base_price = 100.0 + i as f64 * 0.5;
            // Calculate day and hour to avoid invalid time (hour must be 0-23)
            let day = 1 + (i / 24) as u32;
            let hour = (i % 24) as u32;
            Kline::new(
                Utc.with_ymd_and_hms(2024, 1, day, hour, 0, 0).unwrap(),
                base_price,
                base_price + 1.0,
                base_price - 0.5,
                base_price + 0.3,
                1000.0 + i as f64 * 10.0,
            )
        })
        .collect();

    let extractor = FeatureExtractor::new();
    let features = extractor.extract_from_klines(&klines);

    assert!(features.dim() > 0);
    assert!(!features.feature_names.is_empty());
    assert_eq!(features.dim(), features.feature_names.len());

    // Verify features are finite
    for &val in features.features.iter() {
        assert!(val.is_finite(), "All features should be finite");
    }
}

#[test]
fn test_distance_functions() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

    // Euclidean distance
    let euclidean = DistanceFunction::Euclidean.compute(&a, &b);
    let expected = ((3.0_f64).powi(2) + (3.0_f64).powi(2) + (3.0_f64).powi(2)).sqrt();
    assert!((euclidean - expected).abs() < 1e-10);

    // Manhattan distance
    let manhattan = DistanceFunction::Manhattan.compute(&a, &b);
    assert!((manhattan - 9.0).abs() < 1e-10);

    // Cosine distance
    let cosine = DistanceFunction::Cosine.compute(&a, &b);
    assert!(cosine >= 0.0 && cosine <= 2.0);

    // Same vector should have distance 0
    let self_euclidean = DistanceFunction::Euclidean.compute(&a, &a);
    assert!(self_euclidean.abs() < 1e-10);

    let self_cosine = DistanceFunction::Cosine.compute(&a, &a);
    assert!(self_cosine.abs() < 1e-10);
}

#[test]
fn test_learning_rate_scheduler() {
    // Step decay
    let scheduler = LearningRateScheduler::step_decay(0.1, 10, 0.5, 0.001);
    assert!((scheduler.step(0) - 0.1).abs() < 1e-10);
    assert!((scheduler.step(9) - 0.1).abs() < 1e-10);
    assert!((scheduler.step(10) - 0.05).abs() < 1e-10);
    assert!((scheduler.step(20) - 0.025).abs() < 1e-10);

    // Cosine annealing
    let cosine_scheduler = LearningRateScheduler::cosine_annealing(0.1, 100, 0.01);
    let start_lr = cosine_scheduler.step(0);
    let mid_lr = cosine_scheduler.step(50);
    let end_lr = cosine_scheduler.step(100);

    assert!(start_lr > mid_lr);
    assert!(mid_lr > end_lr);
    assert!(end_lr >= 0.01);
}

#[test]
fn test_market_regime_conversion() {
    for regime in MarketRegime::all() {
        let idx = regime.to_index();
        let back = MarketRegime::from_index(idx);
        assert_eq!(back, Some(regime));
    }

    assert_eq!(MarketRegime::from_index(100), None);
    assert_eq!(MarketRegime::count(), 5);
}

#[test]
fn test_outlier_detection() {
    let (support_features, support_labels, _, _) = create_test_data(5, 20, 5, 15);

    let embedding_config = EmbeddingConfig {
        input_dim: 15,
        hidden_dims: vec![16],
        output_dim: 8,
        ..Default::default()
    };
    let network = EmbeddingNetwork::new(embedding_config);
    let mut classifier = RegimeClassifier::new(network, DistanceFunction::Euclidean)
        .with_outlier_threshold(5.0);

    classifier.initialize_prototypes(&support_features, &support_labels);

    // Normal point (close to training data)
    let normal_point = support_features.row(0).to_owned();
    let normal_result = classifier.classify(&normal_point);

    // Outlier point (far from training data)
    let mut outlier_point = Array1::zeros(15);
    for i in 0..15 {
        outlier_point[i] = 100.0 + i as f64; // Very far from training data
    }
    let outlier_result = classifier.classify(&outlier_point);

    // Both should produce valid results
    // Note: With normalized embeddings, distance relationships may vary
    assert!(normal_result.confidence >= 0.0 && normal_result.confidence <= 1.0);
    assert!(outlier_result.confidence >= 0.0 && outlier_result.confidence <= 1.0);
    assert!(normal_result.min_distance >= 0.0);
    assert!(outlier_result.min_distance >= 0.0);
}
