//! Integration tests for Matching Networks for Finance

use matching_networks_finance::{
    MatchingNetwork,
    MarketPattern,
    data::{FeatureExtractor, StockDataLoader, OHLCVBar},
    training::{EpisodicTrainer, TrainingConfig},
    utils::{Metrics, ConfusionMatrix},
};
use ndarray::{Array1, Array2};

/// Test that the matching network can be created and used for prediction
#[test]
fn test_matching_network_basic() {
    // Create a simple network
    let network = MatchingNetwork::new(10, 32, 16, false);

    assert_eq!(network.input_dim(), 10);
    assert_eq!(network.embedding_dim(), 16);
    assert!(!network.uses_fce());

    // Create simple test data
    let support_features = Array2::from_shape_fn((6, 10), |(i, j)| {
        (i * 10 + j) as f64 / 100.0
    });
    let support_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
    let query_features = Array2::from_shape_fn((3, 10), |(i, j)| {
        (i * 10 + j) as f64 / 100.0 + 0.01
    });

    // Make predictions
    let predictions = network.predict(&support_features, &support_labels, &query_features);

    assert_eq!(predictions.len(), 3);
    for &pred in predictions.iter() {
        assert!(pred < 3, "Prediction should be a valid class index");
    }
}

/// Test matching network with FCE
#[test]
fn test_matching_network_with_fce() {
    let network = MatchingNetwork::new(10, 32, 16, true);

    assert!(network.uses_fce());

    let support_features = Array2::from_shape_fn((4, 10), |_| rand::random::<f64>());
    let support_labels = Array1::from_vec(vec![0, 0, 1, 1]);
    let query_features = Array2::from_shape_fn((2, 10), |_| rand::random::<f64>());

    let (predictions, confidences) = network.predict_with_confidence(
        &support_features,
        &support_labels,
        &query_features,
    );

    assert_eq!(predictions.len(), 2);
    assert_eq!(confidences.len(), 2);

    for &conf in confidences.iter() {
        assert!(conf >= 0.0 && conf <= 1.0, "Confidence should be between 0 and 1");
    }
}

/// Test market pattern enum
#[test]
fn test_market_patterns() {
    assert_eq!(MarketPattern::count(), 5);

    for pattern in MarketPattern::all() {
        let index = pattern.to_index();
        let recovered = MarketPattern::from_index(index).unwrap();
        assert_eq!(*pattern, recovered);

        // Check description is not empty
        assert!(!pattern.description().is_empty());
    }

    // Test invalid index
    assert!(MarketPattern::from_index(100).is_none());
}

/// Test feature extractor
#[test]
fn test_feature_extractor() {
    let extractor = FeatureExtractor::new();

    // Create test bars
    let bars: Vec<OHLCVBar> = (0..60)
        .map(|i| {
            let base = 100.0 + (i as f64) * 0.1;
            OHLCVBar::new(
                i * 60000,
                base,
                base + 1.0,
                base - 0.5,
                base + 0.5,
                1000.0 + (i as f64) * 10.0,
            )
        })
        .collect();

    let features = extractor.extract(&bars);

    assert!(features.is_some());
    let features = features.unwrap();
    assert_eq!(features.dim(), extractor.feature_dim());
}

/// Test stock data loader
#[test]
fn test_stock_data_loader() {
    let loader = StockDataLoader::new();

    // Generate synthetic data
    let bars = loader.generate_synthetic(100, 100.0, 0.02, 0.001);

    assert_eq!(bars.len(), 100);

    // Verify OHLCV constraints
    for bar in &bars {
        assert!(bar.high >= bar.low);
        assert!(bar.high >= bar.open.min(bar.close));
        assert!(bar.low <= bar.open.max(bar.close));
        assert!(bar.volume > 0.0);
    }

    // Generate pattern data
    let uptrend = loader.generate_pattern_data("uptrend", 50, 100.0);
    assert_eq!(uptrend.len(), 50);
}

/// Test metrics calculation
#[test]
fn test_metrics() {
    let predictions = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);
    let labels = Array1::from_vec(vec![0, 1, 2, 1, 1, 2]);

    let metrics = Metrics::compute(&predictions, &labels, 3);

    // 5 out of 6 correct = 83.3%
    assert!((metrics.accuracy - 5.0 / 6.0).abs() < 1e-10);

    // Check class metrics exist
    for class_id in 0..3 {
        assert!(metrics.get_class_metrics(class_id).is_some());
    }
}

/// Test confusion matrix
#[test]
fn test_confusion_matrix() {
    let predictions = Array1::from_vec(vec![0, 0, 1, 1]);
    let labels = Array1::from_vec(vec![0, 1, 0, 1]);

    let cm = ConfusionMatrix::compute(&predictions, &labels, 2);

    assert_eq!(cm.total(), 4);
    assert_eq!(cm.get(0, 0), 1); // True positive for class 0
    assert_eq!(cm.get(1, 1), 1); // True positive for class 1
}

/// Test episodic trainer
#[test]
fn test_episodic_trainer() {
    let config = TrainingConfig {
        n_way: 3,
        k_shot: 5,
        n_query: 5,
        num_episodes: 10,
        ..Default::default()
    };

    let mut trainer = EpisodicTrainer::new(config);

    // Generate synthetic data
    let num_samples = 200;
    let num_classes = 5;
    let feature_dim = 10;

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for class_id in 0..num_classes {
        for _ in 0..(num_samples / num_classes) {
            let sample: Vec<f64> = (0..feature_dim)
                .map(|_| rand::random::<f64>() + class_id as f64)
                .collect();
            features.extend(sample);
            labels.push(class_id);
        }
    }

    let features = Array2::from_shape_vec((num_samples, feature_dim), features).unwrap();
    let labels = Array1::from_vec(labels);

    trainer.add_data(&features, &labels, 0.2);

    assert_eq!(trainer.num_classes(), num_classes);

    // Generate an episode
    let episode = trainer.generate_episode(false);
    assert!(episode.is_some());

    let episode = episode.unwrap();
    assert_eq!(episode.support_features.nrows(), 3 * 5); // n_way * k_shot
    assert_eq!(episode.query_features.nrows(), 3 * 5); // n_way * n_query
}
