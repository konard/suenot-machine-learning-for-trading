//! New Asset Adaptation Example
//!
//! This example demonstrates adapting a few-shot model to predict
//! market movements for a new cryptocurrency with minimal data.
//!
//! Run with: cargo run --example new_asset_adaptation

use few_shot_market_prediction::methods::{
    FewShotConfig, FewShotLearner, FewShotMethod,
    MetricBasedLearner, MAMLLearner,
};
use few_shot_market_prediction::network::DistanceMetric;
use few_shot_market_prediction::strategy::{
    MarketRegime, RegimeClassifier,
    SignalGenerator, SignalConfig,
};
use ndarray::Array2;

fn main() {
    println!("=== Few-Shot Adaptation to New Asset ===\n");

    // Scenario: We have a model trained on BTC/ETH, and want to adapt to a new altcoin
    println!("Scenario: Adapting from BTC/ETH patterns to predict DOGE movements\n");

    // Step 1: Generate synthetic "historical" data for major assets (for reference)
    let (btc_features, btc_labels) = generate_asset_data("BTC", 50);
    let (eth_features, eth_labels) = generate_asset_data("ETH", 50);

    println!("Step 1: Historical data available (simulated)");
    println!("  BTC: {} labeled samples", btc_features.nrows());
    println!("  ETH: {} labeled samples", eth_features.nrows());
    println!();

    // Step 2: Get LIMITED data for new asset (few-shot scenario)
    let (doge_support, doge_support_labels) = generate_asset_data("DOGE", 9); // Only 3 per class!
    let (doge_query, doge_query_labels) = generate_asset_data("DOGE_query", 15);

    println!("Step 2: New asset (DOGE) - Few-shot data");
    println!("  Support set: {} examples (only 3 per class!)", doge_support.nrows());
    println!("  Query set: {} examples to predict", doge_query.nrows());
    println!();

    // Step 3: Define configuration for few-shot learning
    let config = FewShotConfig::default()
        .with_input_dim(12)
        .with_embedding_dim(8)
        .with_hidden_dims(vec![64, 32])
        .with_n_way(3)
        .with_k_shot(3)
        .with_adaptation_steps(5)
        .with_adaptation_lr(0.01);

    println!("Step 3: Few-shot configuration");
    println!("  Input dim: {}", config.input_dim);
    println!("  Embedding dim: {}", config.embedding_dim);
    println!("  N-way: {} (3 classes: up, sideways, down)", config.n_way);
    println!("  K-shot: {} per class", config.k_shot);
    println!();

    // Step 4: Adapt using MAML
    println!("Step 4: Adapting model to DOGE using MAML...\n");

    let mut maml_learner = MAMLLearner::new(config.clone());
    maml_learner.fit(&doge_support, &doge_support_labels);
    let maml_predictions = maml_learner.predict(&doge_query);

    // Display results
    let class_names = ["Uptrend", "Sideways", "Downtrend"];

    println!("--- MAML Adaptation Results for DOGE ---\n");

    if let Some(adaptation) = maml_learner.last_adaptation() {
        println!("Adaptation summary:");
        println!("  Steps: {}", adaptation.steps);
        println!("  Initial loss: {:.4}", adaptation.initial_loss);
        println!("  Final loss: {:.4}", adaptation.final_loss);
        println!("  Improvement: {:.1}%\n", adaptation.improvement * 100.0);
    }

    println!("Predictions on query set:");
    for (i, (pred, &actual)) in maml_predictions.iter()
        .zip(doge_query_labels.iter())
        .enumerate()
    {
        let correct = if pred.predicted_class == actual { "✓" } else { "✗" };
        println!(
            "  Sample {:2}: Predicted {} (conf: {:5.1}%) | Actual {} {}",
            i + 1,
            class_names[pred.predicted_class],
            pred.confidence * 100.0,
            class_names[actual],
            correct
        );
    }

    // Calculate MAML accuracy
    let maml_correct = maml_predictions.iter()
        .zip(doge_query_labels.iter())
        .filter(|(pred, &actual)| pred.predicted_class == actual)
        .count();
    let maml_accuracy = maml_correct as f64 / doge_query_labels.len() as f64 * 100.0;

    println!("\nMAML accuracy: {:.1}% ({}/{})",
        maml_accuracy, maml_correct, doge_query_labels.len());

    // Step 5: Compare with metric-based approach
    println!("\n--- Comparison: Metric-Based (no adaptation) ---\n");

    let metric_config = config.clone()
        .with_method(FewShotMethod::Metric)
        .with_distance_metric(DistanceMetric::Cosine);

    let mut metric_learner = MetricBasedLearner::new(metric_config);
    metric_learner.fit(&doge_support, &doge_support_labels);
    let metric_predictions = metric_learner.predict(&doge_query);

    let metric_correct = metric_predictions.iter()
        .zip(doge_query_labels.iter())
        .filter(|(pred, &actual)| pred.predicted_class == actual)
        .count();
    let metric_accuracy = metric_correct as f64 / doge_query_labels.len() as f64 * 100.0;

    println!("Metric-based accuracy: {:.1}% ({}/{})",
        metric_accuracy, metric_correct, doge_query_labels.len());

    println!("\nMethod comparison:");
    println!("  MAML (with adaptation): {:.1}%", maml_accuracy);
    println!("  Metric-based (no adaptation): {:.1}%", metric_accuracy);
    println!();

    // Step 6: Generate trading signals
    println!("--- Trading Signal Generation ---\n");

    // Aggregate predictions to determine overall regime
    let mut class_votes = [0.0f64; 3];
    for pred in &maml_predictions {
        class_votes[pred.predicted_class] += pred.confidence;
    }

    let total_confidence: f64 = class_votes.iter().sum();
    let regime_probs: Vec<f64> = class_votes.iter()
        .map(|&c| c / total_confidence.max(1e-10))
        .collect();

    println!("Aggregated class probabilities:");
    for (i, (name, &prob)) in class_names.iter().zip(regime_probs.iter()).enumerate() {
        println!("  {}: {:.1}%", name, prob * 100.0);
    }

    // Map to market regimes (simplified: 3 classes -> select from 8 regimes)
    let dominant_class = regime_probs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(1);

    let mapped_regime = match dominant_class {
        0 => MarketRegime::WeakUptrend,
        2 => MarketRegime::WeakDowntrend,
        _ => MarketRegime::Sideways,
    };

    println!("\nMapped to regime: {}", mapped_regime);

    // Create classifier and signal
    let mut classifier = RegimeClassifier::new(0.5);
    let avg_confidence = maml_predictions.iter()
        .map(|p| p.confidence)
        .sum::<f64>() / maml_predictions.len() as f64;

    // Build regime probabilities for classifier
    let mut full_regime_probs = vec![0.0; 8];
    match dominant_class {
        0 => {
            full_regime_probs[0] = regime_probs[0] * 0.3; // Strong uptrend
            full_regime_probs[1] = regime_probs[0] * 0.7; // Weak uptrend
        }
        2 => {
            full_regime_probs[3] = regime_probs[2] * 0.7; // Weak downtrend
            full_regime_probs[4] = regime_probs[2] * 0.3; // Strong downtrend
        }
        _ => {
            full_regime_probs[2] = regime_probs[1]; // Sideways
        }
    }

    let classification = classifier.classify_with_uncertainty(&full_regime_probs);

    let signal_config = SignalConfig::default();
    let mut signal_generator = SignalGenerator::new(signal_config);
    let signal = signal_generator.generate("DOGEUSDT", &classification);

    println!("\nGenerated signal for DOGEUSDT:");
    println!("  Type: {:?}", signal.signal_type);
    println!("  Confidence: {:.1}%", signal.confidence * 100.0);
    println!("  Position size: {:.1}%", signal.position_size * 100.0);
    if let Some(tp) = signal.take_profit {
        println!("  Take profit: {:.1}%", tp);
    }
    if let Some(sl) = signal.stop_loss {
        println!("  Stop loss: {:.1}%", sl);
    }

    // Step 7: Real-world usage hint
    println!("\n--- Real-World Usage ---\n");
    println!("To use with real Bybit data:");
    println!("  1. Create BybitClient for API access");
    println!("  2. Fetch klines: client.get_klines(\"DOGEUSDT\", \"1h\", 100).await");
    println!("  3. Extract features using FeatureExtractor");
    println!("  4. Run few-shot prediction with extracted features");
    println!("  5. Generate and execute trading signals");
    println!();

    println!("=== Example Complete ===");
}

/// Generate synthetic market data for an asset.
fn generate_asset_data(asset_name: &str, count: usize) -> (Array2<f64>, Vec<usize>) {
    let dim = 12;
    let mut data = Vec::new();
    let mut labels = Vec::new();

    // Seed based on asset name for reproducibility
    let base_seed: u64 = asset_name.bytes().map(|b| b as u64).sum();

    for i in 0..count {
        // Determine class (0: uptrend, 1: sideways, 2: downtrend)
        let class = i % 3;

        // Generate features based on class
        let base = match class {
            0 => vec![0.02, 0.015, 0.5, 65.0, 1.5, 0.8, 0.7, 0.6, 0.3, 0.7, 0.8, 0.6],
            1 => vec![0.001, 0.002, 0.3, 50.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            _ => vec![-0.02, -0.015, 0.6, 35.0, 0.7, 0.3, 0.3, 0.4, 0.7, 0.3, 0.2, 0.4],
        };

        let noisy = add_noise(&base, 0.15, base_seed + i as u64);
        data.extend(noisy);
        labels.push(class);
    }

    let features = Array2::from_shape_vec((count, dim), data).unwrap();
    (features, labels)
}

/// Add noise to feature vector.
fn add_noise(base: &[f64], noise_level: f64, seed: u64) -> Vec<f64> {
    let mut rng_state = seed;

    base.iter().map(|&v| {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random = ((rng_state >> 16) & 0x7FFF) as f64 / 32768.0 - 0.5;
        v + v.abs().max(0.01) * noise_level * random * 2.0
    }).collect()
}
