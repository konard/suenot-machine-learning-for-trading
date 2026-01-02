//! Train DenseNet Model
//!
//! Training pipeline for DenseNet trading model.
//!
//! Usage:
//!   cargo run --bin train_densenet

use ndarray::Array2;
use rust_densenet_trading::data::candle::{Candle, CandleSeries};
use rust_densenet_trading::data::dataset::TradingDataset;
use rust_densenet_trading::densenet::{DenseNet, DenseNetConfig};
use rust_densenet_trading::features::extractor::FeatureExtractor;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           DenseNet Trading - Training Demo                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let config = DenseNetConfig::small();
    let sequence_length = config.sequence_length;
    let prediction_horizon = 4; // Predict 4 candles ahead
    let threshold = 0.002; // 0.2% threshold for labeling

    println!("Training Configuration:");
    println!("  Model: DenseNet-Small");
    println!("  Sequence Length: {}", sequence_length);
    println!("  Prediction Horizon: {} candles", prediction_horizon);
    println!("  Label Threshold: {:.2}%", threshold * 100.0);
    println!();

    // Generate synthetic training data (in real use, load from Bybit)
    println!("Generating synthetic training data...");
    let candles = generate_synthetic_candles(2000);
    println!("  Generated {} candles", candles.len());

    // Extract features
    println!("Extracting features...");
    let extractor = FeatureExtractor::new();
    let features = extractor.extract_from_candles(&candles);
    println!(
        "  Feature matrix shape: [{}, {}]",
        features.dim().0,
        features.dim().1
    );

    // Create dataset
    println!("Creating training dataset...");
    let dataset = TradingDataset::from_features(
        &candles,
        &features,
        sequence_length,
        prediction_horizon,
        threshold,
    );

    println!("  Dataset size: {} samples", dataset.len());
    println!("  Label distribution: {:?}", dataset.label_distribution());
    println!("  Class weights: {:?}", dataset.class_weights());

    // Split into train/test
    let (train_data, test_data) = dataset.train_test_split(0.8);
    println!(
        "  Train samples: {}, Test samples: {}",
        train_data.len(),
        test_data.len()
    );

    // Print dataset statistics
    let stats = dataset.statistics();
    println!();
    println!("{}", stats);

    // Create model
    println!("Creating DenseNet model...");
    let model = DenseNet::new(config.clone());
    println!("  Total parameters: {}", model.num_parameters());

    // Training loop (simplified - real training would use backpropagation)
    println!();
    println!("Starting training simulation...");
    println!("{:-<60}", "");

    let epochs = 5;
    let batch_size = 32;

    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        // Create batches
        let batches = train_data.batches(batch_size);

        for (batch_idx, batch) in batches.iter().enumerate() {
            // Forward pass for each sample in batch
            for sample in batch.iter() {
                let prediction = model.forward(&sample.features, true);

                // Simulated loss (cross-entropy approximation)
                let target_prob = prediction.probabilities[sample.label];
                let loss = -target_prob.ln().max(-10.0);
                epoch_loss += loss;

                // Accuracy
                if prediction.predicted_class == sample.label {
                    correct += 1;
                }
                total += 1;
            }

            // Progress (every 10 batches)
            if batch_idx % 10 == 0 {
                print!(
                    "\rEpoch {}/{} | Batch {}/{} | Loss: {:.4} | Acc: {:.2}%",
                    epoch,
                    epochs,
                    batch_idx + 1,
                    batches.len(),
                    epoch_loss / total as f64,
                    correct as f64 / total as f64 * 100.0
                );
            }
        }

        let avg_loss = epoch_loss / total as f64;
        let accuracy = correct as f64 / total as f64 * 100.0;

        println!(
            "\rEpoch {}/{} complete | Loss: {:.4} | Accuracy: {:.2}%          ",
            epoch, epochs, avg_loss, accuracy
        );
    }

    // Evaluation on test set
    println!();
    println!("Evaluating on test set...");

    let mut test_correct = 0;
    let mut test_total = 0;
    let mut predictions_by_class = vec![0; 3];
    let mut correct_by_class = vec![0; 3];

    for sample in &test_data.samples {
        let prediction = model.forward(&sample.features, false);

        predictions_by_class[prediction.predicted_class] += 1;

        if prediction.predicted_class == sample.label {
            test_correct += 1;
            correct_by_class[sample.label] += 1;
        }
        test_total += 1;
    }

    let test_accuracy = test_correct as f64 / test_total as f64 * 100.0;

    println!();
    println!("Test Results:");
    println!("{:-<50}", "");
    println!("  Overall Accuracy: {:.2}%", test_accuracy);
    println!("  Predictions distribution: {:?}", predictions_by_class);
    println!();

    // Per-class accuracy
    let class_names = ["Short", "Hold", "Long"];
    let label_dist = test_data.label_distribution();

    println!("Per-class Performance:");
    for (i, name) in class_names.iter().enumerate() {
        let class_total = label_dist[i];
        let class_correct = correct_by_class[i];
        let class_acc = if class_total > 0 {
            class_correct as f64 / class_total as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  {}: {}/{} ({:.2}%)",
            name, class_correct, class_total, class_acc
        );
    }

    println!();
    println!("Training complete!");
    println!();
    println!("Note: This is a demonstration. In production:");
    println!("  1. Use real Bybit data (fetch_bybit_data binary)");
    println!("  2. Implement proper backpropagation (or use tch-rs)");
    println!("  3. Save model weights for inference");
    println!("  4. Use validation set for early stopping");
}

/// Generate synthetic candle data for demonstration
fn generate_synthetic_candles(n: usize) -> CandleSeries {
    let mut candles = CandleSeries::new();
    let mut price = 42000.0; // Starting BTC-like price
    let mut rng_seed = 12345u64;

    for i in 0..n {
        // Simple PRNG for reproducibility
        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let rand1 = (rng_seed % 1000) as f64 / 1000.0 - 0.5;

        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let rand2 = (rng_seed % 1000) as f64 / 1000.0;

        // Add some trend and mean reversion
        let trend = ((i as f64) * 0.001).sin() * 500.0;
        let mean_reversion = (42000.0 - price) * 0.01;

        // Price change with volatility
        let volatility = 200.0;
        let change = rand1 * volatility + trend * 0.1 + mean_reversion;

        price = (price + change).max(30000.0).min(60000.0);

        // Generate OHLC
        let open = price;
        let close = price + rand1 * 100.0;
        let high = open.max(close) + rand2 * 100.0;
        let low = open.min(close) - rand2 * 100.0;
        let volume = 100.0 + rand2 * 500.0;

        candles.push(Candle::new(
            (i as u64) * 900_000, // 15-minute candles
            open,
            high,
            low.max(1.0),
            close,
            volume,
            volume * close,
        ));
    }

    candles
}
