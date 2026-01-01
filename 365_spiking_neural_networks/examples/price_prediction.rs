//! Price Prediction with SNN
//!
//! This example demonstrates using a Spiking Neural Network
//! to predict cryptocurrency price direction using Bybit data.

use snn_trading::{
    network::SNNNetwork,
    encoding::DeltaEncoder,
    data::{BybitClient, Candle, generate_simulated_candles},
    learning::RewardModulatedSTDP,
};

#[tokio::main]
async fn main() {
    println!("=== SNN Price Direction Prediction ===\n");

    // Try to fetch real data, fall back to simulated
    let candles = match fetch_bybit_data().await {
        Ok(data) => {
            println!("Fetched {} candles from Bybit\n", data.len());
            data
        }
        Err(e) => {
            println!("Could not fetch Bybit data: {}", e);
            println!("Using simulated data instead\n");
            generate_simulated_candles(500, 50000.0)
        }
    };

    if candles.is_empty() {
        println!("No data available");
        return;
    }

    // Create network for prediction
    let mut network = SNNNetwork::builder()
        .input_layer(10)   // Features: returns, volume, etc.
        .hidden_layer(50)  // Pattern extraction
        .hidden_layer(20)  // Higher-level features
        .output_layer(2)   // Up / Down
        .with_dt(1.0)
        .build();

    println!("Network architecture: 10 -> 50 -> 20 -> 2");

    // Create encoder
    let mut encoder = DeltaEncoder::for_prices(10);

    // Initialize with first candle
    let initial_features = extract_features(&candles[0]);
    encoder.initialize(&initial_features);

    // Training phase
    println!("\n--- Training Phase ---");
    let train_size = (candles.len() * 80) / 100;
    let mut correct_train = 0;
    let mut total_train = 0;

    for i in 1..train_size {
        let features = extract_features(&candles[i]);
        let currents = encoder.process_to_currents(&features, i as f64);

        // Pad/prepare input
        let mut input = vec![0.0; 10];
        for (j, &c) in currents.iter().enumerate().take(10) {
            input[j] = c * 50.0;
        }

        // Run network
        let spikes = network.run(&input, 50);
        let (up_count, down_count) = count_output_spikes(&spikes);

        // Actual direction
        let actual_up = candles[i].close > candles[i].open;

        // Prediction
        let pred_up = up_count > down_count;

        if pred_up == actual_up {
            correct_train += 1;
        }
        total_train += 1;

        // Learn from result
        let reward = if pred_up == actual_up { 1.0 } else { -1.0 };
        network.learn(reward * 0.01);

        // Progress update
        if i % 100 == 0 {
            let acc = correct_train as f64 / total_train as f64 * 100.0;
            println!("  Processed {} candles, accuracy: {:.1}%", i, acc);
        }
    }

    let train_accuracy = correct_train as f64 / total_train as f64 * 100.0;
    println!("\nTraining accuracy: {:.1}%", train_accuracy);

    // Testing phase
    println!("\n--- Testing Phase ---");
    let mut correct_test = 0;
    let mut total_test = 0;
    let mut predictions = Vec::new();

    for i in train_size..candles.len() {
        let features = extract_features(&candles[i]);
        let currents = encoder.process_to_currents(&features, i as f64);

        let mut input = vec![0.0; 10];
        for (j, &c) in currents.iter().enumerate().take(10) {
            input[j] = c * 50.0;
        }

        let spikes = network.run(&input, 50);
        let (up_count, down_count) = count_output_spikes(&spikes);

        let actual_up = candles[i].close > candles[i].open;
        let pred_up = up_count > down_count;
        let confidence = (up_count as i32 - down_count as i32).abs() as f64
            / (up_count + down_count).max(1) as f64;

        if pred_up == actual_up {
            correct_test += 1;
        }
        total_test += 1;

        predictions.push((pred_up, actual_up, confidence));
    }

    let test_accuracy = correct_test as f64 / total_test as f64 * 100.0;
    println!("Test accuracy: {:.1}%", test_accuracy);

    // Analyze by confidence
    println!("\n--- Accuracy by Confidence Level ---");
    analyze_by_confidence(&predictions);

    // Show sample predictions
    println!("\n--- Sample Predictions ---");
    println!("{:<10} {:<10} {:<10} {:<10}", "Predicted", "Actual", "Correct", "Confidence");
    println!("{}", "-".repeat(42));

    for (pred, actual, conf) in predictions.iter().take(10) {
        let pred_str = if *pred { "UP" } else { "DOWN" };
        let actual_str = if *actual { "UP" } else { "DOWN" };
        let correct_str = if pred == actual { "✓" } else { "✗" };
        println!("{:<10} {:<10} {:<10} {:.2}", pred_str, actual_str, correct_str, conf);
    }

    // Summary
    println!("\n--- Summary ---");
    println!("Training samples: {}", train_size - 1);
    println!("Test samples: {}", candles.len() - train_size);
    println!("Training accuracy: {:.1}%", train_accuracy);
    println!("Test accuracy: {:.1}%", test_accuracy);

    let baseline = 50.0;
    let improvement = test_accuracy - baseline;
    println!("Improvement over random: {:.1}%", improvement);
}

async fn fetch_bybit_data() -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", "5", 500).await?;
    Ok(candles)
}

fn extract_features(candle: &Candle) -> Vec<f64> {
    let ret = candle.return_pct();
    let range = candle.range() / candle.open.max(0.0001);
    let body_ratio = candle.body_size() / candle.range().max(0.0001);
    let position = (candle.close - candle.low) / candle.range().max(0.0001);

    // Normalize volume (assuming typical range)
    let vol_norm = (candle.volume / 1000.0).min(1.0);

    vec![
        ret * 10.0,           // Scaled return
        range * 100.0,        // Scaled range
        body_ratio,           // Already 0-1
        position,             // Already 0-1
        vol_norm,             // Normalized volume
        ret.signum(),         // Direction
        if ret > 0.0 { ret * 10.0 } else { 0.0 },  // Positive return
        if ret < 0.0 { -ret * 10.0 } else { 0.0 }, // Negative return
        (range - 0.01).max(0.0) * 100.0,           // Excess range
        if candle.is_bullish() { 1.0 } else { 0.0 }, // Bullish indicator
    ]
}

fn count_output_spikes(spike_trains: &[Vec<bool>]) -> (usize, usize) {
    let up_count = spike_trains.iter()
        .filter(|s| s.get(0) == Some(&true))
        .count();
    let down_count = spike_trains.iter()
        .filter(|s| s.get(1) == Some(&true))
        .count();
    (up_count, down_count)
}

fn analyze_by_confidence(predictions: &[(bool, bool, f64)]) {
    let thresholds = [0.0, 0.3, 0.5, 0.7];

    for &threshold in &thresholds {
        let filtered: Vec<_> = predictions.iter()
            .filter(|(_, _, conf)| *conf >= threshold)
            .collect();

        if filtered.is_empty() {
            continue;
        }

        let correct = filtered.iter()
            .filter(|(pred, actual, _)| pred == actual)
            .count();

        let accuracy = correct as f64 / filtered.len() as f64 * 100.0;

        println!("  Confidence >= {:.1}: {:.1}% accuracy ({} samples)",
            threshold, accuracy, filtered.len());
    }
}
