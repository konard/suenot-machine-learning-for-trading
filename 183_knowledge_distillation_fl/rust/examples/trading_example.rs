//! # Trading Example: Knowledge Distillation in Federated Learning
//!
//! This example demonstrates the full FedMD pipeline using real market data
//! from the Bybit API:
//!
//! 1. Fetch OHLCV candle data for BTCUSDT from Bybit
//! 2. Engineer features (returns, body ratio, shadows, volume change)
//! 3. Generate labels from forward returns
//! 4. Train a teacher model on the full dataset
//! 5. Distill into student models using FedMD
//! 6. Compare teacher vs student accuracy and inference speed
//!
//! Usage: cargo run --example trading_example

use anyhow::Result;
use knowledge_distillation_fl::*;
use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Bybit API Data Structures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<serde_json::Value>>,
}

/// A single OHLCV candle.
#[derive(Debug, Clone)]
struct Candle {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

// ---------------------------------------------------------------------------
// Data Fetching
// ---------------------------------------------------------------------------

/// Fetch OHLCV kline data from Bybit API.
///
/// Endpoint: GET /v5/market/kline
/// Returns candles sorted by time ascending.
async fn fetch_bybit_candles(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    println!("Fetching data from Bybit API...");
    println!("  URL: {}", url);

    let client = reqwest::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("User-Agent", "ML-Trading-Bot/1.0")
        .send()
        .await?
        .json()
        .await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    open: row[1].as_str()?.parse().ok()?,
                    high: row[2].as_str()?.parse().ok()?,
                    low: row[3].as_str()?.parse().ok()?,
                    close: row[4].as_str()?.parse().ok()?,
                    volume: row[5].as_str()?.parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    candles.reverse();

    println!("  Fetched {} candles for {}", candles.len(), symbol);
    Ok(candles)
}

// ---------------------------------------------------------------------------
// Feature Engineering
// ---------------------------------------------------------------------------

/// Engineer features from OHLCV candles.
///
/// Features per candle:
/// 1. Price return: (close - open) / open
/// 2. Body ratio: |close - open| / (high - low) -- candle shape
/// 3. Upper shadow: (high - max(open, close)) / (high - low) -- rejection
/// 4. Volume change: volume[t] / volume[t-1] -- activity
fn engineer_features(candles: &[Candle]) -> Vec<Array1<f64>> {
    let mut features = Vec::new();

    for i in 1..candles.len() {
        let c = &candles[i];
        let prev = &candles[i - 1];

        let price_range = c.high - c.low;
        if price_range < 1e-10 || prev.volume < 1e-10 {
            continue;
        }

        let price_return = (c.close - c.open) / c.open;
        let body_ratio = (c.close - c.open).abs() / price_range;
        let upper_shadow = (c.high - c.close.max(c.open)) / price_range;
        let volume_change = (c.volume / prev.volume).ln(); // log ratio for stability

        features.push(Array1::from_vec(vec![
            price_return,
            body_ratio,
            upper_shadow,
            volume_change,
        ]));
    }

    features
}

/// Generate labels from forward returns.
///
/// - Class 0 (Down): next return < -threshold
/// - Class 1 (Flat): next return in [-threshold, +threshold]
/// - Class 2 (Up): next return > +threshold
fn generate_labels(candles: &[Candle], threshold: f64) -> Vec<usize> {
    let mut labels = Vec::new();

    for i in 1..candles.len().saturating_sub(1) {
        let current_close = candles[i].close;
        let next_close = candles[i + 1].close;
        let forward_return = (next_close - current_close) / current_close;

        let label = if forward_return < -threshold {
            0 // Down
        } else if forward_return > threshold {
            2 // Up
        } else {
            1 // Flat
        };
        labels.push(label);
    }

    labels
}

// ---------------------------------------------------------------------------
// Demo with Synthetic Data (fallback if API is unavailable)
// ---------------------------------------------------------------------------

/// Generate synthetic market-like data for demonstration.
fn generate_synthetic_data(
    num_samples: usize,
    num_features: usize,
    _num_classes: usize,
) -> (Vec<Array1<f64>>, Vec<usize>) {
    let mut rng = rand::thread_rng();
    let mut inputs = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let features: Vec<f64> = (0..num_features)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Simple rule: label based on weighted sum of features
        let score: f64 = features[0] * 0.5 + features[1] * 0.3
            - features[2] * 0.2
            + features.get(3).unwrap_or(&0.0) * 0.1;

        let label = if score < -0.2 {
            0
        } else if score > 0.2 {
            2
        } else {
            1
        };

        inputs.push(Array1::from_vec(features));
        labels.push(label);
    }

    (inputs, labels)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    println!("=======================================================");
    println!(" Knowledge Distillation in Federated Learning - Trading");
    println!("=======================================================\n");

    let num_features = 4;
    let num_classes = 3;
    let threshold = 0.001; // 0.1% threshold for label generation

    // Try to fetch real data from Bybit
    let (all_inputs, all_labels) =
        match fetch_bybit_candles("BTCUSDT", "5", 200).await {
            Ok(candles) if candles.len() > 10 => {
                println!("  Successfully fetched {} candles\n", candles.len());
                let features = engineer_features(&candles);
                let labels = generate_labels(&candles, threshold);
                let min_len = features.len().min(labels.len());
                (features[..min_len].to_vec(), labels[..min_len].to_vec())
            }
            Ok(_) => {
                println!("  Insufficient candles received, using synthetic data\n");
                generate_synthetic_data(150, num_features, num_classes)
            }
            Err(e) => {
                println!("  API fetch failed ({}), using synthetic data\n", e);
                generate_synthetic_data(150, num_features, num_classes)
            }
        };

    println!(
        "Dataset: {} samples, {} features, {} classes\n",
        all_inputs.len(),
        num_features,
        num_classes
    );

    // Split into train/test
    let split = (all_inputs.len() * 4) / 5;
    let train_inputs = all_inputs[..split].to_vec();
    let train_labels = all_labels[..split].to_vec();
    let test_inputs = all_inputs[split..].to_vec();
    let test_labels = all_labels[split..].to_vec();

    let train_targets: Vec<Vec<f64>> = train_labels
        .iter()
        .map(|&l| one_hot(l, num_classes))
        .collect();
    let test_targets: Vec<Vec<f64>> = test_labels
        .iter()
        .map(|&l| one_hot(l, num_classes))
        .collect();

    // -----------------------------------------------------------------------
    // Step 1: Train the Teacher Model
    // -----------------------------------------------------------------------
    println!("--- Step 1: Training Teacher Model ---");
    let mut teacher = TeacherModel::new(num_features, 32, 16, num_classes, 0.05);
    println!(
        "  Architecture: {} -> 32 -> 16 -> {}",
        num_features, num_classes
    );

    let start = Instant::now();
    teacher.train_batch(&train_inputs, &train_targets, 50);
    let teacher_train_time = start.elapsed();

    // Evaluate teacher
    let mut teacher_correct = 0;
    for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
        let pred = teacher.soft_predict(input, 1.0);
        if argmax(&pred) == argmax(target) {
            teacher_correct += 1;
        }
    }
    let teacher_acc = teacher_correct as f64 / test_inputs.len().max(1) as f64;
    println!(
        "  Teacher accuracy: {:.1}% ({}/{})",
        teacher_acc * 100.0,
        teacher_correct,
        test_inputs.len()
    );
    println!("  Training time: {:?}\n", teacher_train_time);

    // -----------------------------------------------------------------------
    // Step 2: Direct Student Training (baseline, no distillation)
    // -----------------------------------------------------------------------
    println!("--- Step 2: Training Baseline Student (no distillation) ---");
    let mut baseline_student = StudentModel::new(num_features, 8, num_classes, 0.05);
    println!(
        "  Architecture: {} -> 8 -> {} ({} params)",
        num_features,
        num_classes,
        baseline_student.num_parameters()
    );

    let start = Instant::now();
    for _ in 0..50 {
        for (input, target) in train_inputs.iter().zip(train_targets.iter()) {
            baseline_student.distillation_step(
                input,
                target,
                &vec![1.0 / num_classes as f64; num_classes],
                1.0,
                0.0, // alpha=0 -> pure hard label training
            );
        }
    }
    let baseline_train_time = start.elapsed();

    let mut baseline_correct = 0;
    for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
        let pred = baseline_student.predict(input);
        if argmax(&pred) == argmax(target) {
            baseline_correct += 1;
        }
    }
    let baseline_acc = baseline_correct as f64 / test_inputs.len().max(1) as f64;
    println!(
        "  Baseline student accuracy: {:.1}% ({}/{})",
        baseline_acc * 100.0,
        baseline_correct,
        test_inputs.len()
    );
    println!("  Training time: {:?}\n", baseline_train_time);

    // -----------------------------------------------------------------------
    // Step 3: Knowledge Distillation (single teacher -> student)
    // -----------------------------------------------------------------------
    println!("--- Step 3: Knowledge Distillation (T=3.0, alpha=0.7) ---");
    let temperature = 3.0;
    let alpha = 0.7;
    let mut distilled_student = StudentModel::new(num_features, 8, num_classes, 0.05);
    println!(
        "  Architecture: {} -> 8 -> {} ({} params)",
        num_features,
        num_classes,
        distilled_student.num_parameters()
    );

    // Generate teacher soft labels
    let teacher_soft_labels: Vec<Vec<f64>> = train_inputs
        .iter()
        .map(|x| teacher.soft_predict(x, temperature))
        .collect();

    let start = Instant::now();
    distilled_student.distill_batch(
        &train_inputs,
        &train_targets,
        &teacher_soft_labels,
        temperature,
        alpha,
        50,
    );
    let distill_train_time = start.elapsed();

    let mut distilled_correct = 0;
    for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
        let pred = distilled_student.predict(input);
        if argmax(&pred) == argmax(target) {
            distilled_correct += 1;
        }
    }
    let distilled_acc = distilled_correct as f64 / test_inputs.len().max(1) as f64;
    println!(
        "  Distilled student accuracy: {:.1}% ({}/{})",
        distilled_acc * 100.0,
        distilled_correct,
        test_inputs.len()
    );
    println!("  Training time: {:?}\n", distill_train_time);

    // -----------------------------------------------------------------------
    // Step 4: Federated Model Distillation (FedMD)
    // -----------------------------------------------------------------------
    println!("--- Step 4: Federated Model Distillation (3 clients) ---");

    let num_clients = 3;
    let samples_per_client = train_inputs.len() / num_clients;
    let mut rng = rand::thread_rng();

    // Split data among clients (simulating different trading venues)
    let venue_names = ["Venue-A (HFT)", "Venue-B (Mobile)", "Venue-C (Analysis)"];
    let hidden_sizes = [6, 10, 14]; // Different architectures per client

    let mut fed_clients: Vec<FedClient> = Vec::new();
    for i in 0..num_clients {
        let start_idx = i * samples_per_client;
        let end_idx = if i == num_clients - 1 {
            train_inputs.len()
        } else {
            (i + 1) * samples_per_client
        };

        let client_inputs = train_inputs[start_idx..end_idx].to_vec();
        let client_targets = train_targets[start_idx..end_idx].to_vec();

        let model = StudentModel::new(num_features, hidden_sizes[i], num_classes, 0.05);
        println!(
            "  {} -> {} -> {} -> {} ({} params, {} local samples)",
            venue_names[i],
            num_features,
            hidden_sizes[i],
            num_classes,
            model.num_parameters(),
            client_inputs.len()
        );

        fed_clients.push(FedClient {
            model,
            local_inputs: client_inputs,
            local_targets: client_targets,
        });
    }

    // Use a portion of training data as the "public" dataset
    let pub_size = train_inputs.len().min(50);
    let pub_indices: Vec<usize> = (0..pub_size)
        .map(|_| rng.gen_range(0..train_inputs.len()))
        .collect();
    let public_inputs: Vec<Array1<f64>> = pub_indices.iter().map(|&i| train_inputs[i].clone()).collect();
    let public_targets: Vec<Vec<f64>> = pub_indices.iter().map(|&i| train_targets[i].clone()).collect();

    // Create a fresh teacher for FedMD
    let fed_teacher = TeacherModel::new(num_features, 32, 16, num_classes, 0.02);

    let mut fedmd = FederatedDistillation::new(
        fed_teacher,
        fed_clients,
        public_inputs,
        public_targets,
        temperature,
        alpha,
    );

    println!("\n  Running FedMD...");
    let start = Instant::now();
    fedmd.run(20, 3, 3);
    let fedmd_time = start.elapsed();
    println!("  FedMD training time: {:?}\n", fedmd_time);

    // Evaluate FedMD clients
    println!("  FedMD Client Evaluation (on test set):");
    for (i, name) in venue_names.iter().enumerate() {
        let client = &fedmd.clients[i];
        let mut correct = 0;
        for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
            let pred = client.model.predict(input);
            if argmax(&pred) == argmax(target) {
                correct += 1;
            }
        }
        let acc = correct as f64 / test_inputs.len().max(1) as f64;
        println!(
            "    {}: {:.1}% ({}/{})",
            name,
            acc * 100.0,
            correct,
            test_inputs.len()
        );
    }

    // -----------------------------------------------------------------------
    // Step 5: Inference Speed Comparison
    // -----------------------------------------------------------------------
    println!("\n--- Step 5: Inference Speed Comparison ---");
    let test_sample = &test_inputs[0];
    let iterations = 10_000;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = teacher.soft_predict(test_sample, 1.0);
    }
    let teacher_ns = start.elapsed().as_nanos() / iterations as u128;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = distilled_student.predict(test_sample);
    }
    let student_ns = start.elapsed().as_nanos() / iterations as u128;

    println!("  Teacher inference: {} ns/sample", teacher_ns);
    println!("  Student inference: {} ns/sample", student_ns);
    println!(
        "  Speedup: {:.1}x\n",
        teacher_ns as f64 / student_ns.max(1) as f64
    );

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("=======================================================");
    println!("                    SUMMARY");
    println!("=======================================================");
    println!("  Teacher accuracy:           {:.1}%", teacher_acc * 100.0);
    println!(
        "  Baseline student accuracy:  {:.1}%",
        baseline_acc * 100.0
    );
    println!(
        "  Distilled student accuracy: {:.1}%",
        distilled_acc * 100.0
    );
    println!(
        "  Improvement from distillation: {:.1}pp",
        (distilled_acc - baseline_acc) * 100.0
    );
    println!(
        "  Student/Teacher param ratio: 1:{:.0}",
        (32.0 * num_features as f64 + 32.0 + 16.0 * 32.0 + 16.0 + num_classes as f64 * 16.0 + num_classes as f64)
            / distilled_student.num_parameters() as f64
    );
    println!(
        "  Inference speedup: {:.1}x",
        teacher_ns as f64 / student_ns.max(1) as f64
    );
    println!("=======================================================");

    Ok(())
}
