//! # Trading Example: Byzantine-Robust Federated Learning
//!
//! This example demonstrates the full Byzantine-robust FL pipeline using
//! real market data from the Bybit API:
//!
//! 1. Fetch OHLCV candle data for BTCUSDT from Bybit
//! 2. Engineer features (returns, body ratio, shadows, volume change)
//! 3. Generate labels from forward returns
//! 4. Train models with FedAvg (vulnerable) vs Krum/TrimmedMean/Median (robust)
//! 5. Simulate Byzantine attacks (sign-flip) on a fraction of clients
//! 6. Compare accuracy under attack for each aggregation method
//!
//! Usage: cargo run --example trading_example

use anyhow::Result;
use byzantine_robust_fl::*;
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
        let volume_change = (c.volume / prev.volume).ln();

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
// Evaluation Helper
// ---------------------------------------------------------------------------

/// Run a full Byzantine FL experiment with the given aggregation rule.
fn run_experiment(
    name: &str,
    rule: AggregationRule,
    train_inputs: &[Array1<f64>],
    train_targets: &[Vec<f64>],
    test_inputs: &[Array1<f64>],
    test_targets: &[Vec<f64>],
    num_features: usize,
    num_classes: usize,
    num_clients: usize,
    num_byzantine: usize,
    num_rounds: usize,
    local_epochs: usize,
) -> f64 {
    println!("\n  --- {} (Byzantine: {}/{}) ---", name, num_byzantine, num_clients);

    let chunk_size = train_inputs.len() / num_clients;
    let mut clients = Vec::new();

    for i in 0..num_clients {
        let start = i * chunk_size;
        let end = if i == num_clients - 1 {
            train_inputs.len()
        } else {
            (i + 1) * chunk_size
        };

        let is_byz = i >= num_clients - num_byzantine;

        clients.push(FLClient {
            model: SimpleModel::new(num_features, 16, num_classes, 0.02),
            local_inputs: train_inputs[start..end].to_vec(),
            local_targets: train_targets[start..end].to_vec(),
            is_byzantine: is_byz,
        });

        if is_byz {
            println!("    Client {} [BYZANTINE]", i);
        }
    }

    let global_model = SimpleModel::new(num_features, 16, num_classes, 0.02);
    let mut fl = ByzantineFL::new(global_model, clients, rule, num_byzantine);

    let start = Instant::now();
    fl.run(num_rounds, local_epochs);
    let elapsed = start.elapsed();

    let accuracy = fl.evaluate(test_inputs, test_targets);
    println!("    Accuracy: {:.1}%", accuracy * 100.0);
    println!("    Time: {:?}", elapsed);

    accuracy
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    println!("=======================================================");
    println!(" Byzantine-Robust Federated Learning for Trading");
    println!("=======================================================\n");

    let num_features = 4;
    let num_classes = 3;
    let threshold = 0.001;

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

    let num_clients = 7;
    let num_byzantine = 2; // ~30% Byzantine
    let num_rounds = 15;
    let local_epochs = 3;

    // -----------------------------------------------------------------------
    // Experiment 1: FedAvg (no Byzantine robustness)
    // -----------------------------------------------------------------------
    println!("=== Experiment: Comparing Aggregation Rules Under Attack ===");

    let acc_fedavg = run_experiment(
        "FedAvg (NO defense)",
        AggregationRule::FedAvg,
        &train_inputs,
        &train_targets,
        &test_inputs,
        &test_targets,
        num_features,
        num_classes,
        num_clients,
        num_byzantine,
        num_rounds,
        local_epochs,
    );

    // -----------------------------------------------------------------------
    // Experiment 2: Krum
    // -----------------------------------------------------------------------
    let acc_krum = run_experiment(
        "Krum",
        AggregationRule::Krum,
        &train_inputs,
        &train_targets,
        &test_inputs,
        &test_targets,
        num_features,
        num_classes,
        num_clients,
        num_byzantine,
        num_rounds,
        local_epochs,
    );

    // -----------------------------------------------------------------------
    // Experiment 3: Trimmed Mean
    // -----------------------------------------------------------------------
    let acc_trimmed = run_experiment(
        "Trimmed Mean",
        AggregationRule::TrimmedMean(num_byzantine),
        &train_inputs,
        &train_targets,
        &test_inputs,
        &test_targets,
        num_features,
        num_classes,
        num_clients,
        num_byzantine,
        num_rounds,
        local_epochs,
    );

    // -----------------------------------------------------------------------
    // Experiment 4: Median
    // -----------------------------------------------------------------------
    let acc_median = run_experiment(
        "Median",
        AggregationRule::Median,
        &train_inputs,
        &train_targets,
        &test_inputs,
        &test_targets,
        num_features,
        num_classes,
        num_clients,
        num_byzantine,
        num_rounds,
        local_epochs,
    );

    // -----------------------------------------------------------------------
    // Experiment 5: No Attack (baseline) with FedAvg
    // -----------------------------------------------------------------------
    let acc_baseline = run_experiment(
        "FedAvg (no attack baseline)",
        AggregationRule::FedAvg,
        &train_inputs,
        &train_targets,
        &test_inputs,
        &test_targets,
        num_features,
        num_classes,
        num_clients,
        0, // No Byzantine
        num_rounds,
        local_epochs,
    );

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=======================================================");
    println!("                    SUMMARY");
    println!("=======================================================");
    println!(
        "  Baseline (no attack):        {:.1}%",
        acc_baseline * 100.0
    );
    println!(
        "  FedAvg (under attack):       {:.1}%  <-- VULNERABLE",
        acc_fedavg * 100.0
    );
    println!(
        "  Krum (under attack):         {:.1}%",
        acc_krum * 100.0
    );
    println!(
        "  Trimmed Mean (under attack): {:.1}%",
        acc_trimmed * 100.0
    );
    println!(
        "  Median (under attack):       {:.1}%",
        acc_median * 100.0
    );
    println!("-------------------------------------------------------");
    println!(
        "  Attack config: {}/{} Byzantine clients (sign-flip x3)",
        num_byzantine, num_clients
    );
    println!("  Rounds: {}, Local epochs: {}", num_rounds, local_epochs);
    println!("=======================================================");

    Ok(())
}
