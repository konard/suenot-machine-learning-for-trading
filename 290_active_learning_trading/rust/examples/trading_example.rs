//! # Trading Example: Active Learning for Efficient Data Labeling
//!
//! This example demonstrates the full active learning pipeline using real
//! market data from the Bybit API:
//!
//! 1. Fetch OHLCV candle data for BTCUSDT from Bybit
//! 2. Engineer features (returns, body ratio, shadows, volume change)
//! 3. Generate labels from forward returns
//! 4. Compare active learning strategies vs random baseline
//! 5. Evaluate trading performance with Sharpe ratio and drawdown
//!
//! Usage: cargo run --example trading_example

use anyhow::Result;
use active_learning_trading::*;
use ndarray::Array2;
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
// Bybit Data Fetching
// ---------------------------------------------------------------------------

fn fetch_bybit_candles(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    println!("Fetching data from Bybit: {}", url);

    let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

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

    candles.reverse(); // Bybit returns newest first
    println!("Fetched {} candles for {}", candles.len(), symbol);
    Ok(candles)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    println!("=== Active Learning for Trading ===\n");

    // ------------------------------------------------------------------
    // Step 1: Fetch market data
    // ------------------------------------------------------------------
    let candles = fetch_bybit_candles("BTCUSDT", "60", 200)?;

    let open: Vec<f64> = candles.iter().map(|c| c.open).collect();
    let high: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let low: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let close: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volume: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    // ------------------------------------------------------------------
    // Step 2: Engineer features
    // ------------------------------------------------------------------
    let features = compute_trading_features(&open, &high, &low, &close, &volume);
    println!(
        "Feature matrix: {} samples x {} features\n",
        features.nrows(),
        features.ncols()
    );

    // ------------------------------------------------------------------
    // Step 3: Generate labels
    // ------------------------------------------------------------------
    let all_labels = generate_trading_labels(&close[1..], 3, 0.005);
    let n = features.nrows().min(all_labels.len());
    let features = features.slice(ndarray::s![..n, ..]).to_owned();
    let labels = &all_labels[..n];

    // Class distribution
    let mut class_counts = [0usize; 3];
    for &l in labels {
        class_counts[l] += 1;
    }
    println!(
        "Label distribution: Short={}, Hold={}, Long={}\n",
        class_counts[0], class_counts[1], class_counts[2]
    );

    // ------------------------------------------------------------------
    // Step 4: Split train/test
    // ------------------------------------------------------------------
    let split_point = n * 7 / 10;
    let train_features = features.slice(ndarray::s![..split_point, ..]).to_owned();
    let train_labels: Vec<usize> = labels[..split_point].to_vec();
    let test_features = features.slice(ndarray::s![split_point.., ..]).to_owned();
    let test_labels: Vec<usize> = labels[split_point..].to_vec();

    println!(
        "Train: {} samples, Test: {} samples\n",
        train_features.nrows(),
        test_features.nrows()
    );

    // ------------------------------------------------------------------
    // Step 5: Compare active learning strategies
    // ------------------------------------------------------------------
    let strategies: Vec<(&str, AcquisitionFunction)> = vec![
        ("Random", AcquisitionFunction::Random),
        ("Uncertainty (Entropy)", AcquisitionFunction::UncertaintyEntropy),
        ("Uncertainty (Margin)", AcquisitionFunction::UncertaintyMargin),
        ("Uncertainty (Least Conf)", AcquisitionFunction::UncertaintyLeastConfidence),
        ("Diversity", AcquisitionFunction::Diversity),
    ];

    let config = ActiveLearningConfig {
        query_batch_size: 5,
        num_rounds: 8,
        initial_seed_size: 10,
        epochs_per_round: 10,
    };

    println!("--- Active Learning Comparison ---");
    println!(
        "Initial seed: {}, Query batch: {}, Rounds: {}\n",
        config.initial_seed_size, config.query_batch_size, config.num_rounds
    );

    for (name, acq) in &strategies {
        let start = Instant::now();
        let mut pipeline = ActiveLearningPipeline::new(
            config.clone(),
            *acq,
            train_features.nrows(),
        );
        let history = pipeline.run(
            &train_features,
            &train_labels,
            &test_features,
            &test_labels,
            3,
        );
        let elapsed = start.elapsed();

        let final_acc = history.last().copied().unwrap_or(0.0);
        let labeled_counts = pipeline.labeled_count_per_round();

        println!("[{}]", name);
        println!("  Final accuracy: {:.2}%", final_acc * 100.0);
        println!("  Final labeled:  {} samples", labeled_counts.last().copied().unwrap_or(0));
        println!("  Time:           {:.2?}", elapsed);
        println!(
            "  History:        {:?}",
            history.iter().map(|a| format!("{:.1}%", a * 100.0)).collect::<Vec<_>>()
        );
        println!();
    }

    // ------------------------------------------------------------------
    // Step 6: Query by Committee demonstration
    // ------------------------------------------------------------------
    println!("--- Query by Committee ---");
    let mut qbc = QueryByCommittee::new(5, features.ncols(), 3, 0.01);
    qbc.train_all(&train_features, &train_labels, 10);

    let qbc_selected = qbc.select(&test_features, 10);
    println!("QBC selected indices (most disagreed): {:?}", qbc_selected);

    // Show vote entropies for selected samples
    for &idx in qbc_selected.iter().take(5) {
        let x = test_features.row(idx).to_owned();
        let ve = qbc.vote_entropy(&x);
        let ce = qbc.consensus_entropy(&x);
        println!(
            "  Sample {}: vote_entropy={:.4}, consensus_entropy={:.4}",
            idx, ve, ce
        );
    }
    println!();

    // ------------------------------------------------------------------
    // Step 7: Trading simulation with best strategy
    // ------------------------------------------------------------------
    println!("--- Trading Simulation ---");
    let mut best_pipeline = ActiveLearningPipeline::new(
        config.clone(),
        AcquisitionFunction::UncertaintyEntropy,
        train_features.nrows(),
    );
    best_pipeline.run(
        &train_features,
        &train_labels,
        &test_features,
        &test_labels,
        3,
    );

    // Train final model on actively-selected data
    let num_features = train_features.ncols();
    let n_labeled = best_pipeline.labeled_indices.len();
    let active_features = Array2::from_shape_fn((n_labeled, num_features), |(i, j)| {
        train_features[[best_pipeline.labeled_indices[i], j]]
    });
    let active_labels: Vec<usize> = best_pipeline
        .labeled_indices
        .iter()
        .map(|&i| train_labels[i])
        .collect();

    let mut final_model = SimpleClassifier::new(num_features, 3, 0.01);
    final_model.train_batch(&active_features, &active_labels, 20);

    // Generate predictions on test set
    let predictions: Vec<usize> = test_features
        .axis_iter(ndarray::Axis(0))
        .map(|x| final_model.predict(&x.to_owned()))
        .collect();

    let test_close = &close[split_point + 1..];
    let pnl = simulate_trading(&predictions, test_close);

    // Compute returns from PnL
    let returns: Vec<f64> = pnl.windows(2).map(|w| w[1] - w[0]).collect();
    let sr = sharpe_ratio(&returns);
    let mdd = max_drawdown(&pnl);

    println!("  Labeled samples used: {} / {}", n_labeled, train_features.nrows());
    println!("  Test accuracy:  {:.2}%", final_model.accuracy(&test_features, &test_labels) * 100.0);
    println!("  Sharpe ratio:   {:.4}", sr);
    println!("  Max drawdown:   {:.4}", mdd);
    println!(
        "  Final PnL:      {:.4}",
        pnl.last().copied().unwrap_or(0.0)
    );

    // Compare with model trained on ALL data
    println!("\n--- Comparison: Full Data vs Active Learning ---");
    let mut full_model = SimpleClassifier::new(num_features, 3, 0.01);
    full_model.train_batch(&train_features, &train_labels, 20);
    let full_acc = full_model.accuracy(&test_features, &test_labels);
    let active_acc = final_model.accuracy(&test_features, &test_labels);
    println!(
        "  Full data ({} samples): {:.2}%",
        train_features.nrows(),
        full_acc * 100.0
    );
    println!(
        "  Active learning ({} samples): {:.2}%",
        n_labeled,
        active_acc * 100.0
    );
    println!(
        "  Data efficiency: {:.1}x ({:.0}% of data used)",
        train_features.nrows() as f64 / n_labeled as f64,
        n_labeled as f64 / train_features.nrows() as f64 * 100.0
    );

    println!("\n=== Active Learning Trading Example Complete ===");
    Ok(())
}
