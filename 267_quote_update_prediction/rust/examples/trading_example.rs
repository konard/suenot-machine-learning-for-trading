//! # Quote Update Prediction - Trading Example
//!
//! This example demonstrates how to:
//! 1. Fetch live order-book data from Bybit
//! 2. Extract quote-update features
//! 3. Train a linear predictor on synthetic data
//! 4. Score incoming snapshots in real time

use anyhow::Result;
use quote_update_prediction::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Quote Update Prediction - Bybit Live Demo ===\n");

    // ------------------------------------------------------------------
    // Step 1: Train on synthetic data (stand-in for historical L2 data)
    // ------------------------------------------------------------------
    println!("[1] Generating synthetic order-book snapshots...");
    let train_snaps = generate_synthetic_snapshots(500, 50_000.0, 0.01);
    let x_train = build_feature_matrix(&train_snaps);
    let y_train = generate_labels(&train_snaps, 0.5);
    let (x_norm, means, stds) = z_normalise(&x_train);

    println!(
        "    {} samples, {} features",
        x_norm.nrows(),
        x_norm.ncols()
    );

    let mut model = LinearQuotePredictor::new(QuoteFeatures::NUM_FEATURES, 0.001);
    let epochs = 20;
    for epoch in 0..epochs {
        model.train_epoch(&x_norm, &y_train);
        if (epoch + 1) % 5 == 0 {
            let acc = model.accuracy(&x_norm, &y_train);
            println!("    Epoch {:>3}: accuracy = {:.2}%", epoch + 1, acc * 100.0);
        }
    }

    println!("\n    Model weights: {:?}", model.weights);
    println!("    Model bias:    {:.6}", model.bias);

    // ------------------------------------------------------------------
    // Step 2: Fetch live order-book snapshots from Bybit
    // ------------------------------------------------------------------
    println!("\n[2] Fetching live order-book from Bybit (BTCUSDT)...");

    let mut live_snapshots: Vec<OrderBookSnapshot> = Vec::new();
    let n_fetches = 5;

    for i in 0..n_fetches {
        match fetch_bybit_orderbook("BTCUSDT", 10).await {
            Ok(snap) => {
                println!(
                    "    Snapshot {}: bid={:.2} ask={:.2} spread={:.4} imb={:.4}",
                    i + 1,
                    snap.best_bid,
                    snap.best_ask,
                    snap.spread(),
                    snap.imbalance()
                );
                live_snapshots.push(snap);
            }
            Err(e) => {
                println!("    Warning: could not fetch snapshot {}: {}", i + 1, e);
            }
        }
        // Small delay between fetches
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    // ------------------------------------------------------------------
    // Step 3: Score live snapshots
    // ------------------------------------------------------------------
    if live_snapshots.len() >= 2 {
        println!("\n[3] Scoring live quote updates...");
        for i in 1..live_snapshots.len() {
            let feat = extract_features(&live_snapshots[i - 1], &live_snapshots[i]);
            let mut feat_arr = feat.to_array();
            // normalise using training stats
            for j in 0..feat_arr.len() {
                let std = if stds[j] == 0.0 { 1.0 } else { stds[j] };
                feat_arr[j] = (feat_arr[j] - means[j]) / std;
            }
            let raw_score = model.score(&feat_arr);
            let direction = model.predict(&feat_arr);
            println!(
                "    Update {}->{}: score={:.4}, prediction={:?}, imbalance={:.4}, mid_ret={:.2}bps",
                i, i + 1, raw_score, direction, feat.imbalance, feat.mid_return_bps
            );
        }
    } else {
        println!("\n[3] Not enough live snapshots to score (need >= 2).");
    }

    // ------------------------------------------------------------------
    // Step 4: Summary
    // ------------------------------------------------------------------
    println!("\n[4] Summary");
    println!("    - Trained linear model on {} synthetic samples", x_norm.nrows());
    println!("    - Fetched {} live snapshots from Bybit", live_snapshots.len());
    if !live_snapshots.is_empty() {
        let last = live_snapshots.last().unwrap();
        println!(
            "    - Latest mid-price: {:.2} (spread: {:.4})",
            last.mid_price(),
            last.spread()
        );
    }
    println!("\n=== Done ===");
    Ok(())
}
