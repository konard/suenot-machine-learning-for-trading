//! Drift Detection Example
//!
//! Demonstrates concept drift detection in cryptocurrency markets using ADWIN and DDM.
//!
//! Run with: cargo run --example drift_detection

use online_learning::api::BybitClient;
use online_learning::drift::{DriftDetector, ADWIN, DDM};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Concept Drift Detection Demo ===\n");

    // Fetch data
    let client = BybitClient::new();
    let symbol = "BTCUSDT";

    println!("Fetching {} data from Bybit...", symbol);
    let candles = client.get_klines(symbol, "1h", 1000).await?;
    println!("Fetched {} candles\n", candles.len());

    // Calculate returns
    let returns: Vec<f64> = candles
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect();

    println!("Calculated {} returns\n", returns.len());

    // === ADWIN Detection ===
    println!("=== ADWIN Drift Detection ===\n");

    let mut adwin = ADWIN::new(0.002).with_min_samples(30);
    let mut adwin_drifts: Vec<usize> = Vec::new();

    for (i, &ret) in returns.iter().enumerate() {
        // ADWIN works on error/loss values, use absolute return as proxy
        if adwin.update(ret.abs()) {
            adwin_drifts.push(i);
            println!(
                "  ADWIN drift at index {} (price: {:.2}, return: {:.4}%)",
                i,
                candles[i + 1].close,
                ret * 100.0
            );
        }
    }

    println!("\nADWIN Summary:");
    println!("  Total drifts: {}", adwin_drifts.len());
    println!(
        "  Drift frequency: {:.2}%",
        adwin_drifts.len() as f64 / returns.len() as f64 * 100.0
    );
    println!("  Current window width: {}", adwin.window_width());
    println!("  Current mean: {:.6}", adwin.mean());

    // === DDM Detection ===
    println!("\n=== DDM Drift Detection ===\n");

    let mut ddm = DDM::new(30);
    let mut ddm_drifts: Vec<usize> = Vec::new();
    let mut ddm_warnings: Vec<usize> = Vec::new();

    for (i, &ret) in returns.iter().enumerate() {
        // Use prediction error (treating constant prediction as baseline)
        let error = ret.abs(); // Simple baseline error

        if ddm.update(error) {
            ddm_drifts.push(i);
            println!(
                "  DDM DRIFT at index {} (price: {:.2}, return: {:.4}%)",
                i,
                candles[i + 1].close,
                ret * 100.0
            );
        } else if ddm.in_warning() {
            if ddm_warnings.last() != Some(&i) {
                ddm_warnings.push(i);
            }
        }
    }

    println!("\nDDM Summary:");
    println!("  Total drifts: {}", ddm_drifts.len());
    println!("  Total warnings: {}", ddm_warnings.len());
    println!("  Current error rate: {:.6}", ddm.error_rate());

    // === Analyze Drift Patterns ===
    println!("\n=== Drift Analysis ===\n");

    // Find periods between drifts
    if adwin_drifts.len() > 1 {
        let gaps: Vec<usize> = adwin_drifts.windows(2).map(|w| w[1] - w[0]).collect();

        let avg_gap = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;
        let max_gap = *gaps.iter().max().unwrap_or(&0);
        let min_gap = *gaps.iter().min().unwrap_or(&0);

        println!("ADWIN Drift Intervals:");
        println!("  Average period between drifts: {:.1} hours", avg_gap);
        println!("  Longest stable period: {} hours", max_gap);
        println!("  Shortest stable period: {} hours", min_gap);
    }

    // === Volatility Regimes ===
    println!("\n=== Volatility Regime Detection ===\n");

    // Rolling volatility
    let window = 24;
    let mut volatilities: Vec<f64> = Vec::new();

    for i in window..returns.len() {
        let window_returns = &returns[i - window..i];
        let mean = window_returns.iter().sum::<f64>() / window as f64;
        let var = window_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;
        volatilities.push(var.sqrt());
    }

    // Detect volatility regime changes using ADWIN
    let mut vol_adwin = ADWIN::new(0.01);
    let mut vol_changes: Vec<(usize, f64)> = Vec::new();

    for (i, &vol) in volatilities.iter().enumerate() {
        if vol_adwin.update(vol) {
            vol_changes.push((i + window, vol));
        }
    }

    println!("Volatility Regime Changes:");
    for (i, vol) in vol_changes.iter().take(10) {
        println!(
            "  Index {}: Volatility = {:.4}%",
            i,
            vol * 100.0
        );
    }

    if vol_changes.len() > 10 {
        println!("  ... and {} more", vol_changes.len() - 10);
    }

    // === Model Adaptation Strategy ===
    println!("\n=== Recommended Adaptation Strategy ===\n");

    let drift_rate = adwin_drifts.len() as f64 / returns.len() as f64;

    if drift_rate > 0.05 {
        println!("High drift rate detected ({:.1}%):", drift_rate * 100.0);
        println!("  - Use aggressive learning rate (0.1)");
        println!("  - Short training window (50-100 samples)");
        println!("  - Consider model reset on drift detection");
    } else if drift_rate > 0.01 {
        println!("Moderate drift rate detected ({:.1}%):", drift_rate * 100.0);
        println!("  - Use moderate learning rate (0.01)");
        println!("  - Medium training window (100-200 samples)");
        println!("  - Increase learning rate on drift warning");
    } else {
        println!("Low drift rate detected ({:.1}%):", drift_rate * 100.0);
        println!("  - Use conservative learning rate (0.001)");
        println!("  - Can use longer training window (200+ samples)");
        println!("  - Standard online learning should suffice");
    }

    println!("\nNote: In production, combine drift detection with:");
    println!("  1. Adaptive learning rate adjustment");
    println!("  2. Model ensemble with diverse base learners");
    println!("  3. Periodic validation against out-of-sample data");

    Ok(())
}
