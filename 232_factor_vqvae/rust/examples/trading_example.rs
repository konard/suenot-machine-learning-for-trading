//! Trading example: VQ-VAE market regime discovery using Bybit data.
//!
//! This example:
//! 1. Fetches BTCUSDT and ETHUSDT daily klines from Bybit
//! 2. Computes market features (returns, volatility, volume ratio, range, OC spread)
//! 3. Trains a VQ-VAE with K=8 codebook entries
//! 4. Maps each day to a regime (codebook entry)
//! 5. Analyses regime characteristics
//! 6. Generates regime-conditional synthetic data

use factor_vqvae::*;
use ndarray::Array1;

fn main() -> anyhow::Result<()> {
    println!("=== Factor VQ-VAE: Market Regime Discovery ===\n");

    // ---- 1. Fetch data from Bybit ----
    println!("Fetching BTCUSDT daily klines from Bybit...");
    let btc_klines = match fetch_bybit_klines("BTCUSDT", 200) {
        Ok(k) => {
            println!("  Fetched {} BTCUSDT klines", k.len());
            k
        }
        Err(e) => {
            println!("  Warning: Could not fetch BTCUSDT data: {}", e);
            println!("  Using synthetic data instead.\n");
            generate_synthetic_klines(200)
        }
    };

    println!("Fetching ETHUSDT daily klines from Bybit...");
    let eth_klines = match fetch_bybit_klines("ETHUSDT", 200) {
        Ok(k) => {
            println!("  Fetched {} ETHUSDT klines", k.len());
            k
        }
        Err(e) => {
            println!("  Warning: Could not fetch ETHUSDT data: {}", e);
            println!("  Using synthetic data instead.\n");
            generate_synthetic_klines(200)
        }
    };

    // ---- 2. Compute features ----
    let window = 10;
    println!("\nComputing features (window={})...", window);

    let btc_features = compute_features(&btc_klines, window);
    let eth_features = compute_features(&eth_klines, window);

    println!("  BTC feature vectors: {}", btc_features.len());
    println!("  ETH feature vectors: {}", eth_features.len());

    // Combine features from both assets.
    let mut all_features = btc_features.clone();
    all_features.extend(eth_features.clone());

    if all_features.is_empty() {
        println!("No features computed. Exiting.");
        return Ok(());
    }

    // Normalise features.
    let (normalised, means, stds) = normalise_features(&all_features);
    println!(
        "  Total normalised feature vectors: {}",
        normalised.len()
    );
    println!("  Feature means: {:?}", means);
    println!("  Feature stds:  {:?}", stds);

    // ---- 3. Train VQ-VAE ----
    let num_codes = 8;
    let config = VqVaeConfig {
        input_dim: 5,
        hidden_dim: 32,
        embedding_dim: 8,
        num_codes,
        beta: 0.25,
        gamma: 0.99,
        learning_rate: 1e-3,
        num_epochs: 200,
    };

    println!("\nTraining VQ-VAE (K={}, epochs={})...", num_codes, config.num_epochs);
    let mut model = VqVae::new(config.clone());
    let (losses, assignments) = model.train(&normalised);

    println!("  Initial loss: {:.6}", losses[0]);
    println!("  Final loss:   {:.6}", losses[losses.len() - 1]);

    // ---- 4. Regime assignments ----
    println!("\n=== Regime Assignments ===");

    // Show first 20 assignments for BTC.
    let btc_count = btc_features.len().min(normalised.len());
    println!("\nBTC regime assignments (first 20 days):");
    for (i, &regime) in assignments.iter().take(btc_count.min(20)).enumerate() {
        println!("  Day {:3}: Regime {}", i + 1, regime);
    }

    // Regime distribution.
    let mut regime_counts = vec![0usize; num_codes];
    for &a in &assignments {
        regime_counts[a] += 1;
    }
    println!("\nRegime distribution:");
    for (k, count) in regime_counts.iter().enumerate() {
        let pct = 100.0 * *count as f64 / assignments.len() as f64;
        let bar: String = std::iter::repeat('#').take((*count * 40 / assignments.len()).max(0)).collect();
        println!("  Regime {}: {:4} ({:5.1}%) {}", k, count, pct, bar);
    }

    // ---- 5. Regime characteristics ----
    println!("\n=== Regime Characteristics ===");
    let stats = analyse_regimes(&all_features, &assignments, num_codes);

    println!(
        "{:<10} {:<8} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Regime", "Count", "Mean Ret", "Volatility", "Vol Ratio", "Range", "OC Spread"
    );
    println!("{}", "-".repeat(78));

    for s in &stats {
        println!(
            "{:<10} {:<8} {:>12.6} {:>12.6} {:>12.4} {:>12.6} {:>12.6}",
            s.regime_id,
            s.count,
            s.mean_return,
            s.mean_volatility,
            s.mean_volume_ratio,
            s.mean_range,
            s.mean_oc_spread,
        );
    }

    // Classify regimes.
    println!("\nRegime interpretations:");
    for s in &stats {
        if s.count == 0 {
            println!("  Regime {}: UNUSED", s.regime_id);
            continue;
        }
        let label = classify_regime(s);
        println!(
            "  Regime {}: {} (return={:.4}, vol={:.4}, n={})",
            s.regime_id, label, s.mean_return, s.mean_volatility, s.count
        );
    }

    // ---- 6. Generate regime-conditional synthetic data ----
    println!("\n=== Regime-Conditional Synthetic Data ===");
    for k in 0..num_codes {
        if regime_counts[k] == 0 {
            continue;
        }
        println!("\nRegime {} — 3 synthetic samples:", k);
        for s in 0..3 {
            let sample = model.generate_sample(k, 0.3);
            // Denormalise.
            let denorm: Vec<f64> = sample
                .iter()
                .zip(means.iter())
                .zip(stds.iter())
                .map(|((v, m), s)| v * s + m)
                .collect();
            println!(
                "  Sample {}: ret={:.5}, vol={:.5}, vol_ratio={:.3}, range={:.5}, oc={:.5}",
                s + 1,
                denorm[0],
                denorm[1],
                denorm[2],
                denorm[3],
                denorm[4],
            );
        }
    }

    // ---- 7. Codebook visualisation ----
    println!("\n=== Codebook Entries (embedding space) ===");
    for (k, entry) in model.codebook.entries.iter().enumerate() {
        let vals: Vec<String> = entry.iter().map(|v| format!("{:.3}", v)).collect();
        println!("  e_{}: [{}]", k, vals.join(", "));
    }

    println!("\nDone!");
    Ok(())
}

/// Classify a regime based on its statistics.
fn classify_regime(s: &RegimeStats) -> &'static str {
    if s.mean_return > 0.01 && s.mean_volatility < 0.02 {
        "Calm Bull Market"
    } else if s.mean_return > 0.01 && s.mean_volatility >= 0.02 {
        "Volatile Rally"
    } else if s.mean_return < -0.01 && s.mean_volatility > 0.02 {
        "Crisis / Sell-off"
    } else if s.mean_return < -0.01 && s.mean_volatility <= 0.02 {
        "Slow Decline"
    } else if s.mean_volatility > 0.03 {
        "High Volatility Regime"
    } else if s.mean_volume_ratio > 1.5 {
        "High Volume Activity"
    } else if s.mean_volatility < 0.01 {
        "Low Volatility / Quiet"
    } else {
        "Mixed / Transitional"
    }
}

/// Generate synthetic kline data (fallback when Bybit API is not reachable).
fn generate_synthetic_klines(n: usize) -> Vec<Kline> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut klines = Vec::with_capacity(n);

    for i in 0..n {
        let ret = rng.gen_range(-0.05..0.05);
        let close = price * (1.0 + ret);
        let high = close * (1.0 + rng.gen_range(0.0..0.03));
        let low = close * (1.0 - rng.gen_range(0.0..0.03));
        let volume = rng.gen_range(500.0..5000.0);

        klines.push(Kline {
            open_time: 1700000000 + (i as u64) * 86400,
            open: price,
            high,
            low,
            close,
            volume,
        });
        price = close;
    }

    klines
}
