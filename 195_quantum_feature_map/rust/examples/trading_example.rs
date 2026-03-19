//! Trading Example: Quantum Feature Maps on Bybit BTCUSDT Data
//!
//! This example fetches BTCUSDT candle data from Bybit, engineers trading
//! features, applies different quantum feature maps, and compares the
//! resulting kernel matrices.

use quantum_feature_map::*;

fn main() -> Result<(), anyhow::Error> {
    println!("=== Quantum Feature Maps for Trading ===\n");

    // -----------------------------------------------------------------------
    // 1. Fetch data from Bybit
    // -----------------------------------------------------------------------
    println!("Fetching BTCUSDT 1h candles from Bybit...");
    let candles = match fetch_bybit_candles("BTCUSDT", "60", 100) {
        Ok(c) => {
            println!("Fetched {} candles.", c.len());
            c
        }
        Err(e) => {
            println!("Warning: Could not fetch from Bybit ({}). Using synthetic data.", e);
            generate_synthetic_candles(100)
        }
    };

    // -----------------------------------------------------------------------
    // 2. Engineer features
    // -----------------------------------------------------------------------
    println!("\nEngineering features...");
    let raw_features = engineer_features(&candles);
    println!(
        "Generated {} feature vectors with {} features each.",
        raw_features.len(),
        raw_features.first().map_or(0, |f| f.len())
    );

    // Normalize to [0, pi]
    let features = normalize_features(&raw_features);
    let n_features = features.first().map_or(0, |f| f.len());
    println!("Normalized features to [0, pi].");

    if features.is_empty() || n_features == 0 {
        println!("No features to process. Exiting.");
        return Ok(());
    }

    // Use a subset for kernel computation (first 30 points)
    let subset: Vec<Vec<f64>> = features.iter().take(30).cloned().collect();
    let n_data = subset.len();
    println!("Using {} data points for kernel computation.\n", n_data);

    // -----------------------------------------------------------------------
    // 3. Create feature maps
    // -----------------------------------------------------------------------
    let angle_map = AngleMap::new(n_features);
    let zz_map = ZZFeatureMap::new(n_features, 2);
    let iqp_map = IQPFeatureMap::new(n_features, 1);

    // -----------------------------------------------------------------------
    // 4. Compute kernel matrices
    // -----------------------------------------------------------------------
    println!("--- Kernel Matrix Analysis ---\n");

    println!("Computing AngleMap kernel matrix...");
    let km_angle = angle_map.kernel_matrix(&subset);
    let stats_angle = kernel_stats(&km_angle);
    println!(
        "  Off-diagonal mean: {:.4}, std: {:.4}",
        stats_angle.off_diag_mean, stats_angle.off_diag_std
    );

    println!("Computing ZZFeatureMap (depth=2) kernel matrix...");
    let km_zz = zz_map.kernel_matrix(&subset);
    let stats_zz = kernel_stats(&km_zz);
    println!(
        "  Off-diagonal mean: {:.4}, std: {:.4}",
        stats_zz.off_diag_mean, stats_zz.off_diag_std
    );

    println!("Computing IQPFeatureMap (depth=1) kernel matrix...");
    let km_iqp = iqp_map.kernel_matrix(&subset);
    let stats_iqp = kernel_stats(&km_iqp);
    println!(
        "  Off-diagonal mean: {:.4}, std: {:.4}",
        stats_iqp.off_diag_mean, stats_iqp.off_diag_std
    );

    // -----------------------------------------------------------------------
    // 5. Expressibility analysis
    // -----------------------------------------------------------------------
    println!("\n--- Expressibility Analysis ---\n");

    let expr_angle = estimate_expressibility(&angle_map, n_features, 500, 50);
    println!("AngleMap expressibility (KL div): {:.4}", expr_angle);

    let expr_zz = estimate_expressibility(&zz_map, n_features, 500, 50);
    println!("ZZFeatureMap expressibility (KL div): {:.4}", expr_zz);

    let expr_iqp = estimate_expressibility(&iqp_map, n_features, 500, 50);
    println!("IQPFeatureMap expressibility (KL div): {:.4}", expr_iqp);

    // -----------------------------------------------------------------------
    // 6. Entangling capability
    // -----------------------------------------------------------------------
    println!("\n--- Entangling Capability ---\n");

    let ent_angle = estimate_entangling_capability(&angle_map, n_features, 200);
    println!("AngleMap entangling capability: {:.4}", ent_angle);

    let ent_zz = estimate_entangling_capability(&zz_map, n_features, 200);
    println!("ZZFeatureMap entangling capability: {:.4}", ent_zz);

    let ent_iqp = estimate_entangling_capability(&iqp_map, n_features, 200);
    println!("IQPFeatureMap entangling capability: {:.4}", ent_iqp);

    // -----------------------------------------------------------------------
    // 7. Summary
    // -----------------------------------------------------------------------
    println!("\n--- Summary ---\n");
    println!(
        "{:<20} {:>12} {:>12} {:>14} {:>12}",
        "Feature Map", "Off-diag mu", "Off-diag std", "Expressibility", "Entanglement"
    );
    println!("{:-<72}", "");
    println!(
        "{:<20} {:>12.4} {:>12.4} {:>14.4} {:>12.4}",
        "AngleMap",
        stats_angle.off_diag_mean,
        stats_angle.off_diag_std,
        expr_angle,
        ent_angle
    );
    println!(
        "{:<20} {:>12.4} {:>12.4} {:>14.4} {:>12.4}",
        "ZZFeatureMap(d=2)",
        stats_zz.off_diag_mean,
        stats_zz.off_diag_std,
        expr_zz,
        ent_zz
    );
    println!(
        "{:<20} {:>12.4} {:>12.4} {:>14.4} {:>12.4}",
        "IQPFeatureMap(d=1)",
        stats_iqp.off_diag_mean,
        stats_iqp.off_diag_std,
        expr_iqp,
        ent_iqp
    );

    println!("\n--- Interpretation ---\n");
    println!("Best kernel structure (highest off-diagonal std = most differentiated):");
    let best_km = if stats_angle.off_diag_std >= stats_zz.off_diag_std
        && stats_angle.off_diag_std >= stats_iqp.off_diag_std
    {
        "AngleMap"
    } else if stats_zz.off_diag_std >= stats_iqp.off_diag_std {
        "ZZFeatureMap"
    } else {
        "IQPFeatureMap"
    };
    println!("  -> {}", best_km);

    println!("\nBest expressibility (lowest KL divergence):");
    let best_expr = if expr_angle <= expr_zz && expr_angle <= expr_iqp {
        "AngleMap"
    } else if expr_zz <= expr_iqp {
        "ZZFeatureMap"
    } else {
        "IQPFeatureMap"
    };
    println!("  -> {}", best_expr);

    println!("\nHighest entangling capability:");
    let best_ent = if ent_angle >= ent_zz && ent_angle >= ent_iqp {
        "AngleMap"
    } else if ent_zz >= ent_iqp {
        "ZZFeatureMap"
    } else {
        "IQPFeatureMap"
    };
    println!("  -> {}", best_ent);

    println!("\nDone!");
    Ok(())
}

/// Generate synthetic OHLCV candles for testing when Bybit API is unavailable.
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    for i in 0..n {
        let ret: f64 = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = open * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.001..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.001..0.01));
        let volume = rng.gen_range(500.0..2000.0);

        candles.push(Candle {
            timestamp: (i as u64) * 3600000,
            open,
            high,
            low,
            close,
            volume,
        });
        price = close;
    }
    candles
}
