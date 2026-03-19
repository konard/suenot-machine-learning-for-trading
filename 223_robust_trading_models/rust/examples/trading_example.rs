//! # Robust Trading Models - Trading Example
//!
//! Fetches BTCUSDT data from Bybit, trains standard vs robust models,
//! tests across simulated market regimes, and shows a robustness comparison table.
//!
//! Run with: `cargo run --example trading_example`

use robust_trading_models::*;

fn main() {
    println!("=== Robust Trading Models ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch live data from Bybit
    // -----------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT hourly data from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("    Fetched {} candles", c.len());
            if let (Some(first), Some(last)) = (c.first(), c.last()) {
                println!(
                    "    Price range: {:.2} - {:.2}",
                    first.close, last.close
                );
            }
            c
        }
        Err(e) => {
            println!("    Failed to fetch from Bybit: {}", e);
            println!("    Using simulated data instead.\n");
            Vec::new()
        }
    };

    // -----------------------------------------------------------------------
    // Step 2: Prepare training data
    // -----------------------------------------------------------------------
    println!("\n[2] Preparing training data...");

    let (mut train_samples, data_source) = if candles.len() >= 50 {
        // Use real Bybit data
        let samples = candles_to_samples(&candles, 20);
        println!("    Generated {} samples from Bybit data", samples.len());

        // Show regime distribution
        let regimes = label_regimes(&candles, 20);
        let mut regime_counts = std::collections::HashMap::new();
        for (_, regime) in &regimes {
            *regime_counts.entry(format!("{}", regime)).or_insert(0) += 1;
        }
        println!("    Regime distribution in data:");
        for (regime, count) in &regime_counts {
            println!("      {}: {}", regime, count);
        }

        (samples, "Bybit BTCUSDT")
    } else {
        // Use simulated data from multiple regimes
        let configs = default_regime_configs();
        let mut all_samples = Vec::new();
        for config in &configs {
            let prices = simulate_regime(config, 300, 42);
            let samples = prices_to_samples(&prices, 20);
            println!(
                "    {} regime: {} samples",
                config.regime,
                samples.len()
            );
            all_samples.extend(samples);
        }
        println!("    Total simulated samples: {}", all_samples.len());
        (all_samples, "Simulated multi-regime")
    };

    // Apply robust feature engineering
    rank_transform(&mut train_samples);
    clip_features(&mut train_samples, 3.0);
    println!("    Applied rank transform and clipping");
    println!("    Data source: {}", data_source);

    let n_features = if train_samples.is_empty() {
        5
    } else {
        train_samples[0].features.len()
    };

    // -----------------------------------------------------------------------
    // Step 3: Train models
    // -----------------------------------------------------------------------
    println!("\n[3] Training models...");

    // Standard model (ERM baseline)
    println!("    Training standard (ERM) model...");
    let standard_model = LinearModel::train_standard(&train_samples, n_features, 0.01, 30, 42);

    // DRO-robust model
    println!("    Training DRO-robust model (alpha=0.3)...");
    let dro_model = train_robust_dro(&train_samples, n_features, 0.01, 30, 0.3, 42);

    // Noise-injection model
    println!("    Training noise-injection model (sigma=0.15)...");
    let noise_model =
        train_with_noise_injection(&train_samples, n_features, 0.01, 30, 0.15, 42);

    // Diverse ensemble
    println!("    Training diverse ensemble (5 models)...");
    let ensemble = EnsembleModel::train_diverse(&train_samples, n_features, 5, 0.01, 30, 0.3, 42);

    // Fully robust model (DRO + noise + ensemble combined)
    println!("    Training fully robust model...");
    let robust_ensemble = train_fully_robust(&train_samples, n_features, 42);

    println!("    All models trained.");

    // -----------------------------------------------------------------------
    // Step 4: Evaluate robustness
    // -----------------------------------------------------------------------
    println!("\n[4] Evaluating robustness across regimes...\n");

    let regime_configs = default_regime_configs();
    let perturbation_std = 0.2;

    // Evaluate each model
    let mut reports: Vec<(&str, RobustnessReport)> = Vec::new();

    let report = evaluate_robustness(
        &|f: &[f64]| standard_model.predict(f),
        &train_samples, &regime_configs, perturbation_std, 42,
    );
    reports.push(("Standard (ERM)", report));

    let report = evaluate_robustness(
        &|f: &[f64]| dro_model.predict(f),
        &train_samples, &regime_configs, perturbation_std, 42,
    );
    reports.push(("DRO-Robust", report));

    let report = evaluate_robustness(
        &|f: &[f64]| noise_model.predict(f),
        &train_samples, &regime_configs, perturbation_std, 42,
    );
    reports.push(("Noise-Injection", report));

    let report = evaluate_robustness(
        &|f: &[f64]| ensemble.predict(f),
        &train_samples, &regime_configs, perturbation_std, 42,
    );
    reports.push(("Diverse Ensemble", report));

    let report = evaluate_robustness(
        &|f: &[f64]| robust_ensemble.predict(f),
        &train_samples, &regime_configs, perturbation_std, 42,
    );
    reports.push(("Fully Robust", report));

    // -----------------------------------------------------------------------
    // Step 5: Display comparison table
    // -----------------------------------------------------------------------
    println!("╔══════════════════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╗");
    println!(
        "║ {:<20} ║ {:>8} ║ {:>8} ║ {:>8} ║ {:>8} ║ {:>8} ║ {:>8} ║ {:>8} ║",
        "Model", "Clean", "Perturb", "Bull", "Bear", "Sideways", "Crash", "Gap"
    );
    println!("╠══════════════════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╣");

    for (name, report) in &reports {
        let bull_acc = report
            .regime_accuracies
            .iter()
            .find(|(r, _)| r == "Bull")
            .map(|(_, a)| *a)
            .unwrap_or(0.0);
        let bear_acc = report
            .regime_accuracies
            .iter()
            .find(|(r, _)| r == "Bear")
            .map(|(_, a)| *a)
            .unwrap_or(0.0);
        let sideways_acc = report
            .regime_accuracies
            .iter()
            .find(|(r, _)| r == "Sideways")
            .map(|(_, a)| *a)
            .unwrap_or(0.0);
        let crash_acc = report
            .regime_accuracies
            .iter()
            .find(|(r, _)| r == "Crash")
            .map(|(_, a)| *a)
            .unwrap_or(0.0);

        println!(
            "║ {:<20} ║ {:>7.1}% ║ {:>7.1}% ║ {:>7.1}% ║ {:>7.1}% ║ {:>7.1}% ║ {:>7.1}% ║ {:>7.1}% ║",
            name,
            report.clean_accuracy * 100.0,
            report.perturbed_accuracy * 100.0,
            bull_acc * 100.0,
            bear_acc * 100.0,
            sideways_acc * 100.0,
            crash_acc * 100.0,
            report.robustness_gap * 100.0,
        );
    }

    println!("╚══════════════════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╝");

    // -----------------------------------------------------------------------
    // Step 6: Cross-regime generalization matrix
    // -----------------------------------------------------------------------
    println!("\n[5] Cross-regime generalization matrix (Standard model):\n");
    println!("    Train \\ Test   Bull     Bear     Sideways Crash");
    println!("    ─────────────────────────────────────────────────");

    let matrix = cross_regime_matrix(
        &|samples: &[Sample]| {
            let model = LinearModel::train_standard(samples, n_features, 0.01, 20, 42);
            Box::new(move |features: &[f64]| model.predict(features))
        },
        &regime_configs,
        300,
        200,
        20,
        42,
    );

    for row in &matrix {
        if let Some((train_regime, _, _)) = row.first() {
            print!("    {:<12}", train_regime);
            for (_, _, acc) in row {
                print!("  {:>6.1}%", acc * 100.0);
            }
            println!();
        }
    }

    println!("\n    (Diagonal = in-regime accuracy, off-diagonal = cross-regime transfer)");

    // -----------------------------------------------------------------------
    // Step 7: Summary
    // -----------------------------------------------------------------------
    println!("\n[6] Summary:");

    // Find best model by worst-regime accuracy
    if let Some((best_name, best_report)) = reports
        .iter()
        .max_by(|a, b| {
            a.1.worst_regime_accuracy
                .partial_cmp(&b.1.worst_regime_accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        println!(
            "    Best worst-regime accuracy: {} ({:.1}%)",
            best_name,
            best_report.worst_regime_accuracy * 100.0
        );
    }

    // Find smallest robustness gap
    if let Some((best_name, best_report)) = reports.iter().min_by(|a, b| {
        a.1.robustness_gap
            .abs()
            .partial_cmp(&b.1.robustness_gap.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "    Smallest robustness gap:    {} ({:.1}%)",
            best_name,
            best_report.robustness_gap * 100.0
        );
    }

    println!("\n    Robust models sacrifice some clean accuracy for better worst-case");
    println!("    performance — a worthwhile tradeoff in live trading.");
    println!("\n=== Done ===");
}
