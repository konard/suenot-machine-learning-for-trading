//! Trading Example: Uncertainty-Aware Position Sizing
//!
//! This example demonstrates:
//! 1. Fetching BTCUSDT daily candles from Bybit
//! 2. Extracting features from price data
//! 3. Building an ensemble model
//! 4. Generating predictions with uncertainty estimates
//! 5. Risk-adjusted position sizing based on uncertainty

use uncertainty_quantification::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Uncertainty Quantification Trading Example ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // -----------------------------------------------------------------------
    println!("Fetching BTCUSDT daily candles from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "D", 100).await {
        Ok(c) => {
            println!("  Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            println!("  Could not fetch live data ({}), using synthetic data.", e);
            generate_synthetic_candles(100)
        }
    };

    if candles.len() < 25 {
        println!("Not enough candle data. Exiting.");
        return Ok(());
    }

    let last = &candles[candles.len() - 1];
    println!(
        "  Latest close: {:.2}  (timestamp: {})\n",
        last.close, last.timestamp
    );

    // -----------------------------------------------------------------------
    // Step 2: Extract features
    // -----------------------------------------------------------------------
    let lookback = 20;
    let features = extract_features(&candles, lookback);
    println!("Extracted {} feature vectors (lookback={})", features.len(), lookback);

    if features.is_empty() {
        println!("Not enough data to extract features. Exiting.");
        return Ok(());
    }

    let n_features = features[0].values.len();
    println!("  Feature dimension: {}\n", n_features);

    // -----------------------------------------------------------------------
    // Step 3: Create ensemble model
    // -----------------------------------------------------------------------
    let n_members = 7;
    let n_neurons = 8;
    let ensemble = create_ensemble(n_members, n_neurons, n_features, 12345);
    println!(
        "Created ensemble with {} members ({} neurons each)\n",
        n_members, n_neurons
    );

    // -----------------------------------------------------------------------
    // Step 4: Generate predictions with uncertainty for last 10 data points
    // -----------------------------------------------------------------------
    println!("--- Predictions with Uncertainty (last 10 points) ---");
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>14} {:>8}",
        "Idx", "Mean", "Epist", "Aleat", "CI 95%", "Trade?"
    );
    println!("{}", "-".repeat(70));

    let start = if features.len() > 10 {
        features.len() - 10
    } else {
        0
    };

    let mut predictions_for_conformal: Vec<(f64, f64)> = Vec::new();

    for (i, feat) in features[start..].iter().enumerate() {
        let pred = ensemble_predict(&feat.values, &ensemble);
        let trade = should_trade(&pred);
        let (lo, hi) = pred.confidence_interval;

        println!(
            "{:<6} {:>10.6} {:>10.6} {:>10.6} [{:>5.4},{:>5.4}] {:>8}",
            start + i,
            pred.mean,
            pred.epistemic_uncertainty,
            pred.aleatoric_uncertainty,
            lo,
            hi,
            if trade { "YES" } else { "NO" }
        );

        // Store for conformal calibration (using lagged actual returns as ground truth)
        if start + i + 1 < features.len() {
            let actual_return = features[start + i + 1].values[0]; // 1-period return
            predictions_for_conformal.push((pred.mean, actual_return));
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Conformal prediction demonstration
    // -----------------------------------------------------------------------
    println!("\n--- Conformal Prediction Calibration ---");

    // Use first half of data for calibration errors
    let cal_size = features.len() / 2;
    let mut cal_errors: Vec<f64> = Vec::new();
    for i in 0..cal_size.saturating_sub(1) {
        let pred = ensemble_predict(&features[i].values, &ensemble);
        let actual_return = features[i + 1].values[0];
        cal_errors.push((actual_return - pred.mean).abs());
    }

    if !cal_errors.is_empty() {
        let alpha = 0.1;
        let latest_pred = ensemble_predict(&features.last().unwrap().values, &ensemble);
        let (lo, hi) = conformal_intervals(&cal_errors, latest_pred.mean, alpha);
        println!(
            "Latest prediction: {:.6}",
            latest_pred.mean
        );
        println!(
            "Conformal interval (alpha={:.2}): [{:.6}, {:.6}]",
            alpha, lo, hi
        );

        // Evaluate coverage on second half
        let test_preds: Vec<f64> = (cal_size..features.len().saturating_sub(1))
            .map(|i| ensemble_predict(&features[i].values, &ensemble).mean)
            .collect();
        let test_actuals: Vec<f64> = (cal_size + 1..features.len())
            .map(|i| features[i].values[0])
            .collect();
        let n = test_preds.len().min(test_actuals.len());
        if n > 0 {
            let coverage = conformal_coverage(
                &test_actuals[..n],
                &test_preds[..n],
                &cal_errors,
                alpha,
            );
            println!(
                "Test-set coverage (target {:.0}%): {:.1}%",
                (1.0 - alpha) * 100.0,
                coverage * 100.0
            );
        }
    }

    // -----------------------------------------------------------------------
    // Step 6: Risk-adjusted position sizing
    // -----------------------------------------------------------------------
    println!("\n--- Risk-Adjusted Position Sizing ---");
    let base_position = 10000.0; // $10,000 base
    let max_unc = 0.5;

    let latest_feat = &features.last().unwrap().values;
    let pred = ensemble_predict(latest_feat, &ensemble);
    let size = position_size(&pred, base_position, max_unc);
    let trade = should_trade(&pred);

    println!("Base position:      ${:.0}", base_position);
    println!("Predicted return:   {:.6}", pred.mean);
    println!("Total uncertainty:  {:.6}", pred.total_uncertainty);
    println!("Adjusted position:  ${:.0}", size);
    println!(
        "Action:             {}",
        if !trade {
            "SKIP (uncertain direction)"
        } else if pred.mean > 0.0 {
            "LONG"
        } else {
            "SHORT"
        }
    );

    // MC Dropout comparison
    println!("\n--- MC Dropout Comparison ---");
    let mc_weights: Vec<Vec<f64>> = ensemble
        .iter()
        .flat_map(|m| m.weights.clone())
        .collect();
    if !mc_weights.is_empty() {
        let mc_pred = mc_dropout_predict(latest_feat, &mc_weights, 0.3, 200);
        println!("MC Dropout mean:        {:.6}", mc_pred.mean);
        println!("MC Dropout uncertainty:  {:.6}", mc_pred.total_uncertainty);
        println!(
            "MC Dropout CI 95%:      [{:.6}, {:.6}]",
            mc_pred.confidence_interval.0, mc_pred.confidence_interval.1
        );
    }

    println!("\nDone.");
    Ok(())
}

/// Generate synthetic candle data when live API is unavailable.
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut candles = Vec::with_capacity(n);
    for i in 0..n {
        let ret = rng.gen_range(-0.03..0.03);
        let open = price;
        let close = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(500.0..5000.0);
        candles.push(Candle {
            timestamp: 1700000000 + (i as u64) * 86400,
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
