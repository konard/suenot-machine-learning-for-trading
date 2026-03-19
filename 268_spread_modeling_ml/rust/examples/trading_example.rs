use spread_modeling_ml::*;

fn main() {
    println!("=== Chapter 268: Spread Modeling with ML ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch BTCUSDT orderbook from Bybit
    // -----------------------------------------------------------------------
    println!("--- Step 1: Fetch Bybit BTCUSDT Orderbook ---");
    match fetch_bybit_orderbook("BTCUSDT", 25) {
        Ok(ob) => {
            println!("Symbol: {}", ob.symbol);
            if let (Some(bid), Some(ask)) = (ob.best_bid(), ob.best_ask()) {
                println!("Best Bid: {:.2}", bid);
                println!("Best Ask: {:.2}", ask);
                println!("Quoted Spread: {:.2}", ob.spread().unwrap_or(0.0));
                println!(
                    "Relative Spread: {:.6} ({:.4} bps)",
                    ob.relative_spread().unwrap_or(0.0),
                    ob.relative_spread().unwrap_or(0.0) * 10_000.0
                );
                println!("Midpoint: {:.2}", ob.midpoint().unwrap_or(0.0));
                println!("Bid Depth: {:.4} BTC", ob.bid_depth());
                println!("Ask Depth: {:.4} BTC", ob.ask_depth());
                println!("Order Imbalance: {:.4}", ob.order_imbalance());
            }
        }
        Err(e) => {
            println!("Could not fetch orderbook ({}). Using simulated data.", e);
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Compute spread time series from simulated data
    // -----------------------------------------------------------------------
    println!("\n--- Step 2: Spread Time Series (Simulated) ---");
    // Simulate bid-ask prices with realistic microstructure noise
    let n_obs = 200;
    let base_price = 50000.0;
    let mut bids = Vec::with_capacity(n_obs);
    let mut asks = Vec::with_capacity(n_obs);
    let mut spreads = Vec::with_capacity(n_obs);
    let mut midpoints = Vec::with_capacity(n_obs);

    // Simple LCG for reproducible randomness
    let mut seed: u64 = 42;
    let lcg_next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / (u32::MAX as f64)
    };

    for i in 0..n_obs {
        let trend = (i as f64 * 0.01).sin() * 100.0;
        let noise = (lcg_next(&mut seed) - 0.5) * 50.0;
        let mid = base_price + trend + noise;

        // Spread varies with a base + volatility-dependent component
        let vol_component = noise.abs() * 0.02;
        let half_spread = 5.0 + vol_component;

        let bid = mid - half_spread;
        let ask = mid + half_spread;

        bids.push(bid);
        asks.push(ask);
        spreads.push(absolute_spread(bid, ask));
        midpoints.push(midpoint(bid, ask));
    }

    let mean_spread: f64 = spreads.iter().sum::<f64>() / spreads.len() as f64;
    let min_spread = spreads.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_spread = spreads.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Observations: {}", n_obs);
    println!("Mean Spread: {:.4}", mean_spread);
    println!("Min Spread:  {:.4}", min_spread);
    println!("Max Spread:  {:.4}", max_spread);

    // -----------------------------------------------------------------------
    // Step 3: Roll Spread Estimator
    // -----------------------------------------------------------------------
    println!("\n--- Step 3: Roll Spread Estimator ---");
    let roll = roll_spread_estimator(&midpoints);
    println!("Roll Spread Estimate: {:.4}", roll);
    println!("Quoted Spread (mean): {:.4}", mean_spread);
    println!(
        "Roll / Quoted ratio:  {:.4}",
        if mean_spread > 0.0 {
            roll / mean_spread
        } else {
            0.0
        }
    );

    // -----------------------------------------------------------------------
    // Step 4: Linear Regression for Spread Prediction
    // -----------------------------------------------------------------------
    println!("\n--- Step 4: Linear Regression Spread Prediction ---");

    // Build features: volatility (rolling |return|), volume proxy, depth proxy, imbalance proxy
    let window = 10;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for i in window..n_obs {
        // Rolling volatility (mean absolute return over window)
        let vol: f64 = (1..window)
            .map(|j| ((midpoints[i - j] - midpoints[i - j - 1]) / midpoints[i - j - 1]).abs())
            .sum::<f64>()
            / (window - 1) as f64;

        // Volume proxy (simulated): random-ish based on index
        let vol_proxy = 1000.0 + (i as f64 * 0.1).cos() * 500.0;

        // Depth proxy
        let depth = 10.0 + (i as f64 * 0.05).sin() * 5.0;

        // Imbalance proxy
        let imbalance = (i as f64 * 0.07).sin() * 0.3;

        features.push(vec![vol, vol_proxy, depth, imbalance]);
        targets.push(spreads[i]);
    }

    // Split into train/test (80/20)
    let split = (features.len() as f64 * 0.8) as usize;
    let train_features = &features[..split];
    let train_targets = &targets[..split];
    let test_features = &features[split..];
    let test_targets = &targets[split..];

    let mut model = LinearRegression::new();
    match model.fit(&train_features.to_vec(), train_targets) {
        Ok(()) => {
            let coeffs = model.coefficients.as_ref().unwrap();
            println!("Fitted coefficients:");
            println!("  Intercept:  {:.6}", coeffs[0]);
            println!("  Volatility: {:.6}", coeffs[1]);
            println!("  Volume:     {:.6}", coeffs[2]);
            println!("  Depth:      {:.6}", coeffs[3]);
            println!("  Imbalance:  {:.6}", coeffs[4]);

            // Evaluate on test set
            let predictions = model.predict(&test_features.to_vec()).unwrap();
            let mse: f64 = predictions
                .iter()
                .zip(test_targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / test_targets.len() as f64;
            let rmse = mse.sqrt();

            let mean_test: f64 = test_targets.iter().sum::<f64>() / test_targets.len() as f64;
            let ss_tot: f64 = test_targets.iter().map(|t| (t - mean_test).powi(2)).sum();
            let ss_res: f64 = predictions
                .iter()
                .zip(test_targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum();
            let r_squared = 1.0 - ss_res / ss_tot;

            println!("\nTest set performance:");
            println!("  RMSE:      {:.6}", rmse);
            println!("  R-squared: {:.6}", r_squared);

            // Show a few predictions vs actual
            println!("\nSample predictions (first 5 test observations):");
            println!("  {:>12} {:>12}", "Predicted", "Actual");
            for i in 0..5.min(predictions.len()) {
                println!("  {:>12.4} {:>12.4}", predictions[i], test_targets[i]);
            }
        }
        Err(e) => println!("Regression failed: {}", e),
    }

    // -----------------------------------------------------------------------
    // Step 5: Spread Regime Classification
    // -----------------------------------------------------------------------
    println!("\n--- Step 5: Spread Regime Classification ---");
    let classifier = SpreadRegimeClassifier::from_median(&spreads);
    println!("Threshold (median spread): {:.4}", classifier.threshold);

    let regimes = classifier.classify_series(&spreads);
    let tight_count = regimes.iter().filter(|&&r| r == SpreadRegime::Tight).count();
    let wide_count = regimes.iter().filter(|&&r| r == SpreadRegime::Wide).count();
    println!("Tight regime: {} observations ({:.1}%)", tight_count, tight_count as f64 / n_obs as f64 * 100.0);
    println!("Wide regime:  {} observations ({:.1}%)", wide_count, wide_count as f64 / n_obs as f64 * 100.0);

    let (p_tw, p_wt) = classifier.transition_probabilities(&spreads);
    println!("P(Tight -> Wide): {:.4}", p_tw);
    println!("P(Wide -> Tight): {:.4}", p_wt);

    // -----------------------------------------------------------------------
    // Step 6: Transaction Cost Estimates
    // -----------------------------------------------------------------------
    println!("\n--- Step 6: Transaction Cost Estimates ---");
    let estimator = TransactionCostEstimator::new(0.1);

    let scenarios = vec![
        ("Small order (0.1 BTC)", 0.1, 500.0),
        ("Medium order (1.0 BTC)", 1.0, 500.0),
        ("Large order (10 BTC)", 10.0, 500.0),
    ];

    let current_spread = mean_spread;
    let current_mid = base_price;
    let current_vol = 0.001; // 10 bps volatility

    println!(
        "Spread: {:.2}, Midpoint: {:.2}, Volatility: {:.4}",
        current_spread, current_mid, current_vol
    );
    println!();
    println!(
        "  {:30} {:>12} {:>12} {:>12}",
        "Scenario", "Half-Spread", "Impact", "Total (bps)"
    );
    println!("  {}", "-".repeat(66));

    for (name, size, avg_vol) in &scenarios {
        let hs = estimator.half_spread_cost(current_spread);
        let impact = estimator.market_impact(current_vol, *size, *avg_vol);
        let total_bps = estimator.total_cost_bps(
            current_spread,
            current_mid,
            current_vol,
            *size,
            *avg_vol,
        );
        println!(
            "  {:30} {:>12.4} {:>12.6} {:>12.4}",
            name, hs, impact, total_bps
        );
    }

    println!("\n=== Analysis Complete ===");
}
