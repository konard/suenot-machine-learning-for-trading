//! Trading example: Fetch BTCUSDT orderbook from Bybit, compute liquidity
//! metrics, forecast next-period liquidity, and show optimal order sizing.

use anyhow::Result;
use liquidity_prediction::*;
use ndarray::Array1;

fn main() -> Result<()> {
    println!("=== Liquidity Prediction for Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch BTCUSDT orderbook from Bybit
    // -----------------------------------------------------------------------
    println!("--- Step 1: Fetching BTCUSDT orderbook from Bybit ---");

    let snapshot = match BybitClient::new().fetch_orderbook("BTCUSDT", 50) {
        Ok(snap) => {
            println!(
                "Fetched orderbook: {} bids, {} asks",
                snap.bids.len(),
                snap.asks.len()
            );
            println!("Best bid: {:.2}, Best ask: {:.2}", snap.best_bid(), snap.best_ask());
            println!("Mid price: {:.2}", snap.mid_price());
            snap
        }
        Err(e) => {
            println!("Could not fetch live data ({}), using synthetic orderbook", e);
            create_synthetic_snapshot()
        }
    };

    // -----------------------------------------------------------------------
    // Step 2: Compute liquidity metrics
    // -----------------------------------------------------------------------
    println!("\n--- Step 2: Computing Liquidity Metrics ---");

    let metrics = LiquidityMetrics::compute(&snapshot, 25);
    println!("Spread:           {:.4}", metrics.spread);
    println!("Relative spread:  {:.6} ({:.2} bps)", metrics.relative_spread, metrics.relative_spread * 10000.0);
    println!("Bid depth (5 levels): {:.4}", metrics.bid_depth.get(4).copied().unwrap_or(0.0));
    println!("Ask depth (5 levels): {:.4}", metrics.ask_depth.get(4).copied().unwrap_or(0.0));
    println!("Total depth (5 levels): {:.4}", metrics.total_depth.get(4).copied().unwrap_or(0.0));
    println!("Total depth (10 levels): {:.4}", metrics.total_depth.get(9).copied().unwrap_or(0.0));
    println!("Order flow imbalance: {:.4}", metrics.ofi);
    println!("Liquidity score:  {:.4}", metrics.liquidity_score);

    // Resilience proxy example
    let resilience = resilience_proxy(100.0, 92.0);
    println!("Resilience proxy (example): {:.4}", resilience);

    // Slippage estimate
    let slippage = estimate_slippage(
        5.0,
        metrics.total_depth.get(4).copied().unwrap_or(100.0),
        metrics.spread / 5.0,
    );
    println!("Estimated slippage for 5 BTC order: {:.4}", slippage);

    // -----------------------------------------------------------------------
    // Step 3: LSTM Liquidity Forecast
    // -----------------------------------------------------------------------
    println!("\n--- Step 3: LSTM Liquidity Forecasting ---");

    let mut rng = rand::thread_rng();
    let input_size = 6;
    let hidden_size = 16;
    let mut forecaster = LstmForecaster::new(input_size, hidden_size, &mut rng);

    // Generate synthetic historical sequence (in production, from collected snapshots)
    let feature_vec = metrics.to_feature_vector();
    let sequence: Vec<Array1<f64>> = (0..10)
        .map(|i| {
            let noise = 1.0 + (i as f64 * 0.02);
            feature_vec.mapv(|v| v * noise)
        })
        .collect();

    // Train for a few steps on synthetic data
    let target_depth = metrics.total_depth.get(4).copied().unwrap_or(100.0);
    println!("Training LSTM on synthetic data...");
    for epoch in 0..50 {
        let loss = forecaster.train_step(&sequence, target_depth, 0.0001);
        if epoch % 10 == 0 {
            println!("  Epoch {}: loss = {:.4}", epoch, loss);
        }
    }

    let predicted_depth = forecaster.predict(&sequence);
    println!("Predicted next-period depth (5 levels): {:.4}", predicted_depth);
    println!("Actual current depth (5 levels):        {:.4}", target_depth);

    // -----------------------------------------------------------------------
    // Step 4: Liquidity Regime Classification
    // -----------------------------------------------------------------------
    println!("\n--- Step 4: Liquidity Regime Classification ---");

    // Create synthetic training data for regime classifier
    let n_samples = 100;
    let train_features: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| {
            let scale = i as f64 / n_samples as f64;
            vec![
                0.5 + scale * 2.0,         // spread bps
                10.0 + scale * 200.0,       // depth_1
                50.0 + scale * 500.0,       // depth_5
                100.0 + scale * 1000.0,     // depth_10
                -0.5 + scale,               // ofi
                20.0 + scale * 300.0,       // liquidity_score
            ]
        })
        .collect();
    let train_depths: Vec<f64> = (0..n_samples)
        .map(|i| 50.0 + (i as f64 / n_samples as f64) * 500.0)
        .collect();

    let classifier = LiquidityRegimeClassifier::fit(&train_features, &train_depths, 20);

    let current_features = metrics.to_feature_vector();
    let current_features_vec: Vec<f64> = current_features.to_vec();
    let regime = classifier.predict(&current_features_vec);
    let regime_score = classifier.predict_score(&current_features_vec);

    println!("Current regime: {:?} (score: {:.4})", regime, regime_score);
    println!("Regime thresholds: low < {:.2}, high > {:.2}",
        classifier.low_threshold, classifier.high_threshold);

    // -----------------------------------------------------------------------
    // Step 5: Optimal Order Sizing
    // -----------------------------------------------------------------------
    println!("\n--- Step 5: Optimal Order Sizing ---");

    let sizer = OptimalOrderSizer::new(0.10); // 10% participation rate
    let parent_order_qty = 10.0; // BTC

    println!("Parent order: {:.2} BTC", parent_order_qty);
    println!("Participation rate: {:.0}%", sizer.participation_rate * 100.0);
    println!("Predicted depth: {:.4}", predicted_depth);

    let base_size = sizer.optimal_size(parent_order_qty, predicted_depth);
    let regime_size = sizer.regime_adjusted_size(parent_order_qty, predicted_depth, regime);

    println!("\nBase optimal child order size: {:.4} BTC", base_size);
    println!("Regime-adjusted size ({:?}): {:.4} BTC", regime, regime_size);

    // Show sizing across all regimes
    println!("\nSizing by regime:");
    for r in [LiquidityRegime::High, LiquidityRegime::Medium, LiquidityRegime::Low] {
        let s = sizer.regime_adjusted_size(parent_order_qty, predicted_depth, r);
        println!("  {:?}: {:.4} BTC", r, s);
    }

    // Expected slippage comparison
    let slip_base = estimate_slippage(base_size, predicted_depth, metrics.spread / 5.0);
    let slip_regime = estimate_slippage(regime_size, predicted_depth, metrics.spread / 5.0);
    println!("\nExpected slippage (base):   {:.6}", slip_base);
    println!("Expected slippage (regime): {:.6}", slip_regime);

    println!("\n=== Done ===");
    Ok(())
}

/// Create a synthetic orderbook for demo purposes when API is unavailable.
fn create_synthetic_snapshot() -> OrderbookSnapshot {
    let mid_price = 67000.0;
    let tick = 0.5;

    let bids: Vec<PriceLevel> = (0..25)
        .map(|i| PriceLevel {
            price: mid_price - tick * (i as f64 + 1.0),
            quantity: 0.5 + (i as f64) * 0.3,
        })
        .collect();

    let asks: Vec<PriceLevel> = (0..25)
        .map(|i| PriceLevel {
            price: mid_price + tick * (i as f64 + 1.0),
            quantity: 0.4 + (i as f64) * 0.25,
        })
        .collect();

    OrderbookSnapshot {
        bids,
        asks,
        timestamp: 1700000000000,
    }
}
