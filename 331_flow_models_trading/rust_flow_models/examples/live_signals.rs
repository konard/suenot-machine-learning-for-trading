//! Example: Generate live trading signals
//!
//! This example demonstrates how to use the flow-based trading strategy
//! to generate signals from live market data fetched from Bybit.

use flow_models_trading::prelude::*;
use ndarray::{Array1, Array2};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("        Flow Models Trading - Live Signal Generation");
    println!("═══════════════════════════════════════════════════════════════");

    // Fetch historical data for model training
    println!("\n1. Fetching historical data from Bybit...");
    let client = BybitClient::new();

    let symbol = "BTCUSDT";
    let klines = client.get_klines(symbol, "60", 500).await?;
    println!("   Fetched {} candles for {}", klines.len(), symbol);

    // Compute features
    println!("\n2. Computing features...");
    let mut feature_engine = FeatureEngine::new();
    let features = feature_engine.compute_minimal_features(&klines);
    let features = FeatureEngine::clean_features(&features);
    println!("   Computed {} features for {} samples",
             feature_engine.feature_dim(), features.nrows());
    println!("   Feature names: {:?}", feature_engine.get_feature_names());

    // Compute returns for regime labeling
    let returns: Array1<f64> = klines.iter()
        .skip(1)
        .zip(klines.iter())
        .map(|(curr, prev)| (curr.close - prev.close) / prev.close)
        .collect();

    // Pad returns to match features length
    let returns = if returns.len() < features.nrows() {
        let mut padded = Array1::zeros(features.nrows());
        for i in 0..(returns.len().min(features.nrows())) {
            padded[i] = returns[i];
        }
        padded
    } else {
        returns.slice(ndarray::s![0..features.nrows()]).to_owned()
    };

    // Create flow model
    println!("\n3. Creating and training flow model...");
    let config = FlowConfig::default()
        .with_input_dim(feature_engine.feature_dim())
        .with_num_layers(4)
        .with_hidden_dim(64);

    let mut model = NormalizingFlow::new(config);

    // Train on historical data
    let train_size = (features.nrows() as f64 * 0.8) as usize;
    let train_features = features.slice(ndarray::s![0..train_size, ..]).to_owned();
    let train_returns = returns.slice(ndarray::s![0..train_size]).to_owned();

    let _ = model.train(&train_features, 30);
    println!("   Model trained on {} samples", train_features.nrows());

    // Create trading strategy
    println!("\n4. Creating trading strategy...");
    let mut strategy = FlowTradingStrategy::new(model)
        .with_confidence_threshold(0.5)
        .with_n_regimes(4)
        .with_anomaly_percentile(5.0);

    strategy.fit(&train_features, Some(&train_returns));

    // Fetch current data
    println!("\n5. Fetching current market data...");
    let current_ticker = client.get_tickers_filtered(&[symbol]).await?
        .into_iter()
        .next()
        .ok_or("Ticker not found")?;

    let current_orderbook = client.get_orderbook(symbol, 10).await?;

    println!("\n   Current Market State:");
    println!("   ─────────────────────────────────────────────────────");
    println!("   Symbol: {}", symbol);
    println!("   Price: ${:.2}", current_ticker.last_price);
    println!("   24h Change: {:.2}%", current_ticker.price_change_24h * 100.0);
    println!("   24h Volume: {:.2} BTC", current_ticker.volume_24h);
    println!("   Spread: {:.2} bps", current_orderbook.spread_bps().unwrap_or(0.0));
    println!("   Depth Imbalance: {:.4}", current_orderbook.depth_imbalance(5));

    // Generate signal for latest data point
    println!("\n6. Generating trading signal...");

    // Get most recent feature vector
    let latest_features = features.row(features.nrows() - 1).to_owned();
    let signal = strategy.generate_signal(&latest_features);

    println!("\n   ═══════════════════════════════════════════════════════");
    println!("                     TRADING SIGNAL");
    println!("   ═══════════════════════════════════════════════════════");
    println!("   Signal Type:    {:?}", signal.signal_type);
    println!("   Confidence:     {:.2}%", signal.confidence * 100.0);
    println!("   Regime:         {}", signal.regime.as_deref().unwrap_or("Unknown"));
    println!("   Log-Likelihood: {:.4}", signal.log_likelihood.unwrap_or(0.0));
    println!("   Reason:         {}", signal.reason);
    println!("   ═══════════════════════════════════════════════════════");

    // Provide trading recommendation
    println!("\n7. Trading Recommendation:");
    match signal.signal_type {
        SignalType::Long => {
            println!("   [BUY] Consider opening a LONG position on {}", symbol);
            println!("   - Entry: around ${:.2}", current_ticker.last_price);
            println!("   - Stop loss: consider ${:.2} (-2%)",
                     current_ticker.last_price * 0.98);
            println!("   - Take profit: consider ${:.2} (+4%)",
                     current_ticker.last_price * 1.04);
        }
        SignalType::Short => {
            println!("   [SELL] Consider opening a SHORT position on {}", symbol);
            println!("   - Entry: around ${:.2}", current_ticker.last_price);
            println!("   - Stop loss: consider ${:.2} (+2%)",
                     current_ticker.last_price * 1.02);
            println!("   - Take profit: consider ${:.2} (-4%)",
                     current_ticker.last_price * 0.96);
        }
        SignalType::Neutral => {
            println!("   [WAIT] No clear signal - stay on the sidelines");
            println!("   - Market conditions are uncertain");
            println!("   - Wait for higher confidence signal");
        }
        SignalType::ReduceExposure => {
            println!("   [CAUTION] Unusual market conditions detected!");
            println!("   - Close or reduce existing positions");
            println!("   - Wait for market to normalize");
            println!("   - Log-likelihood indicates anomalous state");
        }
    }

    // Recent signal history
    println!("\n8. Recent signal history (last 10 candles):");
    println!("   {:>5} {:>12} {:>12} {:>10}",
             "Index", "Signal", "Confidence", "Regime");
    println!("   {:-<45}", "");

    let start_idx = features.nrows().saturating_sub(10);
    for i in start_idx..features.nrows() {
        let feat = features.row(i).to_owned();
        let sig = strategy.generate_signal(&feat);
        let regime = sig.regime.as_deref().unwrap_or("Unknown");

        println!("   {:>5} {:>12?} {:>10.2}% {:>10}",
                 i - start_idx,
                 sig.signal_type,
                 sig.confidence * 100.0,
                 if regime.len() > 10 { &regime[..10] } else { regime });
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                  Signal generation complete!");
    println!("═══════════════════════════════════════════════════════════════");

    // Disclaimer
    println!("\n[DISCLAIMER] This is for educational purposes only.");
    println!("Cryptocurrency trading involves substantial risk of loss.");
    println!("Do not trade with money you cannot afford to lose.");

    Ok(())
}
