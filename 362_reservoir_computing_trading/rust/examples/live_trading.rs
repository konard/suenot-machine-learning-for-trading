//! Live Trading Example with Bybit
//!
//! This example demonstrates how to use the reservoir computing
//! library for live trading signal generation with Bybit data.
//!
//! Run with: cargo run --example live_trading
//!
//! Note: This is a simulation example. For real trading,
//! you would need to add proper API credentials and order execution.

use reservoir_trading::{
    BybitClient, BybitConfig, EchoStateNetwork, EsnConfig,
    FeatureExtractor, TradingConfig, TradingSystem, MarketFeatures,
};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Reservoir Computing - Live Trading Simulation");
    println!("═══════════════════════════════════════════════════════════\n");

    // Configuration
    let symbol = "BTCUSDT";
    let interval = "5"; // 5-minute candles

    // Create Bybit client (using mainnet for public data)
    println!("1. Connecting to Bybit API...");
    let config = BybitConfig::mainnet();
    let client = BybitClient::new(config);

    // Fetch initial historical data for training
    println!("\n2. Fetching historical data for training...");
    let klines = client.get_klines(symbol, interval, 500).await?;
    println!("   Fetched {} klines for {}", klines.len(), symbol);

    if klines.is_empty() {
        println!("   ERROR: No data received. Please check your connection.");
        return Ok(());
    }

    // Show latest price
    let latest = klines.last().unwrap();
    println!("   Latest close: ${:.2}", latest.close);
    println!("   Latest volume: {:.2}", latest.volume);

    // Extract features from historical data
    println!("\n3. Extracting features from historical data...");
    let mut feature_extractor = FeatureExtractor::default_params();
    let mut features_list = Vec::new();
    let mut targets_list = Vec::new();

    for i in 0..klines.len() {
        if let Some(features) = feature_extractor.update(&klines[i], None) {
            if i + 1 < klines.len() {
                let next_return = (klines[i + 1].close / klines[i].close).ln();
                features_list.push(features);
                targets_list.push(next_return);
            }
        }
    }

    println!("   Extracted {} feature vectors", features_list.len());

    if features_list.len() < 100 {
        println!("   ERROR: Not enough data for training. Need at least 100 samples.");
        return Ok(());
    }

    // Convert to arrays
    let n_samples = features_list.len();
    let n_features = MarketFeatures::n_features();
    let mut train_features = ndarray::Array2::zeros((n_samples, n_features));
    let mut train_targets = ndarray::Array2::zeros((n_samples, 1));

    for (i, (feat, target)) in features_list.iter().zip(targets_list.iter()).enumerate() {
        train_features.row_mut(i).assign(&feat.to_array());
        train_targets[[i, 0]] = *target;
    }

    // Create and train ESN
    println!("\n4. Training Echo State Network...");
    let esn_config = EsnConfig {
        reservoir_size: 300,
        spectral_radius: 0.9,
        input_scaling: 0.3,
        leaking_rate: 0.4,
        sparsity: 0.1,
        regularization: 1e-5,
        seed: 42,
    };

    let mut esn = EchoStateNetwork::new(n_features, 1, esn_config);
    let washout = 50;

    let start = std::time::Instant::now();
    let mse = esn.fit(&train_features, &train_targets, washout)?;
    let training_time = start.elapsed();

    println!("   Training completed in {:?}", training_time);
    println!("   Training MSE: {:.8}", mse);

    // Initialize trading system
    println!("\n5. Initializing trading system...");
    let trading_config = TradingConfig {
        max_position: 1.0,
        position_scale: 0.5,
        transaction_cost: 0.001,
        stop_loss: Some(0.02),
        take_profit: Some(0.04),
        max_drawdown: 0.10,
        ..TradingConfig::default()
    };

    let mut trading_system = TradingSystem::new(trading_config, 10000.0);

    // Reset feature extractor for live simulation
    let mut live_extractor = FeatureExtractor::default_params();

    // Warm up feature extractor with recent data
    for kline in klines.iter().take(klines.len().saturating_sub(10)) {
        live_extractor.update(kline, None);
    }

    // Live trading simulation loop
    println!("\n6. Starting live trading simulation...");
    println!("   (Press Ctrl+C to stop)\n");
    println!(
        "   {:>12} {:>10} {:>8} {:>10} {:>12} {:>10}",
        "Price", "Signal", "Position", "Equity", "Return", "Trades"
    );
    println!("   {:-<12} {:-<10} {:-<8} {:-<10} {:-<12} {:-<10}", "", "", "", "", "", "");

    // Simulate 20 iterations (in real trading, this would be continuous)
    let mut iteration = 0;
    let max_iterations = 20;

    while iteration < max_iterations {
        // Fetch latest data
        let latest_klines = client.get_klines(symbol, interval, 2).await?;

        if latest_klines.is_empty() {
            sleep(Duration::from_secs(5)).await;
            continue;
        }

        let latest = latest_klines.last().unwrap();

        // Update features
        if let Some(features) = live_extractor.update(latest, None) {
            let input = features.to_array();

            // Get prediction from ESN
            let prediction = esn.predict_one(&input)?;
            let signal = prediction[0].tanh();

            // Execute trading decision
            let current_price = latest.close;
            let _ = trading_system.execute(signal, current_price);

            // Display status
            let position = trading_system.position();
            let equity = trading_system.current_equity();
            let total_return = trading_system.total_return();

            println!(
                "   ${:>10.2} {:>10.4} {:>8.2} ${:>9.2} {:>11.2}% {:>10}",
                current_price,
                signal,
                position.size,
                equity,
                total_return * 100.0,
                position.n_trades
            );
        }

        iteration += 1;

        // Wait before next iteration (in real trading, sync with candle close)
        if iteration < max_iterations {
            sleep(Duration::from_secs(3)).await;
        }
    }

    // Final summary
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Trading Simulation Summary");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Initial Capital:  $10,000.00");
    println!("  Final Equity:     ${:.2}", trading_system.current_equity());
    println!("  Total Return:     {:.2}%", trading_system.total_return() * 100.0);
    println!("  Total Trades:     {}", trading_system.position().n_trades);
    println!("  Final Position:   {:.2}", trading_system.position().size);
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
