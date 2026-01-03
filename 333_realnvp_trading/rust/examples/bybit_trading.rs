//! Bybit Trading Example
//!
//! This example demonstrates using RealNVP for cryptocurrency trading
//! with live data from Bybit exchange.

use anyhow::Result;
use ndarray::Array2;

use realnvp_trading::{
    api::BybitClient,
    flow::RealNVP,
    trading::{RealNVPTrader, SignalType},
    utils::{Candle, compute_market_state, normalize_features},
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== RealNVP Bybit Trading Example ===\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch historical data
    println!("Fetching BTCUSDT data from Bybit...");
    let candles = client.get_klines("BTCUSDT", "60", 1000).await?;
    println!("Fetched {} candles", candles.len());

    if candles.is_empty() {
        println!("No data available. Please check your internet connection.");
        return Ok(());
    }

    // Show sample data
    println!("\nLatest candles:");
    for candle in candles.iter().rev().take(5) {
        println!(
            "  {} | O: {:.2} H: {:.2} L: {:.2} C: {:.2} V: {:.0}",
            candle.timestamp.format("%Y-%m-%d %H:%M"),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        );
    }

    // Prepare features
    println!("\nComputing market features...");
    let lookback = 20;
    let mut features_list = Vec::new();

    for i in lookback..candles.len() {
        let window = &candles[i - lookback..i];
        let state = compute_market_state(window);
        features_list.push(state.to_vec());
    }

    let feature_dim = features_list[0].len();
    let n_samples = features_list.len();

    let features_flat: Vec<f64> = features_list.into_iter().flatten().collect();
    let features = Array2::from_shape_vec((n_samples, feature_dim), features_flat)?;

    println!("Feature matrix shape: {:?}", features.dim());
    println!("Feature dimension: {}", feature_dim);

    // Normalize features
    let (normalized, mean, std) = normalize_features(&features);
    println!("Features normalized");

    // Create and setup RealNVP model
    println!("\nCreating RealNVP model...");
    let flow = RealNVP::new(feature_dim, 64, 8);
    let mut trader = RealNVPTrader::new(flow);
    trader.fit_normalizer(&features);

    // Initialize model with data (for ActNorm if used)
    println!("Initializing model with training data...");

    // Compute log probabilities for all points
    let log_probs: Vec<f64> = normalized.rows()
        .into_iter()
        .map(|row| trader.model().log_prob(&row.to_owned()))
        .collect();

    let mean_log_prob = log_probs.iter().sum::<f64>() / log_probs.len() as f64;
    let min_log_prob = log_probs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\nLog probability statistics:");
    println!("  Mean: {:.4}", mean_log_prob);
    println!("  Min: {:.4}", min_log_prob);
    println!("  Max: {:.4}", max_log_prob);

    // Set threshold based on distribution
    let threshold = mean_log_prob - 2.0 * (max_log_prob - min_log_prob).abs();
    trader.set_threshold(threshold);
    println!("  Threshold: {:.4}", threshold);

    // Generate current trading signal
    println!("\n--- Current Trading Signal ---");
    let latest_candles = &candles[candles.len() - lookback..];
    let signal = trader.generate_signal(latest_candles);

    println!("Signal Type: {:?}", signal.signal_type);
    println!("Signal Strength: {:.4}", signal.strength);
    println!("Confidence: {:.4}", signal.confidence);
    println!("Log Probability: {:.4}", signal.log_prob);
    println!("In Distribution: {}", signal.in_distribution);
    println!("Position Size: {:.4}", signal.position_size());
    println!("Should Trade: {}", signal.should_trade());

    // Trading recommendation
    println!("\n--- Trading Recommendation ---");
    match signal.signal_type {
        SignalType::Long if signal.should_trade() => {
            println!("LONG position recommended");
            println!("  Size: {:.2}% of capital", signal.position_size().abs() * 100.0);
        }
        SignalType::Short if signal.should_trade() => {
            println!("SHORT position recommended");
            println!("  Size: {:.2}% of capital", signal.position_size().abs() * 100.0);
        }
        _ => {
            if !signal.in_distribution {
                println!("NO TRADE - Unusual market conditions detected");
                println!("  Current market state has low probability");
                println!("  Recommend waiting for normal conditions");
            } else {
                println!("NO TRADE - Signal not strong enough");
                println!("  Confidence: {:.2}%", signal.confidence * 100.0);
            }
        }
    }

    // Risk analysis
    println!("\n--- Risk Analysis ---");
    let risk_metrics = trader.compute_risk_metrics(1000);
    println!("Value at Risk (5%): {:.4}%", risk_metrics.var_5 * 100.0);
    println!("Value at Risk (1%): {:.4}%", risk_metrics.var_1 * 100.0);
    println!("Expected Shortfall (CVaR 5%): {:.4}%", risk_metrics.cvar_5 * 100.0);
    println!("Maximum Expected Loss: {:.4}%", risk_metrics.max_loss * 100.0);
    println!("Probability of Loss: {:.2}%", risk_metrics.prob_loss * 100.0);

    // Scenario analysis
    println!("\n--- Scenario Analysis ---");
    let scenarios = trader.generate_scenarios(1000);
    let scenario_returns: Vec<f64> = scenarios.column(0).iter().cloned().collect();

    let mean_return = scenario_returns.iter().sum::<f64>() / scenario_returns.len() as f64;
    let mut sorted_returns = scenario_returns.clone();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p10 = sorted_returns[(0.1 * sorted_returns.len() as f64) as usize];
    let p50 = sorted_returns[(0.5 * sorted_returns.len() as f64) as usize];
    let p90 = sorted_returns[(0.9 * sorted_returns.len() as f64) as usize];

    println!("Expected return (mean): {:.4}%", mean_return * 100.0);
    println!("10th percentile: {:.4}%", p10 * 100.0);
    println!("Median (50th): {:.4}%", p50 * 100.0);
    println!("90th percentile: {:.4}%", p90 * 100.0);

    // Get additional symbols
    println!("\n--- Other Available Symbols ---");
    match client.get_symbols().await {
        Ok(symbols) => {
            let sample_symbols: Vec<_> = symbols.iter().take(10).collect();
            println!("Sample symbols: {:?}", sample_symbols);
            println!("Total available: {}", symbols.len());
        }
        Err(e) => {
            println!("Could not fetch symbols: {}", e);
        }
    }

    println!("\n=== Example Complete ===");

    Ok(())
}
