//! Bybit Trading Example
//!
//! This example demonstrates:
//! - Fetching real market data from Bybit
//! - Creating and using a CNF trader
//! - Generating trading signals

use cnf_trading::{
    api::BybitClient,
    cnf::ContinuousNormalizingFlow,
    trading::CNFTrader,
    utils::{compute_features_batch, normalize_features},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== CNF Trading on Bybit Example ===\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch market data
    println!("Fetching BTCUSDT candle data from Bybit...");
    let candles = client.get_klines("BTCUSDT", "60", 500).await?;
    println!("Fetched {} candles", candles.len());

    if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
        println!("Time range: {} to {}", first.timestamp, last.timestamp);
        println!("Price range: {:.2} to {:.2}\n", first.close, last.close);
    }

    // Compute features
    println!("Computing market features...");
    let lookback = 20;
    let features = compute_features_batch(&candles, lookback);
    println!("Feature matrix shape: {:?}\n", features.shape());

    // Normalize features
    println!("Normalizing features...");
    let (normalized, means, stds) = normalize_features(&features);

    // Create CNF model
    println!("Creating CNF model...");
    let cnf = ContinuousNormalizingFlow::new(9, 64, 3);

    // Create trader
    let mut trader = CNFTrader::new(cnf)
        .with_normalization(means.clone(), stds.clone())
        .with_likelihood_threshold(-15.0)
        .with_confidence_threshold(0.5);

    println!("Trader initialized\n");

    // Generate signals for recent candles
    println!("Generating trading signals for recent data...");
    println!("{:<20} {:>10} {:>10} {:>12} {:>10}",
             "Time", "Price", "Signal", "Confidence", "LogLik");
    println!("{}", "-".repeat(70));

    let start_idx = candles.len() - 20;
    for i in start_idx..candles.len() {
        let window = &candles[i - lookback..i];
        let signal = trader.generate_signal(window);

        let signal_str = match signal.signal {
            cnf_trading::trading::SignalType::Long => "LONG",
            cnf_trading::trading::SignalType::Short => "SHORT",
            cnf_trading::trading::SignalType::Neutral => "NEUTRAL",
        };

        println!("{:<20} {:>10.2} {:>10} {:>12.4} {:>10.4}",
                 candles[i].timestamp.format("%Y-%m-%d %H:%M"),
                 candles[i].close,
                 signal_str,
                 signal.confidence,
                 signal.log_likelihood);
    }

    println!();

    // Show current trading recommendation
    let latest_window = &candles[candles.len() - lookback..];
    let current_signal = trader.generate_signal(latest_window);

    println!("=== Current Trading Recommendation ===");
    println!("Signal: {:?}", current_signal.signal);
    println!("Confidence: {:.4}", current_signal.confidence);
    println!("Position Size: {:.4}", current_signal.position_size());
    println!("Expected Return: {:.6}", current_signal.expected_return);
    println!("Return Std: {:.6}", current_signal.return_std);
    println!("Log-Likelihood: {:.4}", current_signal.log_likelihood);
    println!("Regime Change: {}", current_signal.regime_change);
    println!("Actionable: {}", current_signal.is_actionable());

    // Also fetch some other symbols
    println!("\n=== Scanning Other Symbols ===");
    let symbols = ["ETHUSDT", "SOLUSDT", "XRPUSDT"];

    for symbol in symbols {
        match client.get_klines(symbol, "60", 100).await {
            Ok(sym_candles) => {
                if sym_candles.len() >= lookback {
                    let window = &sym_candles[sym_candles.len() - lookback..];
                    let signal = trader.generate_signal(window);

                    let signal_str = match signal.signal {
                        cnf_trading::trading::SignalType::Long => "LONG",
                        cnf_trading::trading::SignalType::Short => "SHORT",
                        cnf_trading::trading::SignalType::Neutral => "NEUTRAL",
                    };

                    println!("{:<10} Price: {:>10.4} Signal: {:>8} Conf: {:.4}",
                             symbol,
                             sym_candles.last().unwrap().close,
                             signal_str,
                             signal.confidence);
                }
            }
            Err(e) => {
                println!("{}: Error - {}", symbol, e);
            }
        }
    }

    println!("\n=== Example Complete ===");

    Ok(())
}
