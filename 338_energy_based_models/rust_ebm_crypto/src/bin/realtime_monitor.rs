//! Real-time market monitoring using EBM
//!
//! Usage:
//!   cargo run --bin realtime_monitor -- --symbol BTCUSDT --interval 1

use clap::Parser;
use log::info;
use std::time::Duration;
use rust_ebm_crypto::data::BybitClient;
use rust_ebm_crypto::ebm::{MarketRegime, OnlineEnergyEstimator};
use rust_ebm_crypto::features::FeatureEngine;
use rust_ebm_crypto::strategy::{SignalConfig, SignalGenerator, SignalType};

#[derive(Parser, Debug)]
#[command(author, version, about = "Real-time EBM market monitor")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval (minutes)
    #[arg(short, long, default_value = "1")]
    interval: String,

    /// Polling interval in seconds
    #[arg(short, long, default_value = "10")]
    poll_seconds: u64,

    /// Anomaly threshold
    #[arg(short, long, default_value = "2.0")]
    threshold: f64,

    /// Number of iterations (0 = infinite)
    #[arg(short, long, default_value = "0")]
    max_iterations: usize,

    /// Initial warmup candles
    #[arg(long, default_value = "200")]
    warmup: usize,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("=== EBM Real-Time Market Monitor ===");
    println!("Symbol:    {}", args.symbol);
    println!("Interval:  {} min", args.interval);
    println!("Threshold: {} std", args.threshold);
    println!();

    let client = BybitClient::public();
    let engine = FeatureEngine::default();
    let mut estimator = OnlineEnergyEstimator::new(100, 0.1);
    let mut signal_gen = SignalGenerator::new(SignalConfig {
        anomaly_threshold: args.threshold,
        ..Default::default()
    });

    // Warmup phase
    info!("Warming up with {} candles...", args.warmup);
    let initial_data = client.get_klines(
        &args.symbol,
        &args.interval,
        args.warmup.min(1000) as u32,
        None,
        None,
    )?;

    let features = engine.compute(&initial_data.data);
    for i in 0..features.nrows() {
        let row = features.row(i).to_owned();
        estimator.update(&row);
    }

    info!("Warmup complete. Starting monitoring...");
    println!();

    // Print header
    println!(
        "{:<20} {:>12} {:>12} {:>10} {:>10} {:>10} {:>15}",
        "Timestamp", "Price", "Energy", "Norm.E", "Regime", "Signal", "Position"
    );
    println!("{}", "=".repeat(95));

    let mut last_timestamp = initial_data.data.last().map(|c| c.timestamp).unwrap_or(0);
    let mut iteration = 0;

    loop {
        // Check iteration limit
        if args.max_iterations > 0 && iteration >= args.max_iterations {
            println!("\nReached maximum iterations. Exiting.");
            break;
        }

        // Fetch latest candle
        let data = client.get_klines(&args.symbol, &args.interval, 10, None, None)?;

        if let Some(candle) = data.data.last() {
            // Only process if new candle
            if candle.timestamp > last_timestamp {
                last_timestamp = candle.timestamp;
                iteration += 1;

                // Compute features for latest candle
                let all_candles = &data.data;
                let features = engine.compute(all_candles);
                let latest_features = features.row(features.nrows() - 1).to_owned();

                // Update estimator
                let energy_result = estimator.update(&latest_features);

                // Calculate return
                let ret = if all_candles.len() >= 2 {
                    let prev = &all_candles[all_candles.len() - 2];
                    (candle.close - prev.close) / prev.close
                } else {
                    0.0
                };

                // Generate signal
                let signal = signal_gen.generate(
                    energy_result.normalized_energy,
                    energy_result.energy,
                    ret,
                    energy_result.regime,
                    candle.timestamp,
                );

                // Format output
                let regime_color = match energy_result.regime {
                    MarketRegime::Calm => "\x1b[32m", // Green
                    MarketRegime::Normal => "\x1b[0m", // Default
                    MarketRegime::Elevated => "\x1b[33m", // Yellow
                    MarketRegime::Crisis => "\x1b[31m", // Red
                };

                let signal_color = match signal.signal_type {
                    SignalType::Long => "\x1b[32m",
                    SignalType::Short => "\x1b[31m",
                    SignalType::Exit => "\x1b[35m",
                    SignalType::ReducePosition => "\x1b[33m",
                    SignalType::Hold => "\x1b[0m",
                };

                println!(
                    "{:<20} {:>12.4} {:>12.4} {:>10.2} {}{:>10}\x1b[0m {}{:>10}\x1b[0m {:>15.1}%",
                    candle.datetime().format("%Y-%m-%d %H:%M"),
                    candle.close,
                    energy_result.energy,
                    energy_result.normalized_energy,
                    regime_color,
                    energy_result.regime.as_str(),
                    signal_color,
                    signal.signal_type.as_str(),
                    signal.position_scale * 100.0
                );

                // Alert on anomaly
                if energy_result.is_anomaly {
                    println!(
                        "\x1b[31m!!! ANOMALY DETECTED: {} - Energy: {:.2} !!!\x1b[0m",
                        signal.reason, energy_result.normalized_energy
                    );
                }
            }
        }

        // Wait before next poll
        std::thread::sleep(Duration::from_secs(args.poll_seconds));
    }

    Ok(())
}
