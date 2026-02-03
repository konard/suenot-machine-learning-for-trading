//! Example: Live trading signal generation
//!
//! This example demonstrates a complete pipeline for generating
//! trading signals in real-time using prototype learning.

use prototype_learning::{
    BybitClient, DataProcessor, FeatureEngineering, Interval, PrototypeNetwork,
    TradingMetrics, Trade,
};
use ndarray::Array1;
use std::thread;
use std::time::Duration;

/// Trading configuration
struct TradingConfig {
    symbol: String,
    interval: Interval,
    confidence_threshold: f64,
    position_size: f64,
    check_interval_secs: u64,
    max_iterations: usize,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            interval: Interval::Hour1,
            confidence_threshold: 0.3,
            position_size: 0.1,          // 10% of capital
            check_interval_secs: 60,      // Check every minute (for demo)
            max_iterations: 5,            // Run 5 iterations for demo
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("Prototype Learning - Live Trading Example");
    println!("=========================================\n");

    let config = TradingConfig::default();

    println!("Configuration:");
    println!("  Symbol: {}", config.symbol);
    println!("  Interval: {:?}", config.interval);
    println!("  Confidence threshold: {:.0}%", config.confidence_threshold * 100.0);
    println!("  Position size: {:.0}%", config.position_size * 100.0);
    println!("  Check interval: {}s", config.check_interval_secs);
    println!("  Max iterations: {}", config.max_iterations);
    println!();

    // Initialize components
    let client = BybitClient::new();
    let fe = FeatureEngineering::new(60);
    let n_features = fe.get_feature_names().len();
    let mut processor = DataProcessor::new(60);

    // Create prototype network (in production, load pre-trained weights)
    let model = PrototypeNetwork::new(n_features, 128, 5, 2);

    // Initialize trading metrics
    let mut metrics = TradingMetrics::new(100000.0);

    // Simulated position tracking
    let mut current_position = 0; // 1 = long, -1 = short, 0 = flat
    let mut entry_price = 0.0;
    let mut entry_time = 0i64;

    println!("Starting trading loop...\n");
    println!(
        "{:<20} {:>12} {:>15} {:>12} {:>10} {:>12}",
        "Time", "Price", "State", "Confidence", "Signal", "Position"
    );
    println!("{}", "=".repeat(85));

    for iteration in 0..config.max_iterations {
        // Fetch latest data
        let klines = client.get_klines(&config.symbol, config.interval, Some(200), None, None)?;

        if klines.is_empty() {
            println!("No data received, skipping...");
            thread::sleep(Duration::from_secs(config.check_interval_secs));
            continue;
        }

        let current_price = klines.last().unwrap().close;
        let current_time = klines.last().unwrap().timestamp;

        // Compute features
        let features = fe.compute_features(&klines)?;
        let (n_samples, n_feat) = features.dim();

        // Prepare input
        let lookback = fe.lookback();
        let start_idx = n_samples.saturating_sub(lookback);

        let mut input_vec = Vec::with_capacity(n_feat * lookback);
        for t in start_idx..n_samples {
            for f in 0..n_feat {
                let val = features[[t, f]];
                input_vec.push(if val.is_nan() { 0.0 } else { val });
            }
        }

        // Pad if needed
        while input_vec.len() < n_feat * lookback {
            input_vec.push(0.0);
        }

        // Normalize (simplified - in production use proper normalization)
        let mean: f64 = input_vec.iter().sum::<f64>() / input_vec.len() as f64;
        let std: f64 = {
            let var: f64 = input_vec.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / input_vec.len() as f64;
            var.sqrt()
        };
        let std = if std < 1e-8 { 1.0 } else { std };

        let input_normalized: Vec<f64> = input_vec.iter().map(|x| (x - mean) / std).collect();
        let input = Array1::from_vec(input_normalized);

        // Get prediction
        let (state, confidence, _) = model.predict_single(&input)?;

        // Generate signal
        let signal = if confidence >= config.confidence_threshold {
            match state {
                prototype_learning::MarketState::Bullish
                | prototype_learning::MarketState::BreakoutUp => 1,
                prototype_learning::MarketState::Bearish
                | prototype_learning::MarketState::BreakoutDown => -1,
                prototype_learning::MarketState::Consolidation => 0,
            }
        } else {
            0 // No signal if confidence too low
        };

        // Position management (simulated)
        let position_str = match current_position {
            1 => "LONG",
            -1 => "SHORT",
            _ => "FLAT",
        };

        // Print status
        println!(
            "{:<20} {:>12.2} {:>15} {:>11.1}% {:>10} {:>12}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
            current_price,
            state.name(),
            confidence * 100.0,
            match signal {
                1 => "BUY",
                -1 => "SELL",
                _ => "HOLD",
            },
            position_str
        );

        // Simulate position changes
        if current_position == 0 && signal != 0 {
            // Open position
            current_position = signal;
            entry_price = current_price;
            entry_time = current_time;
        } else if current_position != 0 && (signal == -current_position || signal == 0) {
            // Close position
            let exit_price = current_price;
            let pnl = if current_position == 1 {
                (exit_price - entry_price) * config.position_size * 100000.0 / entry_price
            } else {
                (entry_price - exit_price) * config.position_size * 100000.0 / entry_price
            };

            let return_pct = if current_position == 1 {
                exit_price / entry_price - 1.0
            } else {
                1.0 - exit_price / entry_price
            };

            metrics.add_trade(Trade {
                entry_time,
                exit_time: current_time,
                entry_price,
                exit_price,
                direction: current_position,
                size: config.position_size,
                pnl,
                return_pct,
            });

            println!(
                "  -> Closed {} at ${:.2}, P&L: ${:.2} ({:.2}%)",
                if current_position == 1 { "LONG" } else { "SHORT" },
                exit_price,
                pnl,
                return_pct * 100.0
            );

            current_position = 0;
        }

        // Wait for next iteration
        if iteration < config.max_iterations - 1 {
            thread::sleep(Duration::from_secs(config.check_interval_secs));
        }
    }

    // Final summary
    println!("\n{}", "=".repeat(85));
    println!("\nTrading Summary:");
    println!("{}", "-".repeat(40));

    let summary = metrics.summary();
    println!("Total trades: {}", summary.total_trades);
    println!("Win rate: {:.1}%", summary.win_rate * 100.0);
    println!("Total return: {:.2}%", summary.total_return * 100.0);
    println!("Final equity: ${:.2}", summary.final_equity);

    if summary.total_trades > 0 {
        println!("Profit factor: {:.2}", summary.profit_factor);
        println!("Avg win: {:.2}%", summary.average_win * 100.0);
        println!("Avg loss: {:.2}%", summary.average_loss * 100.0);
    }

    println!("\nNote: This is a demonstration with a randomly initialized model.");
    println!("For actual trading, use a properly trained model and implement");
    println!("proper risk management, position sizing, and order execution.");

    Ok(())
}
