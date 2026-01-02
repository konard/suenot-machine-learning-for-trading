//! Live Prediction Demo
//!
//! Demonstrates real-time prediction using DenseNet.
//!
//! Usage:
//!   cargo run --bin live_predict -- --symbol BTCUSDT

use std::env;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          DenseNet Trading - Live Prediction Demo              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let args: Vec<String> = env::args().collect();
    let symbol = get_arg(&args, "--symbol").unwrap_or("BTCUSDT".to_string());

    println!("Symbol: {}", symbol);
    println!();
    println!("Note: This is a demonstration of the prediction interface.");
    println!("In production, this would:");
    println!("  1. Connect to Bybit WebSocket for real-time data");
    println!("  2. Maintain a rolling window of candles");
    println!("  3. Extract features on each new candle");
    println!("  4. Run DenseNet inference");
    println!("  5. Generate and execute trading signals");
    println!();

    // Create model
    use rust_densenet_trading::densenet::{DenseNet, DenseNetConfig};

    let config = DenseNetConfig::tiny(); // Fast for real-time
    let model = DenseNet::new(config.clone());

    println!("Model loaded: DenseNet-Tiny");
    println!("  Parameters: {}", model.num_parameters());
    println!("  Input: {} features x {} sequence", config.input_features, config.sequence_length);
    println!();

    // Simulate live predictions
    println!("Simulating live predictions...");
    println!("{:-<70}", "");
    println!(
        "{:<12} {:>12} {:>10} {:>12} {:>12} {:>10}",
        "Time", "Price", "Signal", "Confidence", "Position", "P&L"
    );
    println!("{:-<70}", "");

    let mut position = 0; // -1 short, 0 flat, 1 long
    let mut entry_price = 0.0;
    let mut total_pnl = 0.0;
    let mut price = 45000.0;

    // Simulate 20 time steps
    for i in 0..20 {
        // Simulate price movement
        let change = ((i as f64) * 0.5).sin() * 200.0 + (i as f64 - 10.0) * 10.0;
        price += change;

        // Generate synthetic features and prediction
        let input = ndarray::Array2::from_shape_fn(
            (config.input_features, config.sequence_length),
            |(f, t)| {
                let base = price / 45000.0;
                let feature_noise = (f as f64 * 0.1 + t as f64 * 0.01).sin() * 0.1;
                base + feature_noise
            },
        );

        let prediction = model.forward(&input, false);
        let signal = prediction.signal();
        let confidence = prediction.confidence;

        // Trading logic
        let new_position = match signal {
            rust_densenet_trading::densenet::network::TradingAction::Long if confidence > 0.4 => 1,
            rust_densenet_trading::densenet::network::TradingAction::Short if confidence > 0.4 => -1,
            _ => 0,
        };

        // Calculate P&L if closing position
        let trade_pnl = if position != 0 && new_position != position {
            let pnl = position as f64 * (price - entry_price);
            total_pnl += pnl;
            pnl
        } else {
            0.0
        };

        // Update position
        if new_position != position && new_position != 0 {
            entry_price = price;
        }
        position = new_position;

        // Display
        let time = format!("{:02}:{:02}:00", 9 + i / 4, (i % 4) * 15);
        let signal_str = match signal {
            rust_densenet_trading::densenet::network::TradingAction::Long => "LONG",
            rust_densenet_trading::densenet::network::TradingAction::Short => "SHORT",
            rust_densenet_trading::densenet::network::TradingAction::Hold => "HOLD",
        };
        let pos_str = match position {
            1 => "LONG",
            -1 => "SHORT",
            _ => "FLAT",
        };

        println!(
            "{:<12} {:>12.2} {:>10} {:>11.1}% {:>12} {:>+10.2}",
            time,
            price,
            signal_str,
            confidence * 100.0,
            pos_str,
            trade_pnl
        );

        // Small delay simulation
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    println!("{:-<70}", "");
    println!("Total P&L: ${:.2}", total_pnl);
    println!();
    println!("Live prediction demo complete!");
    println!();
    println!("To use with real data:");
    println!("  1. First run: cargo run --bin fetch_bybit_data -- --symbol {}", symbol);
    println!("  2. Load the CSV data");
    println!("  3. Connect to WebSocket for live updates");
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
