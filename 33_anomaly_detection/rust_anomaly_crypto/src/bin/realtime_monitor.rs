//! Real-time anomaly monitoring for cryptocurrency
//!
//! Usage: cargo run --bin realtime_monitor -- --symbol BTCUSDT --interval 1

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use std::thread;
use std::time::Duration;

use rust_anomaly_crypto::{
    anomaly::OnlineDetector,
    data::{BybitClient, BybitConfig},
    strategy::{PositionManager, SignalConfig, SignalGenerator},
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Real-time anomaly monitoring")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval (1, 5, 15)
    #[arg(short, long, default_value = "1")]
    interval: String,

    /// Polling interval in seconds
    #[arg(short, long, default_value_t = 10)]
    poll_seconds: u64,

    /// Anomaly threshold
    #[arg(short, long, default_value_t = 3.0)]
    threshold: f64,

    /// Window size for anomaly detection
    #[arg(short, long, default_value_t = 100)]
    window: usize,

    /// Enable trading signals
    #[arg(long)]
    signals: bool,

    /// Run for N iterations (0 = infinite)
    #[arg(long, default_value_t = 0)]
    iterations: usize,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           Real-Time Anomaly Monitor                      ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Symbol: {:10} | Interval: {:4} | Threshold: {:.1}     ║",
             args.symbol, args.interval, args.threshold);
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Create client and detectors
    let client = BybitClient::new(BybitConfig::new());

    let mut price_detector = OnlineDetector::new(args.window, args.threshold);
    let mut volume_detector = OnlineDetector::new(args.window, args.threshold);
    let mut return_detector = OnlineDetector::new(args.window, args.threshold);

    let mut signal_generator = SignalGenerator::new(SignalConfig::default());
    let mut position_manager = PositionManager::new(1.0);

    // Initial warmup
    println!("Warming up detectors with historical data...");
    let warmup_data = client.get_klines(&args.symbol, &args.interval, args.window + 50, None, None)?;

    let mut prev_close = 0.0;
    for candle in &warmup_data.data {
        price_detector.update(candle.close);
        volume_detector.update(candle.volume);

        if prev_close > 0.0 {
            let ret = (candle.close - prev_close) / prev_close;
            return_detector.update(ret);
        }
        prev_close = candle.close;
    }

    println!("Warmup complete. Starting real-time monitoring...\n");

    // Main loop
    let mut iteration = 0;
    let mut last_close = prev_close;

    loop {
        iteration += 1;

        if args.iterations > 0 && iteration > args.iterations {
            println!("\nReached {} iterations. Stopping.", args.iterations);
            break;
        }

        // Fetch latest data
        match client.get_klines(&args.symbol, &args.interval, 2, None, None) {
            Ok(data) => {
                if let Some(candle) = data.latest() {
                    // Calculate return
                    let current_return = if last_close > 0.0 {
                        (candle.close - last_close) / last_close
                    } else {
                        0.0
                    };

                    // Update detectors
                    let (price_score, price_anomaly, _) = price_detector.update(candle.close);
                    let (volume_score, volume_anomaly, _) = volume_detector.update(candle.volume);
                    let (return_score, return_anomaly, _) = return_detector.update(current_return);

                    // Combined anomaly score
                    let combined_score = (price_score * 0.3 + volume_score * 0.3 + return_score * 0.4).max(0.0);
                    let is_anomaly = price_anomaly || volume_anomaly || return_anomaly;

                    // Display status
                    let now = Utc::now();
                    let status_icon = if is_anomaly {
                        "🚨"
                    } else if combined_score > 0.7 {
                        "⚡"
                    } else {
                        "✓ "
                    };

                    println!(
                        "[{}] {} {} | Price: ${:.2} ({:+.2}%) | Volume: {:.0} | Score: {:.2}",
                        now.format("%H:%M:%S"),
                        status_icon,
                        args.symbol,
                        candle.close,
                        current_return * 100.0,
                        candle.volume,
                        combined_score
                    );

                    // Show individual detector scores if anomaly
                    if is_anomaly {
                        println!(
                            "    └─ Scores: Price={:.2} Volume={:.2} Return={:.2}",
                            price_score, volume_score, return_score
                        );
                    }

                    // Generate trading signals if enabled
                    if args.signals {
                        let signal = signal_generator.generate(combined_score, current_return);

                        if signal.signal != rust_anomaly_crypto::strategy::Signal::Hold {
                            let (action, _) = position_manager.process_signal(&signal, candle.close);
                            println!(
                                "    └─ Signal: {} | Action: {} | PnL: {:.4}",
                                format!("{:?}", signal.signal),
                                action,
                                position_manager.total_pnl()
                            );
                        }
                    }

                    last_close = candle.close;
                }
            }
            Err(e) => {
                println!("[{}] Error fetching data: {}", Utc::now().format("%H:%M:%S"), e);
            }
        }

        // Wait for next poll
        thread::sleep(Duration::from_secs(args.poll_seconds));
    }

    // Final summary
    if args.signals {
        println!("\n═══════════════════════════════════════");
        println!("Trading Summary:");
        println!("  Total trades: {}", position_manager.trade_count());
        println!("  Realized PnL: {:.4}", position_manager.realized_pnl());
        println!("  Total PnL: {:.4}", position_manager.total_pnl());
        println!("═══════════════════════════════════════");
    }

    Ok(())
}
