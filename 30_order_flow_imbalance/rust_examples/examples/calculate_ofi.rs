//! # Calculate OFI Example
//!
//! Demonstrates Order Flow Imbalance calculation from live data.
//!
//! Run with: `cargo run --example calculate_ofi`

use anyhow::Result;
use order_flow_imbalance::BybitClient;
use order_flow_imbalance::orderflow::ofi::OrderFlowCalculator;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║         Order Flow Imbalance (OFI) Calculator             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Calculating OFI for BTCUSDT...");
    println!("Press Ctrl+C to stop");
    println!();

    let client = BybitClient::new();
    let mut ofi_calculator = OrderFlowCalculator::new();

    // Collect order book snapshots
    let mut iteration = 0;

    loop {
        iteration += 1;

        // Fetch current order book
        let orderbook = match client.get_orderbook("BTCUSDT", 50).await {
            Ok(ob) => ob,
            Err(e) => {
                eprintln!("Error fetching order book: {}", e);
                sleep(Duration::from_secs(1)).await;
                continue;
            }
        };

        // Calculate OFI
        let ofi = ofi_calculator.update(&orderbook);

        // Get statistics
        let mid = orderbook.mid_price().unwrap_or(0.0);
        let spread_bps = orderbook.spread_bps().unwrap_or(0.0);
        let depth_imb = orderbook.depth_imbalance(5);

        // Display
        println!("─────────────────────────────────────────────────────────────");
        println!("  Iteration: {} | Time: {}", iteration, orderbook.timestamp.format("%H:%M:%S%.3f"));
        println!("─────────────────────────────────────────────────────────────");
        println!("  Mid Price:      ${:.2}", mid);
        println!("  Spread:         {:.2} bps", spread_bps);
        println!("  Depth Imbalance:{:.4}", depth_imb);
        println!();

        if let Some(current_ofi) = ofi {
            // Determine signal direction
            let signal = if current_ofi > 10.0 {
                "▲ BUY PRESSURE"
            } else if current_ofi < -10.0 {
                "▼ SELL PRESSURE"
            } else {
                "◆ NEUTRAL"
            };

            println!("  Current OFI:    {:>+.4}", current_ofi);
            println!("  Signal:         {}", signal);
        }

        // Rolling metrics
        println!();
        println!("  Rolling OFI Metrics:");
        println!("    1-min sum:    {:>+.4}", ofi_calculator.ofi_1min());
        println!("    5-min sum:    {:>+.4}", ofi_calculator.ofi_5min());
        println!("    Cumulative:   {:>+.4}", ofi_calculator.cumulative());

        if let Some(zscore) = ofi_calculator.z_score(50) {
            println!("    Z-Score:      {:>+.4}", zscore);

            // Extreme z-score warning
            if zscore.abs() > 2.0 {
                println!();
                println!("  ⚠️  EXTREME IMBALANCE DETECTED! Z-Score: {:.2}", zscore);
            }
        }

        // Statistics
        if iteration % 10 == 0 && iteration > 10 {
            let stats = ofi_calculator.statistics();
            println!();
            println!("  Statistics (n={}):", stats.count);
            println!("    Mean:         {:>+.4}", stats.mean);
            println!("    Std:          {:>.4}", stats.std);
            println!("    Min/Max:      {:>+.4} / {:>+.4}", stats.min, stats.max);
            println!("    +/- Ratio:    {:.1}% / {:.1}%",
                stats.positive_ratio * 100.0,
                stats.negative_ratio * 100.0
            );
        }

        println!();

        // Wait before next update
        sleep(Duration::from_millis(500)).await;

        // Stop after 100 iterations for demo
        if iteration >= 100 {
            println!("═══════════════════════════════════════════════════════════");
            println!("Demo complete. {} iterations processed.", iteration);

            let stats = ofi_calculator.statistics();
            println!();
            println!("FINAL STATISTICS");
            println!("───────────────────────────────────────────────────────────");
            println!("  Total Updates:  {}", stats.count);
            println!("  Cumulative OFI: {:+.4}", stats.cumulative);
            println!("  Mean OFI:       {:+.4}", stats.mean);
            println!("  Std Dev:        {:.4}", stats.std);
            println!();

            if stats.cumulative > 0.0 {
                println!("  Overall: NET BUY PRESSURE over the period");
            } else {
                println!("  Overall: NET SELL PRESSURE over the period");
            }
            println!();
            break;
        }
    }

    Ok(())
}
