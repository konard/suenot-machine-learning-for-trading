//! # Calculate VPIN Example
//!
//! Demonstrates VPIN (Volume-Synchronized Probability of Informed Trading) calculation.
//!
//! Run with: `cargo run --example calculate_vpin`

use anyhow::Result;
use order_flow_imbalance::BybitClient;
use order_flow_imbalance::orderflow::vpin::VpinCalculator;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                  VPIN Calculator                          ║");
    println!("║     Volume-Synchronized Probability of Informed Trading   ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    let client = BybitClient::new();

    // VPIN with smaller buckets for demo (normally use larger buckets)
    let bucket_size = 1.0; // 1 BTC per bucket
    let num_buckets = 10;  // 10 buckets for rolling VPIN

    let mut vpin_calculator = VpinCalculator::new(bucket_size, num_buckets);

    println!("Configuration:");
    println!("  Bucket Size:   {} BTC", bucket_size);
    println!("  Num Buckets:   {}", num_buckets);
    println!();
    println!("Collecting trades to calculate VPIN...");
    println!("Press Ctrl+C to stop");
    println!();

    let mut iteration = 0;
    let mut total_volume = 0.0;

    loop {
        iteration += 1;

        // Fetch recent trades
        let trades = match client.get_trades("BTCUSDT", 100).await {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error fetching trades: {}", e);
                sleep(Duration::from_secs(1)).await;
                continue;
            }
        };

        // Process trades
        for trade in &trades {
            if let Some(vpin) = vpin_calculator.add_trade(trade) {
                // New VPIN calculated (bucket completed)
                let stats = vpin_calculator.statistics();

                println!("─────────────────────────────────────────────────────────────");
                println!("  NEW VPIN VALUE: {:.4}", vpin);
                println!("─────────────────────────────────────────────────────────────");
                println!();

                // Toxicity interpretation
                let level = stats.toxicity_level();
                let indicator = match level {
                    order_flow_imbalance::orderflow::vpin::ToxicityLevel::Low => "🟢 LOW",
                    order_flow_imbalance::orderflow::vpin::ToxicityLevel::Medium => "🟡 MEDIUM",
                    order_flow_imbalance::orderflow::vpin::ToxicityLevel::High => "🟠 HIGH",
                    order_flow_imbalance::orderflow::vpin::ToxicityLevel::VeryHigh => "🔴 VERY HIGH",
                };

                println!("  Toxicity Level: {}", indicator);
                println!();

                if vpin > 0.7 {
                    println!("  ⚠️  WARNING: High toxicity detected!");
                    println!("      Informed traders may be active in the market.");
                    println!("      Consider reducing position size or avoiding trades.");
                    println!();
                }

                println!("  Statistics:");
                println!("    Current:  {:.4}", stats.current);
                println!("    Mean:     {:.4}", stats.mean);
                println!("    Std:      {:.4}", stats.std);
                println!("    Min/Max:  {:.4} / {:.4}", stats.min, stats.max);
                println!();
                println!("  Percentiles:");
                println!("    25th:     {:.4}", stats.percentile_25);
                println!("    50th:     {:.4}", stats.percentile_50);
                println!("    75th:     {:.4}", stats.percentile_75);
                println!("    95th:     {:.4}", stats.percentile_95);
                println!();
            }

            total_volume += trade.size;
        }

        // Progress update
        let bucket_progress = vpin_calculator.current_bucket_progress();
        let completed = vpin_calculator.bucket_count();

        println!("  Iteration: {} | Volume: {:.4} BTC | Buckets: {} | Progress: {:.0}%",
            iteration,
            total_volume,
            completed,
            bucket_progress * 100.0
        );

        // Show current VPIN if available
        if let Some(vpin) = vpin_calculator.current_vpin() {
            println!("  Current VPIN: {:.4}", vpin);
        } else {
            println!("  Current VPIN: Waiting for {} more buckets...", num_buckets - completed);
        }

        println!();

        // Wait before next update
        sleep(Duration::from_secs(2)).await;

        // Stop after sufficient data for demo
        if completed >= num_buckets + 5 {
            println!("═══════════════════════════════════════════════════════════");
            println!("Demo complete.");
            println!();

            let stats = vpin_calculator.statistics();
            println!("FINAL VPIN ANALYSIS");
            println!("───────────────────────────────────────────────────────────");
            println!("  Total Volume Processed: {:.4} BTC", total_volume);
            println!("  Buckets Completed:      {}", completed);
            println!();
            println!("  Final VPIN:            {:.4}", stats.current);
            println!("  Average VPIN:          {:.4}", stats.mean);
            println!("  Toxicity Level:        {}", stats.toxicity_level());
            println!();

            // Market assessment
            println!("MARKET ASSESSMENT");
            println!("───────────────────────────────────────────────────────────");
            if stats.mean < 0.3 {
                println!("  The market appears to be dominated by UNINFORMED traders.");
                println!("  Low probability of adverse selection risk.");
            } else if stats.mean < 0.5 {
                println!("  Mixed market with some informed trading activity.");
                println!("  Moderate caution advised.");
            } else if stats.mean < 0.7 {
                println!("  Elevated informed trading activity detected.");
                println!("  Consider reducing position sizes.");
            } else {
                println!("  HIGH informed trading activity!");
                println!("  Market makers face significant adverse selection risk.");
                println!("  Consider avoiding market-making or directional bets.");
            }
            println!();

            break;
        }
    }

    Ok(())
}
