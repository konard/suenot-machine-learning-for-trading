//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to:
//! 1. Connect to Bybit API
//! 2. Fetch OHLCV (candlestick) data
//! 3. Calculate returns
//! 4. Display statistics

use normalizing_flows_finance::prelude::*;
use normalizing_flows_finance::utils::statistics::describe;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Normalizing Flows for Finance - Data Fetching Example");
    println!("═══════════════════════════════════════════════════════════");

    // Create Bybit client
    println!("\n[1] Connecting to Bybit API...");
    let client = BybitClient::new();

    // Fetch Bitcoin daily data
    println!("\n[2] Fetching BTC/USDT daily data...");
    let symbol = "BTCUSDT";
    let interval = "D"; // Daily
    let limit = 365; // ~1 year

    match client.get_klines_sync(symbol, interval, limit) {
        Ok(klines) => {
            println!("    Fetched {} candles", klines.len());

            // Display recent prices
            println!("\n[3] Recent price data:");
            println!("    {:^12} {:>12} {:>12} {:>12} {:>15}",
                     "Date", "Open", "High", "Low", "Close");
            println!("    {}", "-".repeat(65));

            for kline in klines.iter().rev().take(5) {
                let date = chrono::DateTime::from_timestamp_millis(kline.timestamp)
                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                    .unwrap_or_else(|| "Unknown".to_string());
                println!("    {:^12} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
                         date, kline.open, kline.high, kline.low, kline.close);
            }

            // Calculate returns
            println!("\n[4] Calculating log returns...");
            let returns = calculate_log_returns(&klines);
            println!("    Calculated {} returns", returns.len());

            // Statistics
            println!("\n[5] Return statistics:");
            let stats = describe(&returns);
            println!("    Count:      {}", stats.count);
            println!("    Mean:       {:+.4}%", stats.mean * 100.0);
            println!("    Std Dev:    {:.4}%", stats.std * 100.0);
            println!("    Min:        {:+.4}%", stats.min * 100.0);
            println!("    Max:        {:+.4}%", stats.max * 100.0);
            println!("    Skewness:   {:+.4}", stats.skewness);
            println!("    Kurtosis:   {:+.4}", stats.kurtosis);

            // Distribution info
            println!("\n[6] Return distribution:");
            println!("    1st percentile:  {:+.4}%", stats.q1 * 100.0);
            println!("    Median:          {:+.4}%", stats.median * 100.0);
            println!("    3rd percentile:  {:+.4}%", stats.q3 * 100.0);

            // Fat tails analysis
            let tail_threshold = -0.05; // -5%
            let tail_count = returns.iter().filter(|&&r| r < tail_threshold).count();
            let tail_pct = tail_count as f64 / returns.len() as f64 * 100.0;
            println!("\n[7] Tail analysis:");
            println!("    Days with > 5% loss: {} ({:.2}%)", tail_count, tail_pct);

            // If returns were Gaussian with same mean/std:
            let z_score = (tail_threshold - stats.mean) / stats.std;
            // Approximate Gaussian tail probability
            let gaussian_prob = 0.5 * (1.0 + libm::erf(-z_score / std::f64::consts::SQRT_2));
            println!("    Expected under Gaussian: {:.2}%", gaussian_prob * 100.0);
            println!("    Fat tail ratio: {:.1}x", tail_pct / (gaussian_prob * 100.0));
        }
        Err(e) => {
            println!("    Error fetching data: {}", e);
            println!("\n    Using synthetic data instead...");

            // Generate synthetic returns
            let returns = normalizing_flows_finance::utils::returns::generate_synthetic_returns(365, 4.0);
            let stats = describe(&returns);

            println!("\n[5] Synthetic return statistics (Student-t with df=4):");
            println!("    Count:      {}", stats.count);
            println!("    Mean:       {:+.4}%", stats.mean * 100.0);
            println!("    Std Dev:    {:.4}%", stats.std * 100.0);
            println!("    Skewness:   {:+.4}", stats.skewness);
            println!("    Kurtosis:   {:+.4} (Gaussian = 0)", stats.kurtosis);
        }
    }

    // Fetch multiple symbols
    println!("\n[8] Fetching multiple symbols...");
    let symbols = &["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    match client.get_multiple_klines_sync(symbols, "D", 100) {
        Ok(data) => {
            println!("    Fetched data for {} symbols", data.len());

            for (symbol, klines) in &data {
                let returns = calculate_log_returns(klines);
                if !returns.is_empty() {
                    let stats = describe(&returns);
                    println!("    {}: mean={:+.2}%, std={:.2}%",
                             symbol, stats.mean * 100.0, stats.std * 100.0);
                }
            }
        }
        Err(e) => {
            println!("    Error: {}", e);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Data fetching complete!");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
