//! Fetch cryptocurrency data from Bybit
//!
//! This binary fetches OHLCV data from Bybit exchange.
//!
//! Usage:
//!   cargo run --bin fetch_bybit_data -- --symbol BTCUSDT --interval 15 --limit 1000

use anyhow::Result;
use rust_densenet_trading::data::{BybitClient, Candle};
use rust_densenet_trading::utils::io::save_to_csv;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        DenseNet Trading - Bybit Data Fetcher                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse command line arguments (simple parsing)
    let args: Vec<String> = env::args().collect();

    let symbol = get_arg(&args, "--symbol").unwrap_or("BTCUSDT".to_string());
    let interval = get_arg(&args, "--interval").unwrap_or("15".to_string());
    let limit: usize = get_arg(&args, "--limit")
        .unwrap_or("500".to_string())
        .parse()
        .unwrap_or(500);

    println!("Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Interval: {} minutes", interval);
    println!("  Limit: {} candles", limit);
    println!();

    // Create Bybit client
    let client = BybitClient::new();

    println!("Fetching data from Bybit...");

    // Fetch klines
    let candles = client
        .get_klines(&symbol, &interval, limit, None, None)
        .await?;

    println!("Received {} candles", candles.len());

    if !candles.is_empty() {
        // Display sample data
        println!();
        println!("Sample data (first 5 candles):");
        println!("{:-<80}", "");
        println!(
            "{:<20} {:>12} {:>12} {:>12} {:>12} {:>15}",
            "Timestamp", "Open", "High", "Low", "Close", "Volume"
        );
        println!("{:-<80}", "");

        for candle in candles.iter().take(5) {
            let dt = rust_densenet_trading::utils::time::format_iso(candle.timestamp);
            println!(
                "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.4}",
                &dt[..19],
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume
            );
        }

        // Calculate some basic statistics
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let mean = closes.iter().sum::<f64>() / closes.len() as f64;
        let min = closes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Returns
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
        let volatility = if !returns.is_empty() {
            let ret_mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance =
                returns.iter().map(|r| (r - ret_mean).powi(2)).sum::<f64>() / returns.len() as f64;
            variance.sqrt() * (252.0 * 24.0 * 4.0_f64).sqrt() // Annualized for 15m
        } else {
            0.0
        };

        println!();
        println!("Statistics:");
        println!("  Mean Price: ${:.2}", mean);
        println!("  Min Price: ${:.2}", min);
        println!("  Max Price: ${:.2}", max);
        println!("  Price Range: ${:.2}", max - min);
        println!("  Annualized Volatility: {:.2}%", volatility * 100.0);

        // Save to CSV
        let filename = format!("{}_{}_data.csv", symbol.to_lowercase(), interval);
        println!();
        println!("Saving data to {}...", filename);
        save_to_csv(&candles, &filename)?;
        println!("Data saved successfully!");
    }

    // Also fetch order book
    println!();
    println!("Fetching order book...");

    let orderbook = client.get_orderbook(&symbol, 25).await?;

    if let Some(mid) = orderbook.mid_price() {
        println!("Order Book for {}:", symbol);
        println!("  Mid Price: ${:.2}", mid);
        println!("  Spread: ${:.4}", orderbook.spread().unwrap_or(0.0));
        println!(
            "  Spread %: {:.4}%",
            orderbook.spread_percent().unwrap_or(0.0)
        );
        println!("  Order Imbalance: {:.4}", orderbook.order_imbalance());
        println!(
            "  Bid Volume (top 5): {:.4}",
            orderbook.bid_volume_at_depth(5)
        );
        println!(
            "  Ask Volume (top 5): {:.4}",
            orderbook.ask_volume_at_depth(5)
        );
    }

    // Fetch ticker
    println!();
    println!("Fetching ticker info...");

    match client.get_ticker(&symbol).await {
        Ok(ticker) => {
            println!("Ticker for {}:", symbol);
            println!("  Last Price: ${:.2}", ticker.last_price);
            println!("  24h High: ${:.2}", ticker.high_24h);
            println!("  24h Low: ${:.2}", ticker.low_24h);
            println!("  24h Change: {:.2}%", ticker.price_change_24h * 100.0);
            println!("  24h Volume: {:.2}", ticker.volume_24h);
            println!("  Funding Rate: {:.6}%", ticker.funding_rate * 100.0);
            println!("  Open Interest: {:.2}", ticker.open_interest);
        }
        Err(e) => {
            println!("Could not fetch ticker: {}", e);
        }
    }

    println!();
    println!("Done!");

    Ok(())
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
