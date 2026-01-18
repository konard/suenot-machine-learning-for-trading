//! Bybit Data Fetching Example
//!
//! Demonstrates fetching historical crypto data from Bybit API.

use rust_dct::api::BybitClient;

#[tokio::main]
async fn main() {
    println!("=== Bybit Data Fetching Example ===\n");

    let client = BybitClient::new();

    // Fetch BTCUSDT daily data
    println!("Fetching BTCUSDT daily klines...");

    match client
        .get_klines("BTCUSDT", "D", 100, None, None)
        .await
    {
        Ok(klines) => {
            println!("Fetched {} klines\n", klines.len());

            // Display first 5 and last 5 klines
            println!("First 5 klines:");
            println!("{:-<80}", "");
            println!(
                "{:>12} {:>12} {:>12} {:>12} {:>12} {:>15}",
                "Timestamp", "Open", "High", "Low", "Close", "Volume"
            );
            println!("{:-<80}", "");

            for kline in klines.iter().take(5) {
                println!(
                    "{:>12} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
                    kline.timestamp,
                    kline.open,
                    kline.high,
                    kline.low,
                    kline.close,
                    kline.volume
                );
            }

            if klines.len() > 10 {
                println!("...");
                println!("\nLast 5 klines:");
                println!("{:-<80}", "");

                for kline in klines.iter().rev().take(5).rev() {
                    println!(
                        "{:>12} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
                        kline.timestamp,
                        kline.open,
                        kline.high,
                        kline.low,
                        kline.close,
                        kline.volume
                    );
                }
            }

            // Calculate some statistics
            println!("\n=== Statistics ===");
            let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
            let first_price = closes.first().unwrap_or(&0.0);
            let last_price = closes.last().unwrap_or(&0.0);
            let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let avg_price: f64 = closes.iter().sum::<f64>() / closes.len() as f64;

            println!("Period Return: {:.2}%", (last_price / first_price - 1.0) * 100.0);
            println!("Min Price: ${:.2}", min_price);
            println!("Max Price: ${:.2}", max_price);
            println!("Avg Price: ${:.2}", avg_price);

            // Calculate daily returns volatility
            let returns: Vec<f64> = closes
                .windows(2)
                .map(|w| (w[1] / w[0]).ln())
                .collect();
            let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns
                .iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>()
                / (returns.len() - 1) as f64;
            let daily_vol = variance.sqrt();
            let annual_vol = daily_vol * (252.0_f64).sqrt();

            println!("Daily Volatility: {:.2}%", daily_vol * 100.0);
            println!("Annualized Volatility: {:.2}%", annual_vol * 100.0);
        }
        Err(e) => {
            println!("Error fetching data: {}", e);
            println!("\nNote: Make sure you have internet connectivity.");
            println!("The Bybit API endpoint is: https://api.bybit.com");
        }
    }

    // Fetch other intervals
    println!("\n=== Available Intervals ===");
    println!("1, 3, 5, 15, 30, 60, 120, 240, 360, 720 (minutes)");
    println!("D (daily), W (weekly), M (monthly)");

    println!("\n=== Example Complete ===");
}
