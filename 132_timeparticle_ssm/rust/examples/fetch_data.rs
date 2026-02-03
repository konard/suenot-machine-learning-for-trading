//! Example: Fetching cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the BybitClient to fetch
//! historical kline data for analysis.

use timeparticle_ssm::api::bybit::BybitClient;

fn main() {
    println!("TimeParticle SSM - Data Fetching Example\n");
    println!("==========================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch BTC/USDT hourly data
    println!("Fetching BTC/USDT hourly data...");

    match client.fetch_klines("BTCUSDT", "60", 100) {
        Ok(klines) => {
            println!("Successfully fetched {} klines\n", klines.len());

            // Display first 5 klines
            println!("First 5 klines:");
            println!("{:<25} {:>12} {:>12} {:>12} {:>12} {:>15}",
                "Timestamp", "Open", "High", "Low", "Close", "Volume");
            println!("{}", "-".repeat(100));

            for kline in klines.iter().take(5) {
                println!(
                    "{:<25} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
                    kline.timestamp.format("%Y-%m-%d %H:%M:%S"),
                    kline.open,
                    kline.high,
                    kline.low,
                    kline.close,
                    kline.volume
                );
            }

            // Calculate some basic statistics
            if !klines.is_empty() {
                let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
                let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let avg_price: f64 = closes.iter().sum::<f64>() / closes.len() as f64;

                println!("\nStatistics:");
                println!("  Min Price:  ${:.2}", min_price);
                println!("  Max Price:  ${:.2}", max_price);
                println!("  Avg Price:  ${:.2}", avg_price);
                println!("  Range:      ${:.2}", max_price - min_price);
            }
        }
        Err(e) => {
            println!("Error fetching data: {}", e);
            println!("\nUsing simulated data instead...\n");

            // Generate simulated data
            use timeparticle_ssm::data::loader::generate_simulated_data;
            let data = generate_simulated_data(100, 42);

            println!("Generated {} simulated data points", data.len());
            println!("Sample close prices: {:?}", &data.close[..5]);
        }
    }

    // Fetch ticker
    println!("\n\nFetching current BTC/USDT ticker...");
    match client.fetch_ticker("BTCUSDT") {
        Ok(ticker) => {
            println!("Symbol: {}", ticker.symbol);
            println!("Last Price: {}", ticker.last_price);
            println!("24h High: {}", ticker.high_price_24h);
            println!("24h Low: {}", ticker.low_price_24h);
            println!("24h Volume: {}", ticker.volume_24h);
        }
        Err(e) => {
            println!("Error fetching ticker: {}", e);
        }
    }

    println!("\n==========================================");
    println!("Data fetching example complete!");
}
