//! Example: Trading Signals with Bybit Data
//!
//! This example demonstrates how to fetch cryptocurrency data from Bybit
//! and generate trading signals using Mamba4Cast.

use mamba4cast::prelude::*;

fn main() {
    println!("Mamba4Cast Trading Signals Example");
    println!("===================================\n");

    // Create forecaster
    let forecaster = Mamba4CastForecaster::new(64, 4, 16, 96);

    // Note: Uncomment the following to fetch real data from Bybit
    // This requires network access and may fail if the API is unavailable

    /*
    // Create Bybit client
    let client = BybitClient::new();

    // Fetch Bitcoin kline data
    println!("Fetching BTCUSDT kline data from Bybit...");
    match client.fetch_klines("BTCUSDT", "60", 500) {
        Ok(klines) => {
            println!("  Fetched {} klines", klines.len());

            // Extract close prices
            let prices = BybitClient::extract_close_prices(&klines);

            // Generate forecast
            let result = forecaster.forecast(&prices, 100, 24);

            println!("\nForecast Results:");
            println!("  Current Price:   ${:.2}", result.current_price);
            println!("  Expected Return: {:.2}%", result.expected_return * 100.0);

            // Generate trading signal
            let signal = forecaster.trading_signals(&prices, 100, 24, 0.01);

            println!("\nTrading Signal:");
            println!("  Signal:     {}", signal.signal);
            println!("  Confidence: {:.2}%", signal.confidence * 100.0);
        }
        Err(e) => {
            println!("Failed to fetch data: {}", e);
        }
    }
    */

    // Demo with simulated data
    println!("Using simulated cryptocurrency data...\n");

    // Simulate Bitcoin-like price data
    let btc_prices: Vec<f32> = simulate_crypto_prices(500, 45000.0, 0.02);
    let eth_prices: Vec<f32> = simulate_crypto_prices(500, 3000.0, 0.025);

    // Process multiple assets
    let assets = vec![
        ("BTC", btc_prices),
        ("ETH", eth_prices),
    ];

    println!("Multi-Asset Analysis");
    println!("--------------------");

    for (name, prices) in &assets {
        println!("\n{}:", name);

        let result = forecaster.forecast(prices, 100, 24);
        let signal = forecaster.trading_signals(prices, 100, 24, 0.01);

        println!("  Current Price:   ${:.2}", result.current_price);
        println!("  24h Forecast:    ${:.2}", result.forecast.last().unwrap_or(&0.0));
        println!("  Expected Return: {:.2}%", result.expected_return * 100.0);
        println!("  Signal:          {}", signal.signal);
        println!("  Confidence:      {:.2}%", signal.confidence * 100.0);
    }

    println!("\n");

    // Multi-horizon analysis
    println!("Multi-Horizon Analysis (BTC)");
    println!("----------------------------");

    let horizons = vec![1, 6, 12, 24];

    for h in horizons {
        let result = forecaster.forecast(&assets[0].1, 100, h);
        let signal = forecaster.trading_signals(&assets[0].1, 100, h, 0.01);

        println!("  {}h Horizon:", h);
        println!("    Expected Return: {:>7.2}%", result.expected_return * 100.0);
        println!("    Signal:          {}", signal.signal);
    }

    println!("\nExample completed successfully!");
}

/// Simulate cryptocurrency price data.
fn simulate_crypto_prices(length: usize, start_price: f32, volatility: f32) -> Vec<f32> {
    let mut prices = Vec::with_capacity(length);
    let mut price = start_price;

    for i in 0..length {
        // Random walk with slight trend
        let trend = 0.0001 * (i as f32);
        let noise = (rand::random::<f32>() - 0.5) * 2.0 * volatility;

        price *= 1.0 + trend + noise;
        prices.push(price);
    }

    prices
}
