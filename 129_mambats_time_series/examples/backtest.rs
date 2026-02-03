//! Example: Backtest a MambaTS trading strategy
//!
//! This example demonstrates how to run a backtest
//! using the MambaTS model for signal generation.
//!
//! Run with: cargo run --example backtest

use chrono::{Duration, Utc};
use ndarray::Array2;
use rand::Rng;

use mambats_time_series::{
    BybitClient, Interval,
    prepare_features,
    data::features::normalize_features,
};
use mambats_time_series::model::mambats::{MambaTS, MambaTSConfig, Backtester};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("MambaTS - Backtesting Example");
    println!("==============================\n");

    // Model configuration
    let config = MambaTSConfig {
        n_variables: 17,
        d_model: 64,
        n_layers: 2,
        patch_size: 8,
        forecast_horizon: 1,
        d_state: 8,
    };

    // Create model
    let model = MambaTS::new(config);

    // Backtest parameters
    let lookback = 40;
    let threshold = 0.1;

    println!("Backtest Configuration:");
    println!("  Lookback Window: {} periods", lookback);
    println!("  Signal Threshold: {:.2}", threshold);
    println!();

    // Try to fetch real data
    let client = BybitClient::new();
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(30);

    println!("Fetching 30 days of BTCUSDT data...");

    let (prices, features) = match client.get_klines("BTCUSDT", Interval::OneHour, start_time, end_time).await {
        Ok(candles) if candles.len() >= 100 => {
            println!("Fetched {} candles\n", candles.len());

            let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
            let raw_features = prepare_features(&candles);
            let (normalized, _) = normalize_features(&raw_features);

            // Replace NaN with 0
            let features = normalized.mapv(|x| if x.is_nan() { 0.0 } else { x });

            (prices, features)
        }
        Ok(candles) => {
            println!("Not enough data ({} candles), using synthetic data...\n", candles.len());
            generate_synthetic_data(500, 17)
        }
        Err(e) => {
            println!("Could not fetch data: {}\n", e);
            println!("Using synthetic data for demonstration...\n");
            generate_synthetic_data(500, 17)
        }
    };

    println!("Data Summary:");
    println!("  Total periods: {}", prices.len());
    println!("  Feature columns: {}", features.ncols());

    if prices.len() > 1 {
        let price_change = (prices.last().unwrap() / prices.first().unwrap() - 1.0) * 100.0;
        println!("  Price change: {:.2}%", price_change);
        println!("  Start price: ${:.2}", prices.first().unwrap());
        println!("  End price: ${:.2}", prices.last().unwrap());
    }
    println!();

    // Run backtest
    println!("Running backtest...\n");

    let backtester = Backtester::new(model, threshold, lookback);

    match backtester.run(&prices, &features) {
        Ok(result) => {
            println!("Backtest Results:");
            println!("{:-<50}", "");
            println!("  Total Return:   {:>10.2}%", result.total_return * 100.0);
            println!("  Sharpe Ratio:   {:>10.2}", result.sharpe_ratio);
            println!("  Max Drawdown:   {:>10.2}%", result.max_drawdown * 100.0);
            println!("  Number of Trades: {:>8}", result.n_trades);
            println!("  Win Rate:       {:>10.2}%", result.win_rate * 100.0);
            println!("{:-<50}", "");

            // Performance commentary
            println!("\nNote: This backtest uses random model weights.");
            println!("      Train the model for meaningful results.");

            // Compare to buy-and-hold
            if prices.len() > 1 {
                let buy_hold = (prices.last().unwrap() / prices.first().unwrap() - 1.0) * 100.0;
                println!("\nBuy & Hold Return: {:.2}%", buy_hold);

                let excess = result.total_return * 100.0 - buy_hold;
                if excess > 0.0 {
                    println!("Strategy Excess:   +{:.2}%", excess);
                } else {
                    println!("Strategy Excess:   {:.2}%", excess);
                }
            }
        }
        Err(e) => {
            println!("Backtest error: {}", e);
        }
    }

    Ok(())
}

/// Generate synthetic price and feature data for demonstration
fn generate_synthetic_data(n_samples: usize, n_features: usize) -> (Vec<f64>, Array2<f64>) {
    let mut rng = rand::thread_rng();

    // Generate realistic-looking price series
    let mut prices = Vec::with_capacity(n_samples);
    let mut price = 50000.0; // Starting price (like BTC)

    for _ in 0..n_samples {
        let return_: f64 = rng.gen::<f64>() * 0.04 - 0.02; // -2% to +2%
        price *= 1.0 + return_;
        prices.push(price);
    }

    // Generate synthetic features
    let features = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        let trend = (i as f64 * 0.01).sin();
        let feature_offset = j as f64 * 0.1;
        let noise: f64 = rng.gen::<f64>() * 0.2 - 0.1;
        trend + feature_offset + noise
    });

    (prices, features)
}
