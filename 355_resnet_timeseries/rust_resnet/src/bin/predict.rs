//! Real-time prediction with ResNet model
//!
//! This binary demonstrates making predictions on live data from Bybit.

use anyhow::Result;
use rust_resnet::{
    api::BybitClient,
    data::{Features, StandardScaler},
    model::ResNet18,
    strategy::{TradingSignal, TradingStrategy},
};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== ResNet Real-time Prediction Demo ===\n");

    let symbol = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "BTCUSDT".to_string());

    println!("Symbol: {}", symbol);
    println!("Fetching latest data...\n");

    // Create client
    let client = BybitClient::new();

    // Fetch recent candles
    let candles = client.fetch_klines(&symbol, "1", 300).await?;

    if candles.len() < 256 {
        println!("Not enough data. Need at least 256 candles.");
        return Ok(());
    }

    println!("Fetched {} candles", candles.len());

    // Generate features
    let feature_gen = Features::default();
    let features = feature_gen.generate(&candles);

    println!("Generated {} features", features.shape()[0]);

    // Extract last 256 time steps for prediction
    let seq_len = 256;
    let start_idx = features.shape()[1] - seq_len;

    let mut input = ndarray::Array3::zeros((1, features.shape()[0], seq_len));
    for f in 0..features.shape()[0] {
        for t in 0..seq_len {
            input[[0, f, t]] = features[[f, start_idx + t]];
        }
    }

    // Normalize (using simple z-score for demo)
    let mut scaler = StandardScaler::new(features.shape()[0]);
    let normalized = scaler.fit_transform(&input);

    // Create model
    let model = ResNet18::new(features.shape()[0], 3);

    // Make prediction
    let probs = model.predict_proba(&normalized);
    let prob_down = probs[[0, 0]];
    let prob_neutral = probs[[0, 1]];
    let prob_up = probs[[0, 2]];

    println!("\n=== Prediction ===");
    println!("Down probability:    {:.2}%", prob_down * 100.0);
    println!("Neutral probability: {:.2}%", prob_neutral * 100.0);
    println!("Up probability:      {:.2}%", prob_up * 100.0);

    // Generate trading signal
    let strategy = TradingStrategy::default();
    let (signal, confidence) = strategy.generate_signal(&[prob_down, prob_neutral, prob_up]);

    let signal_str = match signal {
        TradingSignal::Long => "LONG (BUY)",
        TradingSignal::Short => "SHORT (SELL)",
        TradingSignal::Neutral => "NEUTRAL (HOLD)",
    };

    println!("\n=== Trading Signal ===");
    println!("Signal:     {}", signal_str);
    println!("Confidence: {:.2}%", confidence * 100.0);

    // Get current price
    let current_price = client.fetch_ticker(&symbol).await?;
    println!("\nCurrent price: ${:.2}", current_price);

    // Calculate position size
    let portfolio_value = 100000.0; // Demo portfolio
    let position_size = strategy.calculate_position_size(confidence, portfolio_value);
    println!("Suggested position: ${:.2}", position_size);

    println!("\nNote: This is a demo with random weights.");
    println!("Do not use for actual trading!");

    Ok(())
}
