//! Real-time prediction with WebSocket streaming
//!
//! This example demonstrates real-time trading signal generation.

use efficientnet_trading::api::{BybitClient, BybitWebSocket, WebSocketMessage};
use efficientnet_trading::data::Candle;
use efficientnet_trading::imaging::CandlestickRenderer;
use efficientnet_trading::model::ModelPredictor;
use efficientnet_trading::strategy::{Signal, SignalGenerator, SignalType};
use std::collections::VecDeque;
use tokio::time::{timeout, Duration};

const WINDOW_SIZE: usize = 50;
const CONFIDENCE_THRESHOLD: f64 = 0.6;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Real-time EfficientNet Prediction ===\n");

    // Initialize components
    let client = BybitClient::new();
    let renderer = CandlestickRenderer::new(224, 224);
    let predictor = ModelPredictor::new(224);
    let signal_gen = SignalGenerator::new(CONFIDENCE_THRESHOLD);

    // Fetch initial historical data
    println!("Fetching initial data...");
    let initial_candles = client.fetch_klines("BTCUSDT", "1", WINDOW_SIZE + 10).await?;
    println!("  Fetched {} historical candles\n", initial_candles.len());

    // Initialize candle buffer
    let mut candle_buffer: VecDeque<Candle> = initial_candles.into_iter().collect();

    // Keep only last WINDOW_SIZE candles
    while candle_buffer.len() > WINDOW_SIZE {
        candle_buffer.pop_front();
    }

    // Connect to WebSocket
    println!("Connecting to Bybit WebSocket...");
    let mut ws = BybitWebSocket::new();
    ws.subscribe_kline("BTCUSDT", "1");

    let mut rx = ws.connect().await?;
    println!("Connected! Waiting for real-time data...\n");

    println!("{:>12} {:>12} {:>10} {:>12} {:>8}",
        "Time", "Price", "Signal", "Confidence", "Action");
    println!("{}", "=".repeat(60));

    let mut signal_count = 0;
    let max_signals = 20;

    // Process real-time data with timeout
    loop {
        if signal_count >= max_signals {
            println!("\nReached {} signals, stopping.", max_signals);
            break;
        }

        // Wait for message with timeout
        let msg = match timeout(Duration::from_secs(30), rx.recv()).await {
            Ok(Some(msg)) => msg,
            Ok(None) => {
                println!("WebSocket closed");
                break;
            }
            Err(_) => {
                println!("Timeout waiting for data, continuing...");
                continue;
            }
        };

        match msg {
            WebSocketMessage::Candle(candle) => {
                // Update buffer
                candle_buffer.push_back(candle.clone());
                if candle_buffer.len() > WINDOW_SIZE {
                    candle_buffer.pop_front();
                }

                // Skip if not enough data
                if candle_buffer.len() < WINDOW_SIZE {
                    continue;
                }

                // Generate image and predict
                let candles: Vec<Candle> = candle_buffer.iter().cloned().collect();
                let image = renderer.render(&candles);
                let prediction = predictor.predict(&image)?;

                // Generate signal
                let signal = signal_gen.from_prediction(
                    &prediction,
                    candle.timestamp,
                    candle.close,
                );

                // Display result
                let time_str = format_timestamp(candle.timestamp);
                let signal_str = format!("{:?}", signal.signal_type);
                let action = if signal.is_actionable(CONFIDENCE_THRESHOLD) {
                    match signal.signal_type {
                        SignalType::Buy => ">>> BUY",
                        SignalType::Sell => "<<< SELL",
                        SignalType::Hold => "",
                    }
                } else {
                    ""
                };

                println!("{:>12} {:>12.2} {:>10} {:>11.1}% {:>8}",
                    time_str,
                    candle.close,
                    signal_str,
                    signal.confidence * 100.0,
                    action
                );

                signal_count += 1;
            }
            WebSocketMessage::Connected => {
                println!(">>> WebSocket connected");
            }
            WebSocketMessage::Disconnected => {
                println!(">>> WebSocket disconnected");
                break;
            }
            WebSocketMessage::Error(e) => {
                println!(">>> Error: {}", e);
            }
            _ => {}
        }
    }

    // Summary
    println!("\n=== Session Summary ===");
    println!("Processed {} candles", signal_count);
    println!("Window size: {} candles", WINDOW_SIZE);
    println!("Confidence threshold: {:.0}%", CONFIDENCE_THRESHOLD * 100.0);

    println!("\nNote: This demo uses mock predictions.");
    println!("For real trading, load a trained model with:");
    println!("  ModelPredictor::load(\"path/to/model.pt\")");

    Ok(())
}

fn format_timestamp(ts: u64) -> String {
    let secs = ts / 1000;
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let secs = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}
