//! Example: Generate live trading signals
//!
//! This example demonstrates how to generate live trading signals using
//! a trained ConvNeXt model with real-time Bybit data.
//!
//! Usage:
//!   cargo run --example live_signals -- --symbol BTCUSDT --interval 1h

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use convnext_trading::convnext::{ConvNeXt, ConvNeXtConfig};
use convnext_trading::data::{BybitClient, FeatureBuilder, Interval};
use convnext_trading::trading::Signal;

#[derive(Parser)]
#[command(name = "live_signals")]
#[command(about = "Generate live trading signals using ConvNeXt")]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candlestick interval (1m, 5m, 15m, 1h, 4h, 1d)
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Path to model weights (optional, uses random weights if not provided)
    #[arg(short, long)]
    model: Option<String>,

    /// Confidence threshold for signals
    #[arg(short, long, default_value = "0.55")]
    confidence: f64,

    /// Run once and exit (don't loop)
    #[arg(long)]
    once: bool,

    /// Sequence length for model input
    #[arg(long, default_value = "256")]
    seq_length: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    info!("=== ConvNeXt Live Signal Generator ===");
    info!("Symbol: {}", args.symbol);
    info!("Interval: {}", args.interval);
    info!("Confidence threshold: {:.0}%", args.confidence * 100.0);

    // Parse interval
    let interval = Interval::from_str(&args.interval)?;

    // Load or create model
    let model = if let Some(path) = &args.model {
        info!("Loading model from {}", path);
        ConvNeXt::load(path)?
    } else {
        warn!("No model specified, using random weights (for demonstration only)");
        ConvNeXt::new(ConvNeXtConfig::tiny())
    };

    info!("Model loaded with {} parameters", model.count_params());

    // Create client and feature builder
    let client = BybitClient::new();
    let feature_builder = FeatureBuilder::new();

    if args.once {
        // Generate single signal
        let signal = generate_signal(
            &client,
            &model,
            &feature_builder,
            &args.symbol,
            interval,
            args.seq_length,
        )
        .await?;

        print_signal(&args.symbol, &signal).await?;
    } else {
        // Continuous signal generation
        info!("\nStarting live signal generation...");
        info!("Press Ctrl+C to stop\n");

        loop {
            let signal = generate_signal(
                &client,
                &model,
                &feature_builder,
                &args.symbol,
                interval,
                args.seq_length,
            )
            .await?;

            print_signal(&args.symbol, &signal).await?;

            // Wait for next candle
            let wait_secs = match interval {
                Interval::M1 => 60,
                Interval::M5 => 300,
                Interval::M15 => 900,
                Interval::H1 => 3600,
                Interval::H4 => 14400,
                Interval::D1 => 86400,
            };

            info!(
                "Next signal in {} seconds...\n",
                wait_secs
            );

            tokio::time::sleep(tokio::time::Duration::from_secs(wait_secs)).await;
        }
    }

    Ok(())
}

async fn generate_signal(
    client: &BybitClient,
    model: &ConvNeXt,
    feature_builder: &FeatureBuilder,
    symbol: &str,
    interval: Interval,
    seq_length: usize,
) -> Result<Signal> {
    // Fetch recent data
    let end_time = Utc::now();
    let candles_needed = seq_length + 50; // Extra for feature calculation
    let start_time = end_time - chrono::Duration::milliseconds(
        interval.duration_ms() * candles_needed as i64
    );

    let candles = client
        .get_klines(
            symbol,
            interval,
            start_time.timestamp_millis(),
            end_time.timestamp_millis(),
        )
        .await?;

    if candles.len() < seq_length {
        warn!(
            "Not enough data: {} candles, need {}",
            candles.len(),
            seq_length
        );
        return Ok(Signal::Hold);
    }

    // Build features
    let features = feature_builder.build(&candles)?;
    let (n_candles, n_features) = features.dim();

    // Get last sequence
    let start_idx = n_candles.saturating_sub(seq_length);
    let seq_features = features
        .slice(ndarray::s![start_idx.., ..])
        .to_owned();

    // Reshape to [1, features, seq_length] for model input
    let input = seq_features
        .into_shape((1, seq_length, n_features))?
        .permuted_axes([0, 2, 1])
        .to_owned();

    // Run inference
    let output = model.forward(&input);

    // Create signal
    let long_prob = output[[0, 0]];
    let short_prob = output[[0, 1]];
    let hold_prob = output[[0, 2]];

    Ok(Signal::from_probs(long_prob, short_prob, hold_prob))
}

async fn print_signal(symbol: &str, signal: &Signal) -> Result<()> {
    let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S UTC");

    let (action, color) = match signal {
        Signal::Long { confidence } => {
            (format!("LONG ({:.1}%)", confidence * 100.0), "\x1b[32m") // Green
        }
        Signal::Short { confidence } => {
            (format!("SHORT ({:.1}%)", confidence * 100.0), "\x1b[31m") // Red
        }
        Signal::Hold => {
            ("HOLD".to_string(), "\x1b[33m") // Yellow
        }
    };

    println!(
        "[{}] {} | {}{}\x1b[0m",
        timestamp, symbol, color, action
    );

    Ok(())
}
