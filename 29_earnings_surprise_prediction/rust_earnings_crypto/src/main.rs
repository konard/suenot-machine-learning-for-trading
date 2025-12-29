//! CLI tool for crypto event surprise prediction
//!
//! Provides commands for fetching data, detecting events, and analyzing surprises.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use earnings_crypto::api::BybitClient;
use earnings_crypto::events::EventDetector;
use earnings_crypto::features::SurpriseCalculator;
use earnings_crypto::analysis::PostEventAnalyzer;

#[derive(Parser)]
#[command(name = "earnings_crypto")]
#[command(about = "Crypto event surprise prediction tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch market data from Bybit
    Fetch {
        /// Trading pair symbol (e.g., BTCUSDT)
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
        #[arg(short, long, default_value = "1h")]
        interval: String,

        /// Number of candles to fetch
        #[arg(short, long, default_value = "200")]
        limit: usize,
    },

    /// Detect events in market data
    Events {
        /// Trading pair symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Volume spike threshold (multiplier of average)
        #[arg(long, default_value = "2.0")]
        volume_threshold: f64,

        /// Price gap threshold (percentage)
        #[arg(long, default_value = "0.03")]
        gap_threshold: f64,
    },

    /// Analyze surprises in market data
    Surprise {
        /// Trading pair symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Lookback period for calculations
        #[arg(short, long, default_value = "20")]
        lookback: usize,
    },

    /// Analyze post-event drift (PEAD analog)
    Drift {
        /// Trading pair symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Days to analyze after event
        #[arg(short, long, default_value = "5")]
        days: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("earnings_crypto=info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch { symbol, interval, limit } => {
            fetch_data(&symbol, &interval, limit).await?;
        }
        Commands::Events { symbol, volume_threshold, gap_threshold } => {
            detect_events(&symbol, volume_threshold, gap_threshold).await?;
        }
        Commands::Surprise { symbol, lookback } => {
            analyze_surprises(&symbol, lookback).await?;
        }
        Commands::Drift { symbol, days } => {
            analyze_drift(&symbol, days).await?;
        }
    }

    Ok(())
}

async fn fetch_data(symbol: &str, interval: &str, limit: usize) -> Result<()> {
    println!("Fetching {} candles for {} ({} interval)...\n", limit, symbol, interval);

    let client = BybitClient::new();
    let candles = client.get_klines(symbol, interval, limit).await?;

    println!("Fetched {} candles\n", candles.len());
    println!("{:<20} {:>10} {:>10} {:>10} {:>10} {:>15}",
        "Time", "Open", "High", "Low", "Close", "Volume");
    println!("{}", "-".repeat(85));

    for candle in candles.iter().rev().take(10) {
        println!("{:<20} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>15.2}",
            candle.datetime().format("%Y-%m-%d %H:%M"),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume);
    }

    if candles.len() > 10 {
        println!("... and {} more candles", candles.len() - 10);
    }

    Ok(())
}

async fn detect_events(symbol: &str, volume_threshold: f64, gap_threshold: f64) -> Result<()> {
    println!("Detecting events for {}...\n", symbol);
    println!("Volume threshold: {}x average", volume_threshold);
    println!("Gap threshold: {:.1}%\n", gap_threshold * 100.0);

    let client = BybitClient::new();
    let candles = client.get_klines(symbol, "1h", 500).await?;

    let detector = EventDetector::new(volume_threshold, gap_threshold);
    let events = detector.detect_all_events(&candles);

    println!("Found {} events\n", events.len());
    println!("{:<20} {:<20} {:>12} {:>12}",
        "Time", "Type", "Magnitude", "Direction");
    println!("{}", "-".repeat(70));

    for event in events.iter().take(20) {
        let direction = if event.direction > 0.0 { "UP" } else { "DOWN" };
        println!("{:<20} {:<20} {:>12.2} {:>12}",
            event.datetime().format("%Y-%m-%d %H:%M"),
            format!("{:?}", event.event_type),
            event.magnitude,
            direction);
    }

    Ok(())
}

async fn analyze_surprises(symbol: &str, lookback: usize) -> Result<()> {
    println!("Analyzing surprises for {} (lookback: {} periods)...\n", symbol, lookback);

    let client = BybitClient::new();
    let candles = client.get_klines(symbol, "1h", 500).await?;

    let calculator = SurpriseCalculator::new(lookback);
    let surprises = calculator.calculate(&candles);

    // Find significant surprises
    let significant: Vec<_> = surprises.iter()
        .enumerate()
        .filter(|(_, s)| s.price_surprise.abs() > 2.0 || s.volume_surprise.abs() > 2.0)
        .collect();

    println!("Found {} significant surprises (>2 std dev)\n", significant.len());
    println!("{:<20} {:>15} {:>15} {:>12}",
        "Time", "Price Surprise", "Vol Surprise", "Return %");
    println!("{}", "-".repeat(65));

    for (i, surprise) in significant.iter().rev().take(15) {
        let candle = &candles[*i];
        println!("{:<20} {:>15.2} {:>15.2} {:>12.2}%",
            candle.datetime().format("%Y-%m-%d %H:%M"),
            surprise.price_surprise,
            surprise.volume_surprise,
            surprise.actual_return * 100.0);
    }

    // Statistics
    let price_surprises: Vec<f64> = surprises.iter().map(|s| s.price_surprise).collect();
    let vol_surprises: Vec<f64> = surprises.iter().map(|s| s.volume_surprise).collect();

    println!("\n--- Statistics ---");
    println!("Price Surprise - Mean: {:.3}, Std: {:.3}",
        mean(&price_surprises), std(&price_surprises));
    println!("Volume Surprise - Mean: {:.3}, Std: {:.3}",
        mean(&vol_surprises), std(&vol_surprises));

    Ok(())
}

async fn analyze_drift(symbol: &str, days: usize) -> Result<()> {
    println!("Analyzing post-event drift for {} ({} days window)...\n", symbol, days);

    let client = BybitClient::new();
    let candles = client.get_klines(symbol, "1h", 1000).await?;

    let detector = EventDetector::default();
    let events = detector.detect_all_events(&candles);

    let analyzer = PostEventAnalyzer::new(days * 24); // Convert to hourly candles

    let drift_results = analyzer.analyze_drift(&candles, &events);

    println!("Analyzed {} events\n", drift_results.len());
    println!("{:<20} {:>12} {:>12} {:>12} {:>12}",
        "Event Time", "Day 0 %", "Day 1 %", "Day 3 %", "Total %");
    println!("{}", "-".repeat(70));

    for result in drift_results.iter().take(10) {
        println!("{:<20} {:>12.2}% {:>12.2}% {:>12.2}% {:>12.2}%",
            result.event_time.format("%Y-%m-%d %H:%M"),
            result.day0_return * 100.0,
            result.day1_return * 100.0,
            result.day3_return * 100.0,
            result.total_return * 100.0);
    }

    // Aggregate statistics
    if !drift_results.is_empty() {
        let avg_day0: f64 = drift_results.iter().map(|r| r.day0_return).sum::<f64>() / drift_results.len() as f64;
        let avg_day1: f64 = drift_results.iter().map(|r| r.day1_return).sum::<f64>() / drift_results.len() as f64;
        let avg_total: f64 = drift_results.iter().map(|r| r.total_return).sum::<f64>() / drift_results.len() as f64;

        println!("\n--- Average Drift ---");
        println!("Day 0: {:.2}%", avg_day0 * 100.0);
        println!("Day 1: {:.2}%", avg_day1 * 100.0);
        println!("Total: {:.2}%", avg_total * 100.0);
    }

    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { return 0.0; }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std(values: &[f64]) -> f64 {
    if values.len() < 2 { return 0.0; }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}
