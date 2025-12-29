//! Example: Event detection and analysis
//!
//! This example demonstrates how to detect significant market events
//! (volume spikes, price gaps, volatility expansion) in cryptocurrency data.
//!
//! Run with:
//! ```bash
//! cargo run --example event_analysis -- --symbol BTCUSDT
//! ```

use anyhow::Result;
use clap::Parser;
use earnings_crypto::api::BybitClient;
use earnings_crypto::events::{EventDetector, EventSummary, EventType};

#[derive(Parser, Debug)]
#[command(name = "event_analysis")]
#[command(about = "Detect and analyze market events in crypto data")]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of candles to analyze
    #[arg(short, long, default_value = "500")]
    limit: usize,

    /// Volume spike threshold (multiplier of average)
    #[arg(long, default_value = "2.0")]
    volume_threshold: f64,

    /// Price gap threshold (percentage as decimal)
    #[arg(long, default_value = "0.02")]
    gap_threshold: f64,

    /// Lookback period for calculations
    #[arg(long, default_value = "20")]
    lookback: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Crypto Event Analyzer ===\n");
    println!("Symbol: {}", args.symbol);
    println!("Interval: {}", args.interval);
    println!("Volume Threshold: {}x average", args.volume_threshold);
    println!("Gap Threshold: {:.1}%", args.gap_threshold * 100.0);
    println!();

    // Fetch data
    let client = BybitClient::new();
    println!("Fetching {} candles...", args.limit);
    let candles = client
        .get_klines(&args.symbol, &args.interval, args.limit)
        .await?;
    println!("Received {} candles\n", candles.len());

    // Create detector
    let detector = EventDetector::new(args.volume_threshold, args.gap_threshold)
        .with_lookback(args.lookback);

    // Detect events
    println!("Detecting events...\n");

    // Volume events
    let volume_events = detector.detect_volume_events(&candles);
    println!("Volume Spikes: {}", volume_events.len());

    // Gap events
    let gap_events = detector.detect_gap_events(&candles);
    println!("Price Gaps: {}", gap_events.len());

    // Volatility events
    let volatility_events = detector.detect_volatility_events(&candles);
    println!("Volatility Expansions: {}", volatility_events.len());

    // Large moves
    let large_moves = detector.detect_large_moves(&candles);
    println!("Large Moves: {}", large_moves.len());

    // All events (merged)
    let all_events = detector.detect_all_events(&candles);
    println!("\nTotal Events (merged): {}", all_events.len());

    // Event summary
    let summary = EventSummary::from_events(&all_events);
    println!("\n=== Event Summary ===");
    println!("Bullish Events: {}", summary.bullish_count);
    println!("Bearish Events: {}", summary.bearish_count);
    println!("Average Magnitude: {:.2}", summary.avg_magnitude);

    // Display events by type
    println!("\nEvents by Type:");
    for (event_type, count) in &summary.by_type {
        println!("  {:?}: {}", event_type, count);
    }

    // Display recent events
    println!("\n=== Recent Events (Last 15) ===\n");
    println!(
        "{:<20} {:<22} {:>10} {:>12} {:>12} {:>12}",
        "Time", "Type", "Magnitude", "Direction", "Price", "Volume"
    );
    println!("{}", "-".repeat(95));

    for event in all_events.iter().rev().take(15) {
        let direction = if event.direction > 0.0 {
            "BULLISH"
        } else {
            "BEARISH"
        };

        println!(
            "{:<20} {:<22} {:>10.2} {:>12} {:>12.2} {:>12.0}",
            event.datetime().format("%Y-%m-%d %H:%M"),
            format!("{:?}", event.event_type),
            event.magnitude,
            direction,
            event.price,
            event.volume
        );
    }

    // Analyze event clustering
    println!("\n=== Event Clustering Analysis ===\n");

    // Group events by day
    let mut events_by_day: std::collections::HashMap<String, Vec<_>> =
        std::collections::HashMap::new();

    for event in &all_events {
        let day = event.datetime().format("%Y-%m-%d").to_string();
        events_by_day.entry(day).or_default().push(event);
    }

    // Find days with multiple events
    let mut clustered_days: Vec<_> = events_by_day
        .iter()
        .filter(|(_, events)| events.len() >= 2)
        .collect();
    clustered_days.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    println!("Days with Multiple Events:");
    for (day, events) in clustered_days.iter().take(10) {
        let bullish = events.iter().filter(|e| e.direction > 0.0).count();
        let bearish = events.len() - bullish;
        println!(
            "  {} - {} events (Bullish: {}, Bearish: {})",
            day,
            events.len(),
            bullish,
            bearish
        );
    }

    // Analyze strongest events
    println!("\n=== Strongest Events (Top 10) ===\n");
    let mut sorted_events = all_events.clone();
    sorted_events.sort_by(|a, b| b.magnitude.partial_cmp(&a.magnitude).unwrap());

    for event in sorted_events.iter().take(10) {
        let direction = if event.direction > 0.0 {
            "BULLISH"
        } else {
            "BEARISH"
        };

        println!(
            "{} - {:?} (Magnitude: {:.2}, {})",
            event.datetime().format("%Y-%m-%d %H:%M"),
            event.event_type,
            event.magnitude,
            direction
        );

        // Show metadata if available
        if let Some(vol_mult) = event.metadata.volume_multiple {
            println!("    Volume: {:.1}x average", vol_mult);
        }
        if let Some(gap) = event.metadata.gap_size {
            println!("    Gap: {:.2}%", gap);
        }
        if let Some(vol_mult) = event.metadata.volatility_multiple {
            println!("    Volatility: {:.1}x average", vol_mult);
        }
        if let Some(ret) = event.metadata.return_pct {
            println!("    Return: {:.2}%", ret * 100.0);
        }
    }

    println!("\nDone!");
    Ok(())
}
