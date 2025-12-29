//! Example: Full analysis pipeline
//!
//! This example demonstrates the complete workflow:
//! 1. Fetch data from Bybit
//! 2. Detect events (volume spikes, gaps, etc.)
//! 3. Calculate surprise metrics
//! 4. Make predictions
//! 5. Analyze post-event drift (PEAD analog)
//!
//! Run with:
//! ```bash
//! cargo run --example full_pipeline -- --symbol BTCUSDT
//! ```

use anyhow::Result;
use clap::Parser;
use earnings_crypto::api::BybitClient;
use earnings_crypto::events::{EventDetector, EventSummary};
use earnings_crypto::features::SurpriseCalculator;
use earnings_crypto::models::SimplePredictor;
use earnings_crypto::analysis::PostEventAnalyzer;

#[derive(Parser, Debug)]
#[command(name = "full_pipeline")]
#[command(about = "Run the complete event surprise analysis pipeline")]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval (use 1h for best results)
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of candles to analyze (recommend 500+)
    #[arg(short, long, default_value = "720")]
    limit: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       CRYPTO EVENT SURPRISE PREDICTION PIPELINE             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Configuration:");
    println!("  Symbol: {}", args.symbol);
    println!("  Interval: {}", args.interval);
    println!("  Candles: {}", args.limit);
    println!();

    // ═══════════════════════════════════════════════════════════════
    // STEP 1: Fetch Data
    // ═══════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ STEP 1: Fetching Market Data                               │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let client = BybitClient::new();
    let candles = client
        .get_klines(&args.symbol, &args.interval, args.limit)
        .await?;

    println!("✓ Fetched {} candles", candles.len());

    let first = &candles[0];
    let last = &candles[candles.len() - 1];
    println!(
        "  Period: {} to {}",
        first.datetime().format("%Y-%m-%d %H:%M"),
        last.datetime().format("%Y-%m-%d %H:%M")
    );

    let price_range = (
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max),
    );
    println!("  Price Range: {:.2} - {:.2}", price_range.0, price_range.1);
    println!();

    // ═══════════════════════════════════════════════════════════════
    // STEP 2: Detect Events
    // ═══════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ STEP 2: Detecting Market Events                            │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let detector = EventDetector::new(2.0, 0.02).with_lookback(20);
    let events = detector.detect_all_events(&candles);
    let summary = EventSummary::from_events(&events);

    println!("✓ Detected {} events", events.len());
    println!("  Bullish: {}", summary.bullish_count);
    println!("  Bearish: {}", summary.bearish_count);
    println!("  Average Magnitude: {:.2}", summary.avg_magnitude);

    for (event_type, count) in &summary.by_type {
        println!("  {:?}: {}", event_type, count);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // STEP 3: Calculate Surprise Metrics
    // ═══════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ STEP 3: Calculating Surprise Metrics                       │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let surprise_calc = SurpriseCalculator::new(20);
    let surprises = surprise_calc.calculate(&candles);
    let surprise_stats = surprise_calc.rolling_stats(&surprises);

    println!("✓ Calculated {} surprise metrics", surprises.len());
    println!("  Positive Surprises: {} ({:.1}%)",
        surprise_stats.positive_surprises,
        surprise_stats.positive_rate() * 100.0
    );
    println!("  Negative Surprises: {} ({:.1}%)",
        surprise_stats.negative_surprises,
        surprise_stats.negative_rate() * 100.0
    );
    println!("  Avg Price Surprise: {:.3}", surprise_stats.avg_price_surprise);
    println!("  Avg Volume Surprise: {:.3}", surprise_stats.avg_volume_surprise);

    let extremes = surprise_calc.find_extremes(&surprises, 2.5);
    println!("  Extreme Surprises (>2.5 std): {}", extremes.len());
    println!();

    // ═══════════════════════════════════════════════════════════════
    // STEP 4: Make Predictions
    // ═══════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ STEP 4: Running Prediction Model                           │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let predictor = SimplePredictor::new(20, 2.0);
    let backtest_results = predictor.backtest(&candles);
    let metrics = predictor.calculate_metrics(&backtest_results);

    println!("✓ Backtest Results:");
    println!("  Total Predictions: {}", metrics.total_predictions);
    println!("  Accuracy: {:.1}% ({}/{})",
        metrics.accuracy * 100.0,
        metrics.correct_predictions,
        metrics.total_predictions
    );
    println!("  Bullish Accuracy: {:.1}%", metrics.bullish_accuracy * 100.0);
    println!("  Bearish Accuracy: {:.1}%", metrics.bearish_accuracy * 100.0);
    println!("  Cumulative Return: {:.2}%", metrics.cumulative_return * 100.0);

    // Current prediction
    if let Some(prediction) = predictor.predict_from_candles(&candles) {
        println!("\n  Current Signal: {} (Confidence: {:.0}%)",
            prediction.direction,
            prediction.confidence * 100.0
        );
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // STEP 5: Analyze Post-Event Drift (PEAD)
    // ═══════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ STEP 5: Analyzing Post-Event Drift (PEAD Analog)           │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let pead_analyzer = PostEventAnalyzer::new(24); // 24 hours window
    let drift_results = pead_analyzer.analyze_drift(&candles, &events);
    let drift_stats = pead_analyzer.aggregate_stats(&drift_results);

    println!("✓ Analyzed {} events for drift", drift_stats.total_events);
    println!("  Day 0 Return: {:.2}%", drift_stats.avg_day0_return * 100.0);
    println!("  Day 1 Return: {:.2}%", drift_stats.avg_day1_return * 100.0);
    println!("  Day 3 Return: {:.2}%", drift_stats.avg_day3_return * 100.0);
    println!("  Day 5 Return: {:.2}%", drift_stats.avg_day5_return * 100.0);
    println!("  Total Drift: {:.2}%", drift_stats.avg_total_return * 100.0);
    println!("  Drift Continuation Rate: {:.1}%",
        drift_stats.drift_continuation_rate * 100.0
    );

    // Drift by event magnitude
    println!("\n  Drift by Event Magnitude:");
    let by_magnitude = pead_analyzer.analyze_by_magnitude(&drift_results);
    for (label, stats) in &by_magnitude {
        if stats.total_events > 0 {
            println!("    {}: {:.2}% drift ({} events)",
                label,
                stats.avg_total_return * 100.0,
                stats.total_events
            );
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                        SUMMARY                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Key Findings for {}:", args.symbol);
    println!();

    // Event frequency
    let events_per_day = events.len() as f64 / (candles.len() as f64 / 24.0);
    println!("1. EVENT FREQUENCY");
    println!("   {:.1} significant events per day on average", events_per_day);
    println!();

    // Surprise distribution
    println!("2. SURPRISE DISTRIBUTION");
    if surprise_stats.positive_rate() > surprise_stats.negative_rate() {
        println!("   Market shows positive skew (more positive surprises)");
    } else if surprise_stats.negative_rate() > surprise_stats.positive_rate() {
        println!("   Market shows negative skew (more negative surprises)");
    } else {
        println!("   Market shows balanced surprise distribution");
    }
    println!();

    // Prediction accuracy
    println!("3. PREDICTABILITY");
    if metrics.accuracy > 0.55 {
        println!("   Good predictability: {:.1}% accuracy", metrics.accuracy * 100.0);
        println!("   Simple model shows edge over random (50%)");
    } else if metrics.accuracy > 0.50 {
        println!("   Marginal predictability: {:.1}% accuracy", metrics.accuracy * 100.0);
        println!("   Slight edge, but needs refinement");
    } else {
        println!("   Low predictability: {:.1}% accuracy", metrics.accuracy * 100.0);
        println!("   Market appears efficient or model needs improvement");
    }
    println!();

    // PEAD effect
    println!("4. POST-EVENT DRIFT (PEAD Analog)");
    if drift_stats.drift_continuation_rate > 0.55 {
        println!("   Strong drift continuation: {:.1}%", drift_stats.drift_continuation_rate * 100.0);
        println!("   Events tend to follow-through (momentum)");
    } else if drift_stats.drift_continuation_rate < 0.45 {
        println!("   Mean reversion tendency: {:.1}% continuation", drift_stats.drift_continuation_rate * 100.0);
        println!("   Events tend to reverse");
    } else {
        println!("   Neutral drift pattern: {:.1}% continuation", drift_stats.drift_continuation_rate * 100.0);
    }
    println!();

    // Trading implications
    println!("5. TRADING IMPLICATIONS");
    if metrics.cumulative_return > 0.0 && metrics.accuracy > 0.52 {
        println!("   ✓ Potential trading opportunity detected");
        println!("   ✓ Model shows positive expectancy");
    } else {
        println!("   ⚠ No clear edge detected with simple model");
        println!("   ⚠ Consider more sophisticated approaches");
    }
    println!();

    // Current market state
    if let Some(prediction) = predictor.predict_from_candles(&candles) {
        println!("6. CURRENT STATE");
        println!("   Direction Bias: {}", prediction.direction);
        println!("   Confidence: {:.0}%", prediction.confidence * 100.0);

        let last_candle = &candles[candles.len() - 1];
        println!("   Last Price: {:.2}", last_candle.close);
        println!("   Last Return: {:.2}%", last_candle.return_pct() * 100.0);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Pipeline completed successfully!");
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}
