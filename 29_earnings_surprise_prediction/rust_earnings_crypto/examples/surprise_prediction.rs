//! Example: Surprise prediction and analysis
//!
//! This example demonstrates how to calculate "earnings-like" surprise metrics
//! for cryptocurrency data and use them for prediction.
//!
//! Run with:
//! ```bash
//! cargo run --example surprise_prediction -- --symbol BTCUSDT --lookback 20
//! ```

use anyhow::Result;
use clap::Parser;
use earnings_crypto::api::BybitClient;
use earnings_crypto::features::{SurpriseCalculator, SurpriseDirection};
use earnings_crypto::models::SimplePredictor;

#[derive(Parser, Debug)]
#[command(name = "surprise_prediction")]
#[command(about = "Calculate and predict market surprises")]
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

    /// Lookback period for surprise calculation
    #[arg(long, default_value = "20")]
    lookback: usize,

    /// Run backtest
    #[arg(long)]
    backtest: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Crypto Surprise Predictor ===\n");
    println!("Symbol: {}", args.symbol);
    println!("Interval: {}", args.interval);
    println!("Lookback: {} periods", args.lookback);
    println!();

    // Fetch data
    let client = BybitClient::new();
    println!("Fetching {} candles...", args.limit);
    let candles = client
        .get_klines(&args.symbol, &args.interval, args.limit)
        .await?;
    println!("Received {} candles\n", candles.len());

    // Calculate surprises
    let calculator = SurpriseCalculator::new(args.lookback);
    let surprises = calculator.calculate(&candles);

    println!("Calculated {} surprise metrics\n", surprises.len());

    // Get statistics
    let stats = calculator.rolling_stats(&surprises);

    println!("=== Surprise Statistics ===");
    println!("Total Observations: {}", stats.count);
    println!(
        "Positive Surprises: {} ({:.1}%)",
        stats.positive_surprises,
        stats.positive_rate() * 100.0
    );
    println!(
        "Negative Surprises: {} ({:.1}%)",
        stats.negative_surprises,
        stats.negative_rate() * 100.0
    );
    println!();
    println!("Price Surprise - Mean: {:.3}, Std: {:.3}", stats.avg_price_surprise, stats.std_price_surprise);
    println!("Volume Surprise - Mean: {:.3}, Std: {:.3}", stats.avg_volume_surprise, stats.std_volume_surprise);
    println!("Composite Score - Mean: {:.3}, Range: [{:.3}, {:.3}]",
        stats.avg_composite, stats.min_composite, stats.max_composite);

    // Find extreme surprises
    let extremes = calculator.find_extremes(&surprises, 2.0);
    println!("\n=== Extreme Surprises (|score| > 2.0): {} found ===\n", extremes.len());

    println!(
        "{:<20} {:>15} {:>15} {:>12} {:>10}",
        "Time", "Price Surprise", "Vol Surprise", "Composite", "Direction"
    );
    println!("{}", "-".repeat(75));

    for surprise in extremes.iter().rev().take(15) {
        let idx = surprises.iter().position(|s| s.timestamp == surprise.timestamp).unwrap();
        let candle = &candles[idx + args.lookback];

        let direction = match surprise.direction() {
            SurpriseDirection::PositiveSurprise => "POSITIVE",
            SurpriseDirection::NegativeSurprise => "NEGATIVE",
            SurpriseDirection::AsExpected => "NEUTRAL",
        };

        println!(
            "{:<20} {:>15.2} {:>15.2} {:>12.2} {:>10}",
            candle.datetime().format("%Y-%m-%d %H:%M"),
            surprise.price_surprise,
            surprise.volume_surprise,
            surprise.composite_score,
            direction
        );
    }

    // Surprise persistence (autocorrelation)
    println!("\n=== Surprise Persistence ===");
    for lag in [1, 2, 3, 5, 10] {
        let persistence = calculator.surprise_persistence(&surprises, lag);
        println!("Lag {}: {:.4}", lag, persistence);
    }

    // Make current prediction
    let predictor = SimplePredictor::new(args.lookback, 2.0);

    if let Some(prediction) = predictor.predict_from_candles(&candles) {
        println!("\n=== Current Prediction ===");
        println!("Direction: {}", prediction.direction);
        println!("Confidence: {:.2}%", prediction.confidence * 100.0);
        println!("Probability: {:.2}%", prediction.probability * 100.0);

        println!("\nFeatures:");
        println!("  Return Trend: {:.4}", prediction.features.return_trend);
        println!("  Volume Trend: {:.4}", prediction.features.volume_trend);
        println!("  Momentum: {:.4}", prediction.features.momentum);
        println!("  Volatility Percentile: {:.2}%", prediction.features.volatility_percentile * 100.0);
        println!("  Last Return: {:.4}%", prediction.features.last_return * 100.0);
        println!("  Last Volume Z-Score: {:.2}", prediction.features.last_volume_zscore);
    }

    // Run backtest if requested
    if args.backtest {
        println!("\n=== Backtest Results ===\n");

        let results = predictor.backtest(&candles);
        let metrics = predictor.calculate_metrics(&results);

        println!("{}", metrics);

        // Analyze by confidence level
        let high_conf: Vec<_> = results.iter().filter(|r| r.prediction.confidence > 0.5).collect();
        let low_conf: Vec<_> = results.iter().filter(|r| r.prediction.confidence <= 0.5).collect();

        if !high_conf.is_empty() {
            let high_accuracy = high_conf.iter().filter(|r| r.was_correct).count() as f64
                / high_conf.len() as f64;
            println!("\nHigh Confidence (>0.5): {:.1}% accuracy ({} trades)",
                high_accuracy * 100.0, high_conf.len());
        }

        if !low_conf.is_empty() {
            let low_accuracy = low_conf.iter().filter(|r| r.was_correct).count() as f64
                / low_conf.len() as f64;
            println!("Low Confidence (<=0.5): {:.1}% accuracy ({} trades)",
                low_accuracy * 100.0, low_conf.len());
        }

        // Show recent predictions
        println!("\n=== Recent Predictions (Last 10) ===\n");
        println!(
            "{:<20} {:>12} {:>10} {:>12} {:>8}",
            "Time", "Direction", "Conf %", "Actual %", "Correct"
        );
        println!("{}", "-".repeat(65));

        for result in results.iter().rev().take(10) {
            let idx = candles.iter().position(|c| c.timestamp == result.timestamp).unwrap();
            let candle = &candles[idx];

            println!(
                "{:<20} {:>12} {:>10.1}% {:>12.2}% {:>8}",
                candle.datetime().format("%Y-%m-%d %H:%M"),
                result.prediction.direction.to_string(),
                result.prediction.confidence * 100.0,
                result.actual_return * 100.0,
                if result.was_correct { "YES" } else { "NO" }
            );
        }
    }

    println!("\nDone!");
    Ok(())
}
