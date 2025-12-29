//! Detect anomalies in cryptocurrency data
//!
//! Usage: cargo run --bin detect_anomalies -- --symbol BTCUSDT --interval 60

use anyhow::Result;
use clap::Parser;
use rust_anomaly_crypto::{
    anomaly::{
        AnomalyDetector, EnsembleDetector, GlobalMADDetector, GlobalZScoreDetector,
        IQRDetector, MultivariateDetector, IsolationForest,
    },
    data::{BybitClient, BybitConfig},
    features::{FeatureConfig, FeatureEngine},
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Detect anomalies in cryptocurrency data")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to analyze
    #[arg(short, long, default_value_t = 500)]
    limit: usize,

    /// Detection method: zscore, mad, iqr, iforest, ensemble
    #[arg(short, long, default_value = "ensemble")]
    method: String,

    /// Anomaly threshold
    #[arg(short, long, default_value_t = 3.0)]
    threshold: f64,

    /// Window size for rolling calculations
    #[arg(short, long, default_value_t = 20)]
    window: usize,

    /// Show top N anomalies
    #[arg(long, default_value_t = 10)]
    top: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("Anomaly Detection");
    println!("=================");
    println!("Symbol: {}", args.symbol);
    println!("Method: {}", args.method);
    println!("Threshold: {:.2}", args.threshold);
    println!();

    // Fetch data
    let client = BybitClient::new(BybitConfig::new());
    let data = client.get_klines(&args.symbol, &args.interval, args.limit, None, None)?;

    println!("Fetched {} candles", data.len());

    if data.len() < args.window + 10 {
        println!("Not enough data. Need at least {} candles.", args.window + 10);
        return Ok(());
    }

    // Extract features
    let engine = FeatureEngine::with_config(FeatureConfig {
        window: args.window,
        ..Default::default()
    });
    let features = engine.compute(&data);

    println!("Computed {} features", features.num_features());
    println!("Valid data from index: {}", features.valid_from);

    if args.verbose {
        println!("\nFeatures: {:?}", features.names);
    }

    // Get returns for analysis
    let returns = data.returns();
    let closes = data.closes();

    // Detect anomalies
    let result = match args.method.as_str() {
        "zscore" => {
            println!("\nUsing Z-Score detector...");
            let mut detector = GlobalZScoreDetector::new(args.threshold);
            detector.fit(&returns);
            detector.detect(&returns)
        }
        "mad" => {
            println!("\nUsing MAD (Modified Z-Score) detector...");
            let mut detector = GlobalMADDetector::new(args.threshold);
            detector.fit(&returns);
            detector.detect(&returns)
        }
        "iqr" => {
            println!("\nUsing IQR detector...");
            let detector = IQRDetector::new(1.5, args.window);
            detector.detect(&returns)
        }
        "iforest" => {
            println!("\nUsing Isolation Forest detector...");
            let valid_features = features.valid_data();
            let mut detector = IsolationForest::new(100, 0.01);
            detector.fit(&valid_features);
            let iforest_result = detector.detect(&valid_features);

            // Pad result to match original data length
            let padding = features.valid_from + 1; // +1 for returns offset
            let mut padded_is_anomaly = vec![false; padding];
            padded_is_anomaly.extend(iforest_result.is_anomaly);
            let mut padded_scores = vec![0.0; padding];
            padded_scores.extend(iforest_result.scores);
            let mut padded_normalized = vec![0.0; padding];
            padded_normalized.extend(iforest_result.normalized_scores);

            rust_anomaly_crypto::anomaly::AnomalyResult::new(
                padded_is_anomaly,
                padded_scores,
                padded_normalized,
            )
        }
        "ensemble" | _ => {
            println!("\nUsing Ensemble detector (ZScore + MAD + IQR)...");
            let mut detector = EnsembleDetector::new()
                .with_threshold(0.5);
            detector.fit(&returns);
            detector.detect(&returns)
        }
    };

    // Print results
    println!("\nResults:");
    println!("  Total anomalies: {}", result.anomaly_count());
    println!("  Anomaly rate: {:.2}%", result.anomaly_rate() * 100.0);
    println!("  Max score: {:.4}", result.max_score());
    println!("  Mean score: {:.4}", result.mean_score());

    // Find top anomalies
    let mut scored_indices: Vec<(usize, f64)> = result
        .scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    scored_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop {} anomalies:", args.top);
    println!("{:>6} {:>12} {:>12} {:>12} {:>10}", "Index", "Time", "Price", "Return%", "Score");
    println!("{}", "-".repeat(60));

    for (idx, score) in scored_indices.iter().take(args.top) {
        if *idx < data.len() && *idx > 0 {
            let candle = &data.data[*idx];
            let ret = if *idx < returns.len() + 1 {
                returns[idx.saturating_sub(1)] * 100.0
            } else {
                0.0
            };

            println!(
                "{:>6} {:>12} {:>12.2} {:>+12.4} {:>10.4}",
                idx,
                candle.timestamp.format("%Y-%m-%d %H:%M"),
                candle.close,
                ret,
                score
            );
        }
    }

    // Print recent anomalies (last 24 candles)
    println!("\nRecent anomalies (last 24 candles):");
    let start_idx = data.len().saturating_sub(24);
    let mut recent_count = 0;

    for i in start_idx..data.len() {
        if i < result.is_anomaly.len() && result.is_anomaly[i] {
            let candle = &data.data[i];
            let score = result.scores.get(i).unwrap_or(&0.0);
            println!(
                "  {} - Price: ${:.2}, Score: {:.4}",
                candle.timestamp.format("%Y-%m-%d %H:%M"),
                candle.close,
                score
            );
            recent_count += 1;
        }
    }

    if recent_count == 0 {
        println!("  No anomalies in the last 24 candles");
    }

    // Price statistics
    if let Some(last_candle) = data.latest() {
        println!("\nCurrent market state:");
        println!("  Latest price: ${:.2}", last_candle.close);
        println!("  Latest volume: {:.2}", last_candle.volume);

        if let Some(&latest_score) = result.scores.last() {
            println!("  Latest anomaly score: {:.4}", latest_score);

            if latest_score > args.threshold {
                println!("  ⚠️  WARNING: Current price action is anomalous!");
            } else if latest_score > args.threshold * 0.7 {
                println!("  ⚡ CAUTION: Elevated anomaly score");
            } else {
                println!("  ✓  Normal market conditions");
            }
        }
    }

    Ok(())
}
