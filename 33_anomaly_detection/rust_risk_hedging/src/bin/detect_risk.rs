//! Detect risk anomalies in market data
//!
//! Usage: cargo run --bin detect_risk -- --symbol BTCUSDT

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use rust_risk_hedging::{
    anomaly::{AnomalyLevel, EnsembleDetector},
    data::BybitClient,
    features::RiskFeatures,
    risk::{HedgingStrategy, RiskSignal},
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Detect risk anomalies in cryptocurrency markets")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles for analysis
    #[arg(short, long, default_value = "200")]
    limit: usize,

    /// Portfolio value for hedge calculations
    #[arg(short, long, default_value = "100000")]
    portfolio: f64,

    /// Show detailed feature analysis
    #[arg(long)]
    detailed: bool,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("{}", "=== Risk Anomaly Detection ===".bold());
    println!();

    // Fetch data
    println!("Fetching data for {}...", args.symbol.cyan());
    let client = BybitClient::public();
    let data = client.get_klines(&args.symbol, &args.interval, args.limit, None, None)?;

    if data.is_empty() {
        println!("{}", "No data received!".red());
        return Ok(());
    }

    // Extract features
    println!("Extracting risk features...");
    let features = RiskFeatures::from_ohlcv(&data);

    // Run ensemble detection
    println!("Running anomaly detection...\n");
    let detector = EnsembleDetector::default();
    let results = detector.detect_from_ohlcv(&data);

    // Get latest result
    if let Some(latest) = results.last() {
        print_risk_summary(&args.symbol, latest.score, latest.level);

        if args.detailed {
            print_detailed_analysis(&features, latest);
        }

        // Hedging recommendation
        println!("\n{}", "=== Hedging Recommendation ===".bold());
        let strategy = HedgingStrategy::default();
        let allocation = strategy.decide(latest.score, args.portfolio);

        print_hedge_recommendation(&allocation, args.portfolio);

        // Signal
        let signal = RiskSignal::from_level(latest.level);
        println!("\n{}", "=== Action Signal ===".bold());
        println!("Signal: {:?}", signal);
        println!("Urgency: {}/10", signal.urgency());
        println!("{}", signal.description());
    }

    // Historical analysis
    println!("\n{}", "=== Historical Risk Levels ===".bold());
    print_risk_history(&results);

    Ok(())
}

fn print_risk_summary(symbol: &str, score: f64, level: AnomalyLevel) {
    let level_str = match level {
        AnomalyLevel::Normal => "NORMAL".green(),
        AnomalyLevel::Elevated => "ELEVATED".yellow(),
        AnomalyLevel::High => "HIGH".truecolor(255, 165, 0), // Orange
        AnomalyLevel::Extreme => "EXTREME".red().bold(),
    };

    println!("{}", "=== Current Risk Status ===".bold());
    println!("Symbol: {}", symbol.cyan());
    println!("Risk Score: {:.2} / 1.00", score);
    println!("Risk Level: {}", level_str);

    // Visual bar
    let bar_len = 40;
    let filled = (score * bar_len as f64) as usize;
    let bar: String = (0..bar_len)
        .map(|i| if i < filled { '█' } else { '░' })
        .collect();

    let colored_bar = if score < 0.5 {
        bar.green()
    } else if score < 0.7 {
        bar.yellow()
    } else if score < 0.9 {
        bar.truecolor(255, 165, 0)
    } else {
        bar.red()
    };

    println!("Risk Meter: [{}]", colored_bar);
}

fn print_detailed_analysis(features: &RiskFeatures, result: &rust_risk_hedging::anomaly::AnomalyResult) {
    println!("\n{}", "=== Detailed Analysis ===".bold());

    println!("Detector Contributions:");
    println!("  Z-Score:     {:.2} ({:.0}%)",
        result.zscore_contribution,
        result.zscore_contribution * 100.0);
    println!("  Isolation:   {:.2} ({:.0}%)",
        result.isolation_contribution,
        result.isolation_contribution * 100.0);
    println!("  Mahalanobis: {:.2} ({:.0}%)",
        result.mahalanobis_contribution,
        result.mahalanobis_contribution * 100.0);

    println!("\nLatest Feature Values:");
    let latest = features.latest_vector();
    let labels = [
        "Return", "Volatility", "Volume Change", "ATR",
        "BB Width", "RSI", "Max DD", "Momentum", "VWAP Dev", "Vol Percentile"
    ];

    for (label, value) in labels.iter().zip(&latest) {
        let formatted = if value.abs() > 100.0 {
            format!("{:.0}", value)
        } else {
            format!("{:.4}", value)
        };
        println!("  {:15}: {}", label, formatted);
    }

    println!("\nComposite Risk Score: {:.2}", features.composite_risk_score());
}

fn print_hedge_recommendation(
    allocation: &rust_risk_hedging::risk::HedgeAllocation,
    portfolio: f64,
) {
    println!("Total Hedge: {:.1}%", allocation.total_hedge_pct * 100.0);
    println!("Estimated Annual Cost: {:.2}%", allocation.estimated_annual_cost * 100.0);
    println!("Reason: {}", allocation.reason);

    if !allocation.allocations.is_empty() {
        println!("\nAllocation by Instrument:");
        let amounts = allocation.dollar_amounts(portfolio);
        for (instrument, amount) in &amounts {
            let pct = allocation.allocations.get(instrument).unwrap_or(&0.0);
            println!("  {:20}: ${:>10.2} ({:.1}%)",
                format!("{:?}", instrument),
                amount,
                pct * 100.0);
        }
    } else {
        println!("\n{}", "No hedging required at this time.".green());
    }
}

fn print_risk_history(results: &[rust_risk_hedging::anomaly::AnomalyResult]) {
    // Show distribution of risk levels
    let mut normal = 0;
    let mut elevated = 0;
    let mut high = 0;
    let mut extreme = 0;

    for result in results {
        match result.level {
            AnomalyLevel::Normal => normal += 1,
            AnomalyLevel::Elevated => elevated += 1,
            AnomalyLevel::High => high += 1,
            AnomalyLevel::Extreme => extreme += 1,
        }
    }

    let total = results.len() as f64;

    println!("Risk Level Distribution (last {} periods):", results.len());
    println!("  {} Normal:   {:>4} ({:>5.1}%)", "●".green(), normal, normal as f64 / total * 100.0);
    println!("  {} Elevated: {:>4} ({:>5.1}%)", "●".yellow(), elevated, elevated as f64 / total * 100.0);
    println!("  {} High:     {:>4} ({:>5.1}%)", "●".truecolor(255, 165, 0), high, high as f64 / total * 100.0);
    println!("  {} Extreme:  {:>4} ({:>5.1}%)", "●".red(), extreme, extreme as f64 / total * 100.0);

    // Average score trend
    if results.len() >= 10 {
        let recent_avg: f64 = results.iter().rev().take(10).map(|r| r.score).sum::<f64>() / 10.0;
        let older_avg: f64 = results.iter().rev().skip(10).take(10).map(|r| r.score).sum::<f64>() / 10.0.min(results.len() as f64 - 10.0).max(1.0);

        let trend = if recent_avg > older_avg * 1.1 {
            "↑ INCREASING".red()
        } else if recent_avg < older_avg * 0.9 {
            "↓ DECREASING".green()
        } else {
            "→ STABLE".normal()
        };

        println!("\nRisk Trend: {}", trend);
        println!("  Recent avg:  {:.2}", recent_avg);
        println!("  Older avg:   {:.2}", older_avg);
    }
}
