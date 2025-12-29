//! Example: Volume-based surprise analysis
//!
//! This example focuses on volume anomalies as a proxy for
//! "earnings-like" events in cryptocurrency markets.
//!
//! Run with:
//! ```bash
//! cargo run --example volume_surprise -- --symbol ETHUSDT
//! ```

use anyhow::Result;
use clap::Parser;
use earnings_crypto::api::BybitClient;
use earnings_crypto::features::{TechnicalIndicators, VolumeAnalyzer, VolumeStats};

#[derive(Parser, Debug)]
#[command(name = "volume_surprise")]
#[command(about = "Analyze volume surprises in crypto data")]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "ETHUSDT")]
    symbol: String,

    /// Kline interval
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of candles to analyze
    #[arg(short, long, default_value = "500")]
    limit: usize,

    /// Lookback period for volume analysis
    #[arg(long, default_value = "20")]
    lookback: usize,

    /// Volume anomaly threshold (std deviations)
    #[arg(long, default_value = "2.0")]
    threshold: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Volume Surprise Analyzer ===\n");
    println!("Symbol: {}", args.symbol);
    println!("Interval: {}", args.interval);
    println!("Lookback: {} periods", args.lookback);
    println!("Anomaly Threshold: {} std devs", args.threshold);
    println!();

    // Fetch data
    let client = BybitClient::new();
    println!("Fetching {} candles...", args.limit);
    let candles = client
        .get_klines(&args.symbol, &args.interval, args.limit)
        .await?;
    println!("Received {} candles\n", candles.len());

    // Create volume analyzer
    let analyzer = VolumeAnalyzer::new(args.lookback);

    // Calculate volume metrics
    let relative_vol = analyzer.relative_volume(&candles);
    let vol_zscore = analyzer.volume_zscore(&candles);
    let obv = analyzer.obv(&candles);
    let mfi = analyzer.mfi(&candles, 14);

    // Basic volume statistics
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    let stats = VolumeStats::calculate(&volumes);

    println!("=== Volume Statistics ===");
    println!("Mean Volume: {:.2}", stats.mean);
    println!("Std Dev: {:.2}", stats.std);
    println!("Median: {:.2}", stats.median);
    println!("Min: {:.2}", stats.min);
    println!("Max: {:.2}", stats.max);
    println!("Total Volume: {:.2}", stats.total);

    // Detect anomalies
    let anomalies = analyzer.detect_anomalies(&candles, args.threshold);
    let anomaly_count = anomalies.iter().filter(|&&a| a).count();

    println!("\n=== Volume Anomalies ===");
    println!(
        "Detected {} anomalies ({:.1}% of data)",
        anomaly_count,
        anomaly_count as f64 / candles.len() as f64 * 100.0
    );

    // List recent anomalies
    println!("\n=== Recent Volume Anomalies ===\n");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "Time", "Volume", "Rel. Vol", "Z-Score", "Return %", "Direction"
    );
    println!("{}", "-".repeat(80));

    let mut anomaly_indices: Vec<usize> = anomalies
        .iter()
        .enumerate()
        .filter(|(_, &is_anomaly)| is_anomaly)
        .map(|(i, _)| i)
        .collect();
    anomaly_indices.reverse();

    for idx in anomaly_indices.iter().take(15) {
        let candle = &candles[*idx];
        let rel = relative_vol[*idx];
        let zscore = vol_zscore[*idx];
        let ret = candle.return_pct() * 100.0;
        let direction = if candle.is_bullish() { "UP" } else { "DOWN" };

        println!(
            "{:<20} {:>12.0} {:>12.2}x {:>12.2} {:>10.2}% {:>10}",
            candle.datetime().format("%Y-%m-%d %H:%M"),
            candle.volume,
            rel,
            zscore,
            ret,
            direction
        );
    }

    // Analyze returns after volume spikes
    println!("\n=== Returns After Volume Spikes ===\n");

    let mut post_spike_returns = Vec::new();
    for (i, &is_anomaly) in anomalies.iter().enumerate() {
        if is_anomaly && i + 1 < candles.len() {
            let next_return = candles[i + 1].return_pct();
            let spike_direction = if candles[i].is_bullish() { 1.0 } else { -1.0 };
            post_spike_returns.push((spike_direction, next_return));
        }
    }

    if !post_spike_returns.is_empty() {
        // Continuation vs reversal
        let continuations = post_spike_returns
            .iter()
            .filter(|(dir, ret)| dir * ret > 0.0)
            .count();
        let reversals = post_spike_returns.len() - continuations;

        println!("Total Volume Spikes Analyzed: {}", post_spike_returns.len());
        println!(
            "Continuation Rate: {:.1}%",
            continuations as f64 / post_spike_returns.len() as f64 * 100.0
        );
        println!(
            "Reversal Rate: {:.1}%",
            reversals as f64 / post_spike_returns.len() as f64 * 100.0
        );

        // Average returns
        let bullish_spikes: Vec<_> = post_spike_returns
            .iter()
            .filter(|(dir, _)| *dir > 0.0)
            .map(|(_, ret)| *ret)
            .collect();
        let bearish_spikes: Vec<_> = post_spike_returns
            .iter()
            .filter(|(dir, _)| *dir < 0.0)
            .map(|(_, ret)| *ret)
            .collect();

        if !bullish_spikes.is_empty() {
            let avg: f64 = bullish_spikes.iter().sum::<f64>() / bullish_spikes.len() as f64;
            println!(
                "\nAfter Bullish Spikes - Avg Next Return: {:.3}% ({} spikes)",
                avg * 100.0,
                bullish_spikes.len()
            );
        }

        if !bearish_spikes.is_empty() {
            let avg: f64 = bearish_spikes.iter().sum::<f64>() / bearish_spikes.len() as f64;
            println!(
                "After Bearish Spikes - Avg Next Return: {:.3}% ({} spikes)",
                avg * 100.0,
                bearish_spikes.len()
            );
        }
    }

    // Volume profile
    println!("\n=== Volume Profile (Price Levels) ===\n");
    let profile = analyzer.volume_profile(&candles, 10);

    println!("{:>15} {:>20}", "Price Level", "Volume");
    println!("{}", "-".repeat(40));

    let max_vol = profile.iter().map(|(_, v)| *v).fold(0.0, f64::max);
    for (price, vol) in &profile {
        let bar_len = (vol / max_vol * 30.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("{:>15.2} {:>20.0} {}", price, vol, bar);
    }

    if let Some(poc) = analyzer.point_of_control(&candles, 10) {
        println!("\nPoint of Control (POC): {:.2}", poc);
    }

    // OBV trend
    println!("\n=== On-Balance Volume (OBV) Analysis ===");
    let obv_len = obv.len();
    if obv_len >= 20 {
        let obv_sma = TechnicalIndicators::sma(&obv, 20);

        let current_obv = obv[obv_len - 1];
        let obv_sma_val = obv_sma[obv_len - 1];

        println!("Current OBV: {:.0}", current_obv);
        println!("OBV 20-SMA: {:.0}", obv_sma_val);

        if current_obv > obv_sma_val {
            println!("OBV Trend: BULLISH (above SMA)");
        } else {
            println!("OBV Trend: BEARISH (below SMA)");
        }

        // OBV momentum
        let obv_change = current_obv - obv[obv_len - 5];
        println!("OBV 5-period Change: {:.0}", obv_change);
    }

    // Money Flow Index
    println!("\n=== Money Flow Index (MFI) ===");
    let mfi_len = mfi.len();
    if mfi_len > 0 {
        let current_mfi = mfi[mfi_len - 1];
        println!("Current MFI: {:.2}", current_mfi);

        if current_mfi > 80.0 {
            println!("MFI Signal: OVERBOUGHT");
        } else if current_mfi < 20.0 {
            println!("MFI Signal: OVERSOLD");
        } else {
            println!("MFI Signal: NEUTRAL");
        }
    }

    println!("\nDone!");
    Ok(())
}
