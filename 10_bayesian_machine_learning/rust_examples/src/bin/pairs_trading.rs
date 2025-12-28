//! Bayesian Rolling Regression for Pairs Trading
//!
//! This example demonstrates how to use Bayesian linear regression
//! to dynamically estimate hedge ratios for pairs trading in crypto markets.
//!
//! The rolling Bayesian approach provides:
//! - Time-varying hedge ratio estimates
//! - Uncertainty bands for the hedge ratio
//! - Detection of regime changes in the relationship

use anyhow::Result;
use bayesian_crypto::bayesian::linear_regression::RollingBayesianRegression;
use bayesian_crypto::data::{correlation, BybitClient, Returns, Symbol};
use clap::Parser;
use colored::Colorize;
use tabled::{Table, Tabled};

#[derive(Parser, Debug)]
#[command(name = "pairs_trading")]
#[command(about = "Bayesian rolling regression for crypto pairs trading")]
struct Args {
    /// First trading symbol (Y variable)
    #[arg(short = '1', long, default_value = "ETHUSDT")]
    symbol1: String,

    /// Second trading symbol (X variable)
    #[arg(short = '2', long, default_value = "BTCUSDT")]
    symbol2: String,

    /// Time interval (1, 5, 15, 30, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "500")]
    limit: u32,

    /// Rolling window size
    #[arg(short, long, default_value = "60")]
    window: usize,
}

#[derive(Tabled)]
struct HedgeRatioRow {
    #[tabled(rename = "Period")]
    period: String,
    #[tabled(rename = "Hedge Ratio")]
    hedge_ratio: String,
    #[tabled(rename = "Std Error")]
    std_error: String,
    #[tabled(rename = "95% CI")]
    ci: String,
    #[tabled(rename = "Intercept")]
    intercept: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "\n═══════════════════════════════════════════════════════════════".green());
    println!("{}", "  Bayesian Pairs Trading: Rolling Regression Analysis".green().bold());
    println!("{}", "═══════════════════════════════════════════════════════════════\n".green());

    // Parse symbols
    let symbol1 = Symbol::from_str(&args.symbol1)
        .ok_or_else(|| anyhow::anyhow!("Unknown symbol: {}", args.symbol1))?;
    let symbol2 = Symbol::from_str(&args.symbol2)
        .ok_or_else(|| anyhow::anyhow!("Unknown symbol: {}", args.symbol2))?;

    println!(
        "Pair: {} (Y) vs {} (X)",
        symbol1.to_string().cyan(),
        symbol2.to_string().yellow()
    );
    println!("Interval: {}, Window: {}\n", args.interval, args.window);

    // Fetch data
    let client = BybitClient::new();

    println!("Fetching data from Bybit...");
    let klines1 = client.get_klines(symbol1, &args.interval, args.limit).await?;
    let klines2 = client.get_klines(symbol2, &args.interval, args.limit).await?;
    println!("Fetched {} and {} candles\n", klines1.len(), klines2.len());

    // Align data (use minimum length)
    let n = klines1.len().min(klines2.len());
    let prices1: Vec<f64> = klines1[..n].iter().map(|k| k.close).collect();
    let prices2: Vec<f64> = klines2[..n].iter().map(|k| k.close).collect();
    let timestamps: Vec<i64> = klines1[..n].iter().map(|k| k.timestamp).collect();

    // Calculate returns for correlation
    let returns1 = Returns::from_klines(&klines1[..n]);
    let returns2 = Returns::from_klines(&klines2[..n]);

    // Correlation analysis
    let corr = correlation(&returns1, &returns2);

    println!("{}", "Correlation Analysis:".yellow().bold());
    println!("  Correlation: {:.4}", corr);

    let corr_interpretation = if corr.abs() > 0.7 {
        "Strong".green()
    } else if corr.abs() > 0.4 {
        "Moderate".yellow()
    } else {
        "Weak".red()
    };
    println!("  Strength: {} correlation", corr_interpretation);

    if corr < 0.3 {
        println!(
            "\n  {} Correlation is low. Pairs trading may not be suitable.",
            "Warning:".red().bold()
        );
    }
    println!();

    // Log prices for regression (common in pairs trading)
    let log_prices1: Vec<f64> = prices1.iter().map(|p| p.ln()).collect();
    let log_prices2: Vec<f64> = prices2.iter().map(|p| p.ln()).collect();

    // Run rolling Bayesian regression
    println!("{}", "Running Rolling Bayesian Regression...".yellow().bold());
    println!("  Model: log({}) = alpha + beta * log({})", symbol1, symbol2);
    println!();

    let mut rolling = RollingBayesianRegression::new(args.window, 0.01);
    rolling.fit(&log_prices2, &log_prices1, &timestamps, true);

    if rolling.results.is_empty() {
        println!("Not enough data for rolling regression with window size {}", args.window);
        return Ok(());
    }

    // Get hedge ratio (slope) time series
    let slopes = rolling.slope_series();
    let intercepts = rolling.intercept_series();

    // Summary statistics
    let hedge_ratios: Vec<f64> = slopes.iter().map(|(_, hr, _)| *hr).collect();
    let hr_mean: f64 = hedge_ratios.iter().sum::<f64>() / hedge_ratios.len() as f64;
    let hr_std: f64 = {
        let var: f64 = hedge_ratios.iter().map(|hr| (hr - hr_mean).powi(2)).sum::<f64>()
            / (hedge_ratios.len() - 1) as f64;
        var.sqrt()
    };
    let hr_min = hedge_ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    let hr_max = hedge_ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("{}", "Hedge Ratio Summary:".yellow().bold());
    println!("{}", "─".repeat(50));
    println!("  Mean hedge ratio:     {:>10.4}", hr_mean);
    println!("  Std deviation:        {:>10.4}", hr_std);
    println!("  Min:                  {:>10.4}", hr_min);
    println!("  Max:                  {:>10.4}", hr_max);
    println!("  Range:                {:>10.4}", hr_max - hr_min);
    println!();

    // Show sample of results
    println!("{}", "Sample Hedge Ratios Over Time:".yellow().bold());

    // Select evenly spaced samples
    let sample_indices: Vec<usize> = if slopes.len() > 10 {
        let step = slopes.len() / 10;
        (0..10).map(|i| i * step).collect()
    } else {
        (0..slopes.len()).collect()
    };

    let mut rows = Vec::new();
    for &idx in &sample_indices {
        if idx < slopes.len() {
            let (ts, hr, std) = &slopes[idx];
            let (_, intercept, _) = &intercepts[idx];

            // Calculate CI
            let ci_low = hr - 1.96 * std;
            let ci_high = hr + 1.96 * std;

            // Format timestamp
            let datetime = chrono::DateTime::from_timestamp_millis(*ts)
                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| "N/A".to_string());

            rows.push(HedgeRatioRow {
                period: datetime,
                hedge_ratio: format!("{:.4}", hr),
                std_error: format!("{:.4}", std),
                ci: format!("[{:.3}, {:.3}]", ci_low, ci_high),
                intercept: format!("{:.4}", intercept),
            });
        }
    }

    let table = Table::new(rows).to_string();
    println!("{}", table);

    // Spread analysis
    println!("\n{}", "Spread Analysis:".yellow().bold());
    println!("{}", "─".repeat(50));

    // Calculate spread using average hedge ratio
    let spread: Vec<f64> = log_prices1
        .iter()
        .zip(log_prices2.iter())
        .map(|(y, x)| y - hr_mean * x)
        .collect();

    let spread_mean: f64 = spread.iter().sum::<f64>() / spread.len() as f64;
    let spread_std: f64 = {
        let var: f64 = spread.iter().map(|s| (s - spread_mean).powi(2)).sum::<f64>()
            / (spread.len() - 1) as f64;
        var.sqrt()
    };

    println!("  Using average hedge ratio: {:.4}", hr_mean);
    println!("  Spread mean:               {:>10.4}", spread_mean);
    println!("  Spread std:                {:>10.4}", spread_std);

    // Current spread position
    let current_spread = spread.last().unwrap_or(&0.0);
    let z_score = (current_spread - spread_mean) / spread_std;

    println!("\n  Current spread:            {:>10.4}", current_spread);
    println!("  Z-score:                   {:>10.4}", z_score);

    // Trading signals
    println!("\n{}", "Trading Signal:".green().bold());
    println!("{}", "─".repeat(50));

    if z_score > 2.0 {
        println!(
            "  {} Spread is significantly ABOVE mean (z = {:.2})",
            "SIGNAL:".red().bold(),
            z_score
        );
        println!(
            "  Potential trade: SHORT {} / LONG {} (ratio: {:.4})",
            symbol1, symbol2, hr_mean
        );
    } else if z_score < -2.0 {
        println!(
            "  {} Spread is significantly BELOW mean (z = {:.2})",
            "SIGNAL:".green().bold(),
            z_score
        );
        println!(
            "  Potential trade: LONG {} / SHORT {} (ratio: {:.4})",
            symbol1, symbol2, hr_mean
        );
    } else if z_score > 1.0 {
        println!(
            "  {} Spread is moderately above mean (z = {:.2})",
            "WATCH:".yellow().bold(),
            z_score
        );
        println!("  Consider short position if spread widens further");
    } else if z_score < -1.0 {
        println!(
            "  {} Spread is moderately below mean (z = {:.2})",
            "WATCH:".yellow().bold(),
            z_score
        );
        println!("  Consider long position if spread narrows further");
    } else {
        println!(
            "  {} Spread is near mean (z = {:.2})",
            "NEUTRAL:".dimmed(),
            z_score
        );
        println!("  No clear trading opportunity");
    }

    // Regime change detection
    println!("\n{}", "Regime Analysis:".yellow().bold());
    println!("{}", "─".repeat(50));

    // Check for significant changes in hedge ratio
    if slopes.len() > 10 {
        let first_quarter: Vec<f64> = slopes[..slopes.len() / 4]
            .iter()
            .map(|(_, hr, _)| *hr)
            .collect();
        let last_quarter: Vec<f64> = slopes[3 * slopes.len() / 4..]
            .iter()
            .map(|(_, hr, _)| *hr)
            .collect();

        let first_mean: f64 = first_quarter.iter().sum::<f64>() / first_quarter.len() as f64;
        let last_mean: f64 = last_quarter.iter().sum::<f64>() / last_quarter.len() as f64;

        let change = (last_mean - first_mean).abs();
        let pct_change = change / first_mean.abs() * 100.0;

        println!("  First quarter avg HR:  {:>10.4}", first_mean);
        println!("  Last quarter avg HR:   {:>10.4}", last_mean);
        println!("  Change:                {:>10.4} ({:.1}%)", last_mean - first_mean, pct_change);

        if pct_change > 10.0 {
            println!(
                "\n  {} Hedge ratio has changed significantly!",
                "Warning:".red().bold()
            );
            println!("  The relationship between assets may be evolving.");
            println!("  Consider using more recent hedge ratio: {:.4}", last_mean);
        } else {
            println!(
                "\n  {} Relationship appears stable.",
                "Good:".green().bold()
            );
        }
    }

    // Position sizing suggestion
    println!("\n{}", "Position Sizing Example:".yellow().bold());
    println!("{}", "─".repeat(50));

    let notional = 10000.0; // Example notional value
    let current_price1 = prices1.last().unwrap_or(&0.0);
    let current_price2 = prices2.last().unwrap_or(&0.0);

    if *current_price1 > 0.0 && *current_price2 > 0.0 {
        let units1 = notional / current_price1;
        let units2 = (notional * hr_mean) / current_price2;

        println!("  For ${:.0} notional per leg:", notional);
        println!(
            "    {} position: {:.6} units @ ${:.2}",
            symbol1, units1, current_price1
        );
        println!(
            "    {} position: {:.6} units @ ${:.2}",
            symbol2, units2, current_price2
        );
        println!("    Hedge ratio: {:.4}", hr_mean);
    }

    println!("\n{}", "═══════════════════════════════════════════════════════════════\n".green());

    Ok(())
}
