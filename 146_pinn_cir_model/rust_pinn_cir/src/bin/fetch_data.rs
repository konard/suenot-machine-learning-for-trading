//! Fetch Bybit Funding Rate Data
//!
//! Retrieves funding rate history from Bybit API for CIR model calibration.
//! Bybit perpetual futures have funding every 8 hours.

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use clap::Parser;
use colored::*;
use pinn_cir_model::*;
use serde_json::Value;
use std::thread;
use std::time::Duration as StdDuration;

#[derive(Parser, Debug)]
#[command(name = "Bybit Data Fetcher")]
#[command(about = "Fetch funding rate data from Bybit for CIR calibration")]
struct Args {
    /// Trading symbol
    #[arg(long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of days of history
    #[arg(long, default_value = "90")]
    days: u64,

    /// Market category
    #[arg(long, default_value = "linear")]
    category: String,

    /// Output CSV file
    #[arg(long)]
    output: Option<String>,

    /// Use synthetic data (skip API call)
    #[arg(long)]
    synthetic: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("{}", "=".repeat(70).blue());
    println!("{}", "Bybit Funding Rate Fetcher".bold().blue());
    println!("{}", "=".repeat(70).blue());
    println!("Symbol: {}", args.symbol);
    println!("Days: {}", args.days);

    let rates = if args.synthetic {
        println!("\n{}", "Generating synthetic funding rates...".yellow());
        generate_synthetic_rates(args.days as usize)
    } else {
        println!("\n{}", "Fetching from Bybit API...".yellow());
        match fetch_bybit_rates(&args.symbol, args.days, &args.category).await {
            Ok(rates) => {
                println!("{}", format!("Fetched {} records", rates.len()).green());
                rates
            }
            Err(e) => {
                println!(
                    "{}",
                    format!("API error: {}. Using synthetic data.", e).red()
                );
                generate_synthetic_rates(args.days as usize)
            }
        }
    };

    if rates.is_empty() {
        println!("{}", "No data available!".red());
        return Ok(());
    }

    // Statistics
    println!("\n{}", "Data Statistics:".bold().cyan());
    let n = rates.len();
    let mean = rates.iter().sum::<f64>() / n as f64;
    let min = rates.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    println!("  Records:   {}", n);
    println!("  Mean rate: {:.8} ({:.4}% annualized)", mean, mean * 3.0 * 365.0 * 100.0);
    println!("  Std dev:   {:.8}", std_dev);
    println!("  Min rate:  {:.8}", min);
    println!("  Max rate:  {:.8}", max);

    let n_negative = rates.iter().filter(|&&r| r < 0.0).count();
    println!(
        "  Negative:  {} ({:.1}%)",
        n_negative,
        n_negative as f64 / n as f64 * 100.0
    );

    // Calibrate CIR
    println!("\n{}", "CIR Calibration (OLS):".bold().cyan());
    let abs_rates: Vec<f64> = rates.iter().map(|r| r.abs().max(1e-10)).collect();
    let dt = 8.0 / (24.0 * 365.0); // 8 hours in years

    let calibrated = calibrate_cir_ols(&abs_rates, dt);
    print_params_summary(&calibrated);

    // Yield curve from calibrated parameters
    let yield_maturities = vec![0.25, 0.5, 1.0, 2.0, 5.0];
    print_yield_curve(calibrated.r0, &yield_maturities, &calibrated);

    // Save to CSV if requested
    if let Some(output_path) = &args.output {
        let mut csv_content = String::from("index,rate,rate_annualized\n");
        for (i, &rate) in rates.iter().enumerate() {
            csv_content.push_str(&format!(
                "{},{:.10},{:.10}\n",
                i,
                rate,
                rate * 3.0 * 365.0
            ));
        }
        std::fs::write(output_path, csv_content)?;
        println!(
            "\n{}",
            format!("Data saved to {}", output_path).green()
        );
    }

    println!("\n{}", "Done.".green());
    Ok(())
}

/// Fetch funding rates from Bybit API.
async fn fetch_bybit_rates(
    symbol: &str,
    days: u64,
    category: &str,
) -> Result<Vec<f64>> {
    let base_url = "https://api.bybit.com/v5/market/funding/history";
    let client = reqwest::Client::new();
    let mut all_rates = Vec::new();

    let now = Utc::now();
    let start = now - Duration::days(days as i64);
    let mut cursor_time = now.timestamp_millis();
    let start_time = start.timestamp_millis();

    let limit = 200;

    while cursor_time > start_time {
        let response = client
            .get(base_url)
            .query(&[
                ("category", category),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
                ("endTime", &cursor_time.to_string()),
            ])
            .send()
            .await?;

        let body: Value = response.json().await?;

        let ret_code = body["retCode"].as_i64().unwrap_or(-1);
        if ret_code != 0 {
            let msg = body["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(anyhow::anyhow!("Bybit API error: {}", msg));
        }

        let list = body["result"]["list"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid response format"))?;

        if list.is_empty() {
            break;
        }

        let mut oldest_ts = i64::MAX;

        for record in list {
            let rate_str = record["fundingRate"]
                .as_str()
                .unwrap_or("0");
            let rate: f64 = rate_str.parse().unwrap_or(0.0);

            let ts_str = record["fundingRateTimestamp"]
                .as_str()
                .unwrap_or("0");
            let ts: i64 = ts_str.parse().unwrap_or(0);

            all_rates.push(rate);
            oldest_ts = oldest_ts.min(ts);
        }

        cursor_time = oldest_ts - 1;

        // Rate limiting
        thread::sleep(StdDuration::from_millis(100));
    }

    // Sort by time (oldest first)
    all_rates.reverse();

    Ok(all_rates)
}

/// Generate synthetic funding rates using a CIR-like process.
fn generate_synthetic_rates(days: usize) -> Vec<f64> {
    let n_periods = days * 3; // 3 funding periods per day
    let dt = 8.0 / (24.0 * 365.0);

    let kappa = 50.0;
    let theta = 0.0001; // ~0.01% per 8h
    let sigma = 0.005;

    let mut rates = vec![0.0; n_periods];
    rates[0] = theta;

    let mut rng = rand::thread_rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    for i in 1..n_periods {
        let r = rates[i - 1].max(1e-10);
        let drift = kappa * (theta - r) * dt;
        let dw: f64 = rand_distr::Distribution::sample(&normal, &mut rng) * dt.sqrt();
        let diffusion = sigma * r.sqrt() * dw;
        rates[i] = (r + drift + diffusion).max(-0.001);
    }

    // Add some regime shifts
    use rand::Rng;
    let shift_indices: Vec<usize> = (0..3)
        .map(|_| rng.gen_range(0..n_periods))
        .collect();

    for &shift in &shift_indices {
        let shift_size: f64 = rng.gen_range(-0.0005..0.002);
        for i in shift..n_periods {
            let decay = (-(i - shift) as f64 * 0.01).exp();
            rates[i] += shift_size * decay;
        }
    }

    println!("Generated {} synthetic funding rate records", rates.len());
    rates
}
