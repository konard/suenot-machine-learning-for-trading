//! Fetch Bybit funding rate data for Hull-White calibration.
//!
//! Usage:
//!     cargo run --bin fetch_data -- --symbol BTCUSDT --limit 200

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use tabled::{Table, Tabled};

use pinn_hull_white::{
    data::{BybitClient, generate_synthetic_funding_rates, sample_treasury_curve},
    calibration::{calibrate_from_rates, calibrate_to_curve},
};

#[derive(Parser, Debug)]
#[command(name = "fetch_data", about = "Fetch funding rate data from Bybit")]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of records to fetch
    #[arg(short, long, default_value = "200")]
    limit: usize,

    /// Use synthetic data instead of live API
    #[arg(long)]
    synthetic: bool,

    /// Calibrate Hull-White parameters
    #[arg(short, long)]
    calibrate: bool,
}

#[derive(Tabled)]
struct FundingRow {
    #[tabled(rename = "DateTime")]
    datetime: String,
    #[tabled(rename = "Rate")]
    rate: String,
    #[tabled(rename = "Ann. Rate (%)")]
    annualized: String,
}

#[derive(Tabled)]
struct TreasuryRow {
    #[tabled(rename = "Maturity")]
    maturity: String,
    #[tabled(rename = "Rate (%)")]
    rate: String,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("{}", "=" .repeat(60).blue());
    println!("{}", "  Hull-White Data Fetcher".bold().blue());
    println!("{}", "=" .repeat(60).blue());

    // ═══════════════════════════════════════════════════════════════
    // Treasury Yield Curve
    // ═══════════════════════════════════════════════════════════════

    println!("\n{}", "US Treasury Yield Curve (Sample):".bold().green());
    let (mats, rates) = sample_treasury_curve();

    let treasury_rows: Vec<TreasuryRow> = mats.iter().zip(rates.iter()).map(|(&m, &r)| {
        TreasuryRow {
            maturity: format!("{:.2}Y", m),
            rate: format!("{:.2}", r * 100.0),
        }
    }).collect();

    let table = Table::new(&treasury_rows).to_string();
    println!("{}", table);

    // ═══════════════════════════════════════════════════════════════
    // Bybit Funding Rates
    // ═══════════════════════════════════════════════════════════════

    println!("\n{}", format!("Bybit {} Funding Rates:", args.symbol).bold().green());

    let funding_rates = if args.synthetic {
        println!("  (Using synthetic data)");
        generate_synthetic_funding_rates(args.limit, 0.0001, 0.3, 0.0003, 42)
    } else {
        println!("  Fetching from Bybit API...");
        let client = BybitClient::new();
        match client.fetch_funding_rates(&args.symbol, args.limit) {
            Ok(rates) => {
                println!("  Fetched {} records", rates.len());
                rates
            }
            Err(e) => {
                println!("  {} Using synthetic data.", format!("API error: {}.", e).yellow());
                generate_synthetic_funding_rates(args.limit, 0.0001, 0.3, 0.0003, 42)
            }
        }
    };

    // Show last 10 records
    let display_count = funding_rates.len().min(10);
    let start = funding_rates.len().saturating_sub(display_count);
    let recent: Vec<FundingRow> = funding_rates[start..].iter().map(|r| {
        FundingRow {
            datetime: r.datetime[..19].to_string(),
            rate: format!("{:.6}", r.funding_rate),
            annualized: format!("{:.2}", r.annualized_rate * 100.0),
        }
    }).collect();

    println!("\n  Last {} records:", display_count);
    let table = Table::new(&recent).to_string();
    println!("{}", table);

    // Statistics
    let fr_values: Vec<f64> = funding_rates.iter().map(|r| r.funding_rate).collect();
    let mean_rate: f64 = fr_values.iter().sum::<f64>() / fr_values.len() as f64;
    let std_rate: f64 = {
        let var: f64 = fr_values.iter().map(|&x| (x - mean_rate).powi(2)).sum::<f64>()
            / fr_values.len() as f64;
        var.sqrt()
    };
    let min_rate = fr_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_rate = fr_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n  {}", "Statistics:".bold());
    println!("    Observations:  {}", fr_values.len());
    println!("    Mean rate:     {:.6} ({:.2}% ann.)", mean_rate, mean_rate * 3.0 * 365.0 * 100.0);
    println!("    Std rate:      {:.6}", std_rate);
    println!("    Min rate:      {:.6}", min_rate);
    println!("    Max rate:      {:.6}", max_rate);

    // Autocorrelation
    if fr_values.len() > 1 {
        let n = fr_values.len() - 1;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        for i in 0..n {
            let x = fr_values[i] - mean_rate;
            let y = fr_values[i + 1] - mean_rate;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
        let autocorr = if sum_x2 * sum_y2 > 0.0 {
            sum_xy / (sum_x2 * sum_y2).sqrt()
        } else {
            0.0
        };
        println!("    Autocorr(1):   {:.4}", autocorr);
    }

    // ═══════════════════════════════════════════════════════════════
    // Calibration
    // ═══════════════════════════════════════════════════════════════

    if args.calibrate {
        println!("\n{}", "Hull-White Calibration:".bold().green());

        // Calibrate from funding rates
        println!("\n  {}", "From Funding Rates (OU-MLE):".bold());
        let cal = calibrate_from_rates(&fr_values, 8.0);
        println!("    Mean reversion (a): {:.4}", cal.a);
        println!("    Volatility (sigma): {:.8}", cal.sigma);
        println!("    Long-run mean:      {:.6}", cal.long_run_mean);
        println!("    Half-life:          {:.1}h ({:.1}d)", cal.half_life_hours, cal.half_life_days);

        // Calibrate from yield curve
        println!("\n  {}", "From Yield Curve (Grid Search):".bold());
        let cal_yc = calibrate_to_curve(&mats, &rates);
        println!("    Mean reversion (a): {:.4}", cal_yc.a);
        println!("    Volatility (sigma): {:.6}", cal_yc.sigma);
        println!("    Half-life:          {:.1}d", cal_yc.half_life_days);
    }

    println!("\n{}", "Done.".bold().green());

    Ok(())
}
