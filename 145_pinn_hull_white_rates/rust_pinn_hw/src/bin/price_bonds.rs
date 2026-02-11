//! Price zero-coupon bonds and derivatives using the Hull-White model.
//!
//! Usage:
//!     cargo run --bin price_bonds -- --rate 0.03 --maturity 5.0

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use tabled::{Table, Tabled};

use pinn_hull_white::{
    analytical::HullWhiteAnalytical,
    derivatives::DerivativesPricer,
    data::sample_treasury_curve,
};

#[derive(Parser, Debug)]
#[command(name = "price_bonds", about = "Price bonds and derivatives under Hull-White")]
struct Args {
    /// Current short rate
    #[arg(short, long, default_value = "0.05")]
    rate: f64,

    /// Mean reversion speed
    #[arg(short, long, default_value = "0.1")]
    a: f64,

    /// Short rate volatility
    #[arg(short, long, default_value = "0.01")]
    sigma: f64,

    /// Specific maturity to price (optional, prices all standard maturities if not set)
    #[arg(short, long)]
    maturity: Option<f64>,

    /// Cap/floor strike rate
    #[arg(short, long, default_value = "0.04")]
    strike: f64,

    /// Notional amount
    #[arg(short, long, default_value = "1000000")]
    notional: f64,

    /// Number of simulation paths
    #[arg(long, default_value = "1000")]
    paths: usize,

    /// Use market curve (otherwise flat)
    #[arg(long)]
    market_curve: bool,
}

#[derive(Tabled)]
struct BondRow {
    #[tabled(rename = "Maturity")]
    maturity: String,
    #[tabled(rename = "Price")]
    price: String,
    #[tabled(rename = "Yield (%)")]
    yield_pct: String,
    #[tabled(rename = "Duration")]
    duration: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=" .repeat(60).blue());
    println!("{}", "  Hull-White Bond & Derivatives Pricer".bold().blue());
    println!("{}", "=" .repeat(60).blue());

    // Create the model
    let hw = if args.market_curve {
        let (mats, rates) = sample_treasury_curve();
        println!("\n{}", "Using market yield curve:".bold());
        for (&m, &r) in mats.iter().zip(rates.iter()) {
            println!("  {:6.2}Y: {:.2}%", m, r * 100.0);
        }
        HullWhiteAnalytical::new_with_curve(args.a, args.sigma, mats, rates)
    } else {
        println!("\n{}", format!("Using flat curve at {:.2}%", args.rate * 100.0).bold());
        HullWhiteAnalytical::new_flat(args.a, args.sigma, args.rate)
    };

    println!("\n{}", "Parameters:".bold());
    println!("  a = {:.4}, sigma = {:.6}, r0 = {:.4}", args.a, args.sigma, args.rate);

    // ═══════════════════════════════════════════════════════════════
    // Bond Pricing
    // ═══════════════════════════════════════════════════════════════

    println!("\n{}", "Zero-Coupon Bond Prices".bold().green());

    let maturities = if let Some(m) = args.maturity {
        vec![m]
    } else {
        vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    };

    let mut rows = Vec::new();
    for &mat in &maturities {
        let price = hw.bond_price(args.rate, 0.0, mat);
        let yld = hw.yield_from_price(price, 0.0, mat);

        // Approximate duration using finite differences
        let eps = 0.0001;
        let p_up = hw.bond_price(args.rate + eps, 0.0, mat);
        let p_down = hw.bond_price(args.rate - eps, 0.0, mat);
        let duration = -(p_up - p_down) / (2.0 * eps * price);

        rows.push(BondRow {
            maturity: format!("{:.2}Y", mat),
            price: format!("{:.6}", price),
            yield_pct: format!("{:.4}", yld * 100.0),
            duration: format!("{:.4}", duration),
        });
    }

    let table = Table::new(&rows).to_string();
    println!("{}", table);

    // ═══════════════════════════════════════════════════════════════
    // Interest Rate Derivatives
    // ═══════════════════════════════════════════════════════════════

    let pricer = DerivativesPricer::new(hw.clone());
    let notional = args.notional;
    let strike = args.strike;

    println!("\n{}", "Interest Rate Derivatives".bold().green());
    println!("  Notional: {:,.0}", notional);
    println!("  Strike:   {:.2}%", strike * 100.0);

    // Caps
    println!("\n  {}", "Caps:".bold());
    let reset_dates: Vec<f64> = (1..=20).map(|i| i as f64 * 0.25).collect();
    for &cap_mat in &[1.0, 2.0, 3.0, 5.0] {
        let dates: Vec<f64> = reset_dates.iter()
            .filter(|&&d| d <= cap_mat)
            .cloned()
            .collect();
        if dates.len() > 1 {
            let (cap_price, _) = pricer.cap(args.rate, 0.0, &dates, strike, notional);
            println!("    {}Y Cap at {:.2}%: {:>12,.2}", cap_mat, strike * 100.0, cap_price);
        }
    }

    // Floors
    println!("\n  {}", "Floors:".bold());
    for &floor_strike in &[0.02, 0.03, 0.04] {
        let dates: Vec<f64> = reset_dates.iter()
            .filter(|&&d| d <= 5.0)
            .cloned()
            .collect();
        if dates.len() > 1 {
            let (floor_price, _) = pricer.floor(args.rate, 0.0, &dates, floor_strike, notional);
            println!(
                "    5Y Floor at {:.2}%: {:>12,.2}",
                floor_strike * 100.0, floor_price
            );
        }
    }

    // Collars
    println!("\n  {}", "Collars:".bold());
    let collar_dates: Vec<f64> = (1..=20).map(|i| i as f64 * 0.25).collect();
    let collar = pricer.collar(args.rate, 0.0, &collar_dates, 0.06, 0.02, notional);
    println!("    5Y Collar (cap=6%, floor=2%): {:>12,.2}", collar);

    // Bond Options
    println!("\n  {}", "Bond Options (expiry=1Y, bond=5Y):".bold());
    for &k in &[0.80, 0.85, 0.90, 0.95] {
        let call = hw.bond_option(args.rate, 0.0, 1.0, 5.0, k, true);
        let put = hw.bond_option(args.rate, 0.0, 1.0, 5.0, k, false);
        println!(
            "    K={:.2}  Call={:>10.6}  Put={:>10.6}",
            k, call * notional, put * notional
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Rate Simulation
    // ═══════════════════════════════════════════════════════════════

    println!("\n{}", "Rate Path Simulation".bold().green());
    let (times, paths) = hw.simulate_paths(
        args.rate, 5.0, 1260, 5, 42,
    );

    println!("  {} paths, {} steps, T=5Y", 5, 1260);
    for (i, path) in paths.iter().enumerate() {
        let final_rate = path.last().unwrap_or(&0.0);
        let min_rate = path.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_rate = path.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  Path {}: r(5) = {:.4}%  [min={:.4}%, max={:.4}%]",
            i, final_rate * 100.0, min_rate * 100.0, max_rate * 100.0
        );
    }

    // Rate distribution statistics
    println!("\n  {}", "Rate Distribution at T=1:".bold());
    let mean_r = hw.rate_mean(args.rate, 0.0, 1.0);
    let var_r = hw.rate_variance(0.0, 1.0);
    println!("    E[r(1)]   = {:.4}%", mean_r * 100.0);
    println!("    Std[r(1)] = {:.4}%", var_r.sqrt() * 100.0);

    println!("\n{}", "Done.".bold().green());

    Ok(())
}
