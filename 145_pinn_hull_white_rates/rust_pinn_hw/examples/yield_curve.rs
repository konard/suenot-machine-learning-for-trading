//! Yield curve construction and analysis example.
//!
//! Demonstrates:
//! - Building yield curves from Hull-White bond prices
//! - Forward rate curves
//! - Impact of parameters on curve shape
//! - Theta(t) function
//!
//! Run: cargo run --example yield_curve

use pinn_hull_white::{
    analytical::HullWhiteAnalytical,
    data::sample_treasury_curve,
};

fn main() {
    println!("=== Hull-White Yield Curve Analysis ===\n");

    let r0 = 0.03;
    let a = 0.1;
    let sigma = 0.01;

    // ── Flat Curve Yield Curve ───────────────────────────────────

    println!("1. Yield Curve (Flat Initial Curve, r0={:.2}%)\n", r0 * 100.0);

    let hw = HullWhiteAnalytical::new_flat(a, sigma, r0);
    let maturities: Vec<f64> = (1..=60).map(|i| i as f64 * 0.5).collect();
    let yields = hw.yield_curve(r0, 0.0, &maturities);

    println!("{:>10} {:>12}", "Maturity", "Yield (%)");
    println!("{:-<24}", "");
    for (i, &mat) in maturities.iter().enumerate() {
        if mat == (mat as u64) as f64 || mat == 0.5 {
            println!("{:>10.1} {:>12.4}", mat, yields[i] * 100.0);
        }
    }

    // ── Impact of Mean Reversion ─────────────────────────────────

    println!("\n2. Impact of Mean Reversion Speed (a)\n");
    println!("{:>10} {:>12} {:>12} {:>12}", "Maturity", "a=0.01", "a=0.1", "a=0.5");
    println!("{:-<48}", "");

    let a_values = [0.01, 0.1, 0.5];
    let test_mats = vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0];

    for &mat in &test_mats {
        let mut row = format!("{:>10.1}", mat);
        for &a_val in &a_values {
            let hw_a = HullWhiteAnalytical::new_flat(a_val, sigma, r0);
            let price = hw_a.bond_price(r0, 0.0, mat);
            let y = if mat > 0.0 { -price.ln() / mat } else { 0.0 };
            row += &format!(" {:>12.4}", y * 100.0);
        }
        println!("{}", row);
    }

    // ── Impact of Volatility ─────────────────────────────────────

    println!("\n3. Impact of Volatility (sigma)\n");
    println!("{:>10} {:>12} {:>12} {:>12}", "Maturity", "s=0.005", "s=0.01", "s=0.02");
    println!("{:-<48}", "");

    let sigma_values = [0.005, 0.01, 0.02];

    for &mat in &test_mats {
        let mut row = format!("{:>10.1}", mat);
        for &s_val in &sigma_values {
            let hw_s = HullWhiteAnalytical::new_flat(a, s_val, r0);
            let price = hw_s.bond_price(r0, 0.0, mat);
            let y = if mat > 0.0 { -price.ln() / mat } else { 0.0 };
            row += &format!(" {:>12.4}", y * 100.0);
        }
        println!("{}", row);
    }

    // ── Forward Rates ────────────────────────────────────────────

    println!("\n4. Forward Rate Curve\n");
    println!("{:>10} {:>12} {:>12}", "Maturity", "Zero Rate", "Fwd Rate");
    println!("{:-<36}", "");

    for &mat in &test_mats {
        let price = hw.bond_price(r0, 0.0, mat);
        let zero_rate = if mat > 0.0 { -price.ln() / mat } else { 0.0 };

        let dm = 0.01;
        let p1 = hw.bond_price(r0, 0.0, mat);
        let p2 = hw.bond_price(r0, 0.0, mat + dm);
        let fwd = -(p2.ln() - p1.ln()) / dm;

        println!("{:>10.1} {:>12.4} {:>12.4}", mat, zero_rate * 100.0, fwd * 100.0);
    }

    // ── Theta Function ───────────────────────────────────────────

    println!("\n5. Theta(t) Function (Time-Dependent Drift)\n");
    println!("{:>10} {:>12}", "Time", "Theta");
    println!("{:-<24}", "");

    for &t in &[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0] {
        let theta = hw.theta(t);
        println!("{:>10.1} {:>12.6}", t, theta);
    }

    // ── Market Curve Example ─────────────────────────────────────

    println!("\n6. Market Yield Curve (Sample Treasury Data)\n");
    let (market_mats, market_rates) = sample_treasury_curve();

    let hw_market = HullWhiteAnalytical::new_with_curve(
        a, sigma,
        market_mats.clone(),
        market_rates.clone(),
    );

    println!("{:>10} {:>12} {:>12} {:>12}",
             "Maturity", "Market (%)", "Model (%)", "Diff (bps)");
    println!("{:-<48}", "");

    for (&mat, &mkt_rate) in market_mats.iter().zip(market_rates.iter()) {
        let price = hw_market.bond_price(market_rates[0], 0.0, mat);
        let model_rate = if mat > 0.0 { -price.ln() / mat } else { 0.0 };
        let diff_bps = (model_rate - mkt_rate) * 10000.0;
        println!(
            "{:>10.2} {:>12.4} {:>12.4} {:>12.2}",
            mat, mkt_rate * 100.0, model_rate * 100.0, diff_bps
        );
    }

    // ── Duration Profile ─────────────────────────────────────────

    println!("\n7. Duration Profile\n");
    println!("{:>10} {:>12} {:>12} {:>12}", "Maturity", "Price", "Duration", "Convexity");
    println!("{:-<48}", "");

    let eps = 0.0001;
    for &mat in &test_mats {
        let p = hw.bond_price(r0, 0.0, mat);
        let p_up = hw.bond_price(r0 + eps, 0.0, mat);
        let p_down = hw.bond_price(r0 - eps, 0.0, mat);

        let duration = -(p_up - p_down) / (2.0 * eps * p);
        let convexity = (p_up - 2.0 * p + p_down) / (eps * eps * p);

        println!(
            "{:>10.1} {:>12.6} {:>12.4} {:>12.4}",
            mat, p, duration, convexity
        );
    }

    println!("\nDone.");
}
