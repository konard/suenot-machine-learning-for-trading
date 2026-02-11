//! Basic Hull-White bond pricing example.
//!
//! Demonstrates:
//! - Analytical zero-coupon bond pricing
//! - PINN-based pricing
//! - Bond option pricing
//! - Rate simulation
//!
//! Run: cargo run --example basic_pricing

use pinn_hull_white::{
    analytical::HullWhiteAnalytical,
    model::{HullWhitePINN, PINNConfig},
};

fn main() {
    println!("=== Hull-White Bond Pricing Example ===\n");

    // Parameters
    let a = 0.1;       // Mean reversion speed
    let sigma = 0.01;  // Volatility
    let r0 = 0.03;     // Current short rate

    // Create analytical model
    let hw = HullWhiteAnalytical::new_flat(a, sigma, r0);

    // ── Zero-Coupon Bond Prices ──────────────────────────────────

    println!("Zero-Coupon Bond Prices (r0 = {:.2}%):", r0 * 100.0);
    println!("{:>10} {:>12} {:>12}", "Maturity", "Price", "Yield (%)");
    println!("{:-<36}", "");

    for &t in &[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0] {
        let price = hw.bond_price(r0, 0.0, t);
        let yld = hw.yield_from_price(price, 0.0, t);
        println!("{:>10.2} {:>12.6} {:>12.4}", t, price, yld * 100.0);
    }

    // ── Bond Price Sensitivity ───────────────────────────────────

    println!("\nBond Price Sensitivity to Rate (T=5Y):");
    println!("{:>10} {:>12}", "Rate (%)", "Price");
    println!("{:-<24}", "");

    for r_bps in (-200..=200).step_by(50) {
        let r = r0 + r_bps as f64 / 10000.0;
        let price = hw.bond_price(r, 0.0, 5.0);
        println!("{:>10.2} {:>12.6}", r * 100.0, price);
    }

    // ── Bond Options ─────────────────────────────────────────────

    println!("\nBond Options (expiry=1Y, bond=5Y):");
    println!("{:>10} {:>12} {:>12}", "Strike", "Call", "Put");
    println!("{:-<36}", "");

    for k in [0.85, 0.88, 0.90, 0.92, 0.95] {
        let call = hw.bond_option(r0, 0.0, 1.0, 5.0, k, true);
        let put = hw.bond_option(r0, 0.0, 1.0, 5.0, k, false);
        println!("{:>10.3} {:>12.6} {:>12.6}", k, call, put);
    }

    // ── Rate Distribution ────────────────────────────────────────

    println!("\nRate Distribution:");
    for &t in &[0.5, 1.0, 2.0, 5.0] {
        let mean = hw.rate_mean(r0, 0.0, t);
        let std = hw.rate_variance(0.0, t).sqrt();
        println!(
            "  At T={:.1}: E[r] = {:.4}%, Std[r] = {:.4}%",
            t, mean * 100.0, std * 100.0
        );
    }

    // ── Rate Simulation ──────────────────────────────────────────

    println!("\nSimulated Rate Paths (5 paths, T=5Y):");
    let (times, paths) = hw.simulate_paths(r0, 5.0, 1260, 5, 42);

    for (i, path) in paths.iter().enumerate() {
        let final_r = path.last().unwrap_or(&0.0);
        let min_r = path.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_r = path.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  Path {}: r(5)={:.4}%, min={:.4}%, max={:.4}%",
            i, final_r * 100.0, min_r * 100.0, max_r * 100.0
        );
    }

    // ── PINN Forward Pass ────────────────────────────────────────

    println!("\nPINN Forward Pass (untrained, random weights):");
    let config = PINNConfig {
        hidden_layers: vec![32, 32, 16],
        activation: "tanh".to_string(),
        learning_rate: 1e-3,
    };
    let pinn = HullWhitePINN::new(a, sigma, config);

    for &t in &[1.0, 5.0, 10.0] {
        let p = pinn.predict(r0, 0.0, t);
        let p_anal = hw.bond_price(r0, 0.0, t);
        println!(
            "  T={:.1}: PINN={:.6}, Analytical={:.6}, Diff={:.4}",
            t, p, p_anal, (p - p_anal).abs()
        );
    }

    println!("\n(Note: PINN prices are from untrained network. Run 'train' binary to fit.)");
    println!("\nDone.");
}
