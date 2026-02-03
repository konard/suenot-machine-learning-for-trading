//! Basic HiPPO usage example.
//!
//! Demonstrates HiPPO-LegS and HiPPO-LagT on a synthetic signal,
//! showing how polynomial coefficients track signal dynamics.

use hippo_trading::model::hippo::{HiPPOLagT, HiPPOLegS};

fn main() {
    println!("=== HiPPO Framework - Basic Example ===\n");

    // Generate a synthetic signal: sine wave with trend
    let n_points = 200;
    let signal: Vec<f64> = (0..n_points)
        .map(|i| {
            let t = i as f64 * 0.05;
            t.sin() + 0.01 * t // sine + upward trend
        })
        .collect();

    // --- HiPPO-LegS ---
    println!("HiPPO-LegS (Scaled Legendre, N=16):");
    let hippo_legs = HiPPOLegS::new(16);
    let history_legs = hippo_legs.process_sequence(&signal, 0.05);

    println!("  Signal length: {}", signal.len());
    println!("  State dimension: {}", hippo_legs.n);
    println!(
        "  Coefficients at t=50:  c0={:.4}, c1={:.4}, c2={:.4}",
        history_legs[50][0], history_legs[50][1], history_legs[50][2]
    );
    println!(
        "  Coefficients at t=100: c0={:.4}, c1={:.4}, c2={:.4}",
        history_legs[100][0], history_legs[100][1], history_legs[100][2]
    );
    println!(
        "  Coefficients at t=199: c0={:.4}, c1={:.4}, c2={:.4}",
        history_legs[199][0], history_legs[199][1], history_legs[199][2]
    );

    // --- HiPPO-LagT ---
    println!("\nHiPPO-LagT (Translated Laguerre, N=16):");
    let hippo_lagt = HiPPOLagT::new(16);
    let history_lagt = hippo_lagt.process_sequence(&signal, 0.05);

    println!(
        "  Coefficients at t=100: c0={:.4}, c1={:.4}, c2={:.4}",
        history_lagt[100][0], history_lagt[100][1], history_lagt[100][2]
    );

    // --- Bilinear discretization ---
    println!("\nHiPPO-LegS with bilinear discretization:");
    match hippo_legs.process_sequence_bilinear(&signal, 0.05) {
        Ok(history_bl) => {
            println!(
                "  Coefficients at t=100: c0={:.4}, c1={:.4}, c2={:.4}",
                history_bl[100][0], history_bl[100][1], history_bl[100][2]
            );
        }
        Err(e) => println!("  Error: {}", e),
    }

    // --- Demonstrate multi-scale decomposition ---
    println!("\nMulti-scale decomposition at t=150:");
    let coeffs = &history_legs[150];
    println!("  Low-order  (c0, c1):       [{:.4}, {:.4}] — long-term trend", coeffs[0], coeffs[1]);
    println!("  Mid-order  (c2, c3):       [{:.4}, {:.4}] — medium-term dynamics", coeffs[2], coeffs[3]);
    println!(
        "  High-order (c14, c15):     [{:.4}, {:.4}] — short-term oscillations",
        coeffs[14], coeffs[15]
    );

    println!("\n=== Done ===");
}
