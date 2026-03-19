//! Multi-asset HiPPO analysis example.
//!
//! Demonstrates using HiPPO to extract features from multiple
//! cryptocurrency pairs and compare their coefficient dynamics.

use hippo_trading::data::bybit::generate_synthetic_data;
use hippo_trading::model::hippo::HiPPOLegS;

fn main() {
    println!("=== HiPPO Framework - Multi-Asset Analysis ===\n");

    let hippo = HiPPOLegS::new(32);

    // Generate synthetic data for multiple assets
    let assets = vec![
        ("BTCUSDT", 50000.0, 0.02),
        ("ETHUSDT", 3000.0, 0.03),
        ("SOLUSDT", 100.0, 0.05),
    ];

    for (symbol, base_price, vol) in &assets {
        let data = generate_synthetic_data(symbol, 500, *base_price, *vol, 42);
        let prices = data.close_prices();

        // Normalize prices
        let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        let std: f64 = {
            let var: f64 = prices.iter().map(|&p| (p - mean).powi(2)).sum::<f64>()
                / prices.len() as f64;
            var.sqrt().max(1e-10)
        };
        let normalized: Vec<f64> = prices.iter().map(|&p| (p - mean) / std).collect();

        // Run HiPPO (use small dt for stability with large N)
        let dt = 1.0 / (hippo.n as f64 + 1.0);
        let history = hippo.process_sequence(&normalized, dt);

        // Analyze last coefficient state
        let last = &history[history.len() - 1];
        let _c0 = last[0];
        let c1 = last[1];
        let c2 = last[2];

        // Compute high-order coefficient energy
        let high_energy: f64 = last[16..].iter().map(|&c| c * c).sum::<f64>();

        println!("{symbol}:");
        println!("  Price range: {:.2} - {:.2}",
            prices.iter().cloned().fold(f64::INFINITY, f64::min),
            prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        println!("  Trend (c1):       {c1:>8.4}");
        println!("  Curvature (c2):   {c2:>8.4}");
        println!("  High-freq energy: {high_energy:>8.4}");
        println!("  Interpretation:   {}", interpret_coefficients(c1, c2, high_energy));
        println!();
    }

    println!("=== Done ===");
}

fn interpret_coefficients(c1: f64, c2: f64, _high_energy: f64) -> &'static str {
    if c1.abs() < 0.1 {
        "Ranging / Sideways market"
    } else if c1 > 0.0 && c2 > 0.0 {
        "Strong uptrend (accelerating)"
    } else if c1 > 0.0 && c2 <= 0.0 {
        "Uptrend (decelerating)"
    } else if c1 < 0.0 && c2 < 0.0 {
        "Strong downtrend (accelerating)"
    } else {
        "Downtrend (decelerating)"
    }
}
