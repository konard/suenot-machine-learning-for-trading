//! Byzantine-Robust Federated Learning for Trading
//!
//! This example fetches BTCUSDT candle data from Bybit, generates synthetic
//! gradient updates based on market features, simulates Byzantine attacks,
//! and compares the robustness of different aggregation methods.
//!
//! Run with: cargo run --example trading_example

use anyhow::Result;
use byzantine_robust_fl::*;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> Result<()> {
    println!("=== Byzantine-Robust Federated Learning for Trading ===\n");

    // ── Fetch market data from Bybit ─────────────────────────────────
    println!("[1] Fetching BTCUSDT 1h candles from Bybit...");

    let true_gradient = match fetch_market_gradient() {
        Ok(grad) => {
            println!("    Successfully fetched market data.");
            println!(
                "    True gradient (from market features): [{:.6}, {:.6}, {:.6}, {:.6}]",
                grad[0], grad[1], grad[2], grad[3]
            );
            grad
        }
        Err(e) => {
            println!("    Warning: Could not fetch from Bybit: {}", e);
            println!("    Using synthetic gradient instead.\n");
            Array1::from_vec(vec![0.002, 1.05, 1.001, 0.015])
        }
    };

    println!();

    // ── Simulation parameters ────────────────────────────────────────
    let num_honest = 8;
    let num_byzantine = 2;
    let total_clients = num_honest + num_byzantine;
    let honest_noise_std = 0.01;
    let num_rounds = 50;

    println!("[2] Simulation setup:");
    println!("    Total clients:    {}", total_clients);
    println!("    Honest clients:   {}", num_honest);
    println!("    Byzantine clients: {}", num_byzantine);
    println!("    Noise std (honest): {}", honest_noise_std);
    println!("    Rounds per test:  {}", num_rounds);
    println!();

    // ── Attack scenarios ─────────────────────────────────────────────
    let attacks = vec![
        ("Random Noise (scale=5.0)", AttackStrategy::RandomNoise { scale: 5.0 }),
        ("Sign Flip (scale=3.0)", AttackStrategy::SignFlip { scale: 3.0 }),
        ("Fixed Direction (scale=10.0)", AttackStrategy::FixedDirection { scale: 10.0 }),
    ];

    let methods = vec![
        AggregationMethod::FedAvg,
        AggregationMethod::Krum,
        AggregationMethod::TrimmedMean { beta: 2 },
        AggregationMethod::CoordinateMedian,
    ];

    // ── Run simulations ──────────────────────────────────────────────
    println!("[3] Running simulations...\n");

    let mut rng = StdRng::seed_from_u64(42);

    for (attack_name, attack) in &attacks {
        println!("  Attack: {}", attack_name);
        println!(
            "  {:<25} {:>15} {:>18}",
            "Method", "Cos Similarity", "Euclidean Dist"
        );
        println!("  {}", "-".repeat(60));

        for &method in &methods {
            let mut total_cos_sim = 0.0;
            let mut total_euc_dist = 0.0;

            for _ in 0..num_rounds {
                let result = simulate_fl_round(
                    &true_gradient,
                    num_honest,
                    num_byzantine,
                    honest_noise_std,
                    *attack,
                    method,
                    &mut rng,
                );
                total_cos_sim += result.cosine_sim_to_true;
                total_euc_dist += result.euclidean_dist_to_true;
            }

            let avg_cos_sim = total_cos_sim / num_rounds as f64;
            let avg_euc_dist = total_euc_dist / num_rounds as f64;

            println!(
                "  {:<25} {:>15.6} {:>18.6}",
                method.to_string(),
                avg_cos_sim,
                avg_euc_dist
            );
        }
        println!();
    }

    // ── Detailed single-round example ────────────────────────────────
    println!("[4] Detailed single-round breakdown:\n");

    let attack = AttackStrategy::SignFlip { scale: 5.0 };

    // Generate all gradients
    let mut all_gradients = Vec::new();
    let mut labels = Vec::new();

    for i in 0..num_honest {
        let g = generate_honest_gradient(&true_gradient, honest_noise_std, &mut rng);
        println!(
            "    Honest client {}: [{:.4}, {:.4}, {:.4}, {:.4}]",
            i, g[0], g[1], g[2], g[3]
        );
        all_gradients.push(g);
        labels.push("honest");
    }

    for i in 0..num_byzantine {
        let g = generate_byzantine_gradient(&true_gradient, attack, &mut rng);
        println!(
            "    Byzantine client {}: [{:.4}, {:.4}, {:.4}, {:.4}]",
            i, g[0], g[1], g[2], g[3]
        );
        all_gradients.push(g);
        labels.push("byzantine");
    }

    println!(
        "\n    True gradient:     [{:.4}, {:.4}, {:.4}, {:.4}]",
        true_gradient[0], true_gradient[1], true_gradient[2], true_gradient[3]
    );

    println!("\n    Aggregation results:");
    for &method in &methods {
        let result = aggregate(&all_gradients, method, num_byzantine);
        let cos_sim = cosine_similarity(&result, &true_gradient);
        let euc_dist = euclidean_distance(&result, &true_gradient);
        println!(
            "    {:<25} -> [{:.4}, {:.4}, {:.4}, {:.4}]  (cos_sim={:.4}, dist={:.4})",
            method.to_string(),
            result[0],
            result[1],
            result[2],
            result[3],
            cos_sim,
            euc_dist
        );
    }

    // ── Summary ──────────────────────────────────────────────────────
    println!("\n[5] Summary:");
    println!("    - FedAvg is heavily influenced by Byzantine gradients");
    println!("    - Krum selects the single most central gradient, ignoring outliers");
    println!("    - TrimmedMean removes extremes per coordinate, using remaining data");
    println!("    - CoordinateMedian picks the middle value, robust to up to 50% Byzantine");
    println!("\n    For trading applications, TrimmedMean generally provides the best");
    println!("    balance between robustness and information utilization.");

    Ok(())
}

/// Fetch BTCUSDT market data from Bybit and compute a synthetic gradient.
fn fetch_market_gradient() -> Result<Array1<f64>> {
    let client = BybitClient::new();
    let candles = client.fetch_klines("BTCUSDT", "60", 200)?;

    if candles.is_empty() {
        anyhow::bail!("No candle data returned");
    }

    println!("    Fetched {} candles", candles.len());
    println!(
        "    Price range: {:.2} - {:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
    );

    let features = compute_features(&candles, 20);
    if features.is_empty() {
        anyhow::bail!("Not enough data to compute features");
    }

    Ok(features_to_gradient(&features))
}
