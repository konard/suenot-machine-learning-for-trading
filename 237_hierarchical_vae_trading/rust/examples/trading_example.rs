//! Trading Example: Hierarchical VAE with Bybit BTCUSDT Data
//!
//! This example:
//! 1. Fetches BTCUSDT kline data from Bybit
//! 2. Detects market regimes (bull/bear/sideways)
//! 3. Trains a Hierarchical VAE with multi-scale latent representations
//! 4. Generates scenarios and evaluates per-level latent structure
//! 5. Demonstrates controlled generation with fixed top-level latents

use anyhow::Result;
use hierarchical_vae_trading::*;
use ndarray::{Array1, Axis};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> Result<()> {
    println!("=== Hierarchical VAE Trading Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Fetch data from Bybit
    // -----------------------------------------------------------------------
    println!("Fetching BTCUSDT 1h klines from Bybit...");
    let prices = match fetch_bybit_klines("BTCUSDT", "60", 500) {
        Ok(p) => {
            println!("Fetched {} candles from Bybit.\n", p.len());
            p
        }
        Err(e) => {
            println!(
                "Could not fetch from Bybit ({}), using synthetic data.\n",
                e
            );
            generate_synthetic_prices(500)
        }
    };

    // -----------------------------------------------------------------------
    // 2. Prepare training data and detect regimes
    // -----------------------------------------------------------------------
    let feature_window = 5;
    let (data, regimes) = prepare_training_data(&prices, feature_window);
    println!("Training samples: {}", data.len());

    if data.is_empty() {
        println!("Not enough data to train. Exiting.");
        return Ok(());
    }

    let bull_count = regimes.iter().filter(|r| **r == MarketRegime::Bull).count();
    let bear_count = regimes.iter().filter(|r| **r == MarketRegime::Bear).count();
    let side_count = regimes
        .iter()
        .filter(|r| **r == MarketRegime::Sideways)
        .count();

    println!("Regime distribution:");
    println!(
        "  Bull:     {} ({:.1}%)",
        bull_count,
        100.0 * bull_count as f64 / regimes.len() as f64
    );
    println!(
        "  Bear:     {} ({:.1}%)",
        bear_count,
        100.0 * bear_count as f64 / regimes.len() as f64
    );
    println!(
        "  Sideways: {} ({:.1}%)\n",
        side_count,
        100.0 * side_count as f64 / regimes.len() as f64
    );

    // -----------------------------------------------------------------------
    // 3. Create and train Hierarchical VAE
    // -----------------------------------------------------------------------
    let mut rng = StdRng::seed_from_u64(42);

    let input_dim = feature_window;
    let hidden_dim = 16;
    let latent_dims = [3, 2, 2]; // bottom, middle, top

    let mut hvae = HierarchicalVAE::new(input_dim, hidden_dim, latent_dims, &mut rng);

    println!(
        "Training HVAE (input={}, hidden={}, latent=[{},{},{}])...\n",
        input_dim, hidden_dim, latent_dims[0], latent_dims[1], latent_dims[2]
    );

    let epochs = 30;
    let lr = 0.001;
    let free_bits = 0.1;
    let loss_history = train_hvae(&mut hvae, &data, epochs, lr, free_bits, &mut rng);

    if let Some(last) = loss_history.last() {
        println!(
            "\nFinal: loss={:.4} recon={:.4} KL=[{:.4}, {:.4}, {:.4}]\n",
            last.total, last.recon_loss, last.kl_per_level[0], last.kl_per_level[1], last.kl_per_level[2]
        );
    }

    // -----------------------------------------------------------------------
    // 4. Evaluate generation quality per regime
    // -----------------------------------------------------------------------
    println!("=== Per-Regime Evaluation ===\n");

    let metrics = evaluate_generation(&hvae, &data, &regimes, 500, &mut rng);
    for m in &metrics {
        println!("{}", m);
    }

    // -----------------------------------------------------------------------
    // 5. Analyze top-level latent representations per regime
    // -----------------------------------------------------------------------
    println!("\n=== Top-Level Latent Analysis ===\n");

    for regime_idx in 0..MarketRegime::COUNT {
        let regime = MarketRegime::from_index(regime_idx);

        let regime_data: Vec<&Array1<f64>> = data
            .iter()
            .zip(regimes.iter())
            .filter(|(_, r)| **r == regime)
            .map(|(x, _)| x)
            .collect();

        if regime_data.is_empty() {
            continue;
        }

        let top_latents: Vec<Array1<f64>> = regime_data
            .iter()
            .map(|x| hvae.encode_top_level(x))
            .collect();

        let n = top_latents.len() as f64;
        let mean: Vec<f64> = (0..latent_dims[2])
            .map(|d| top_latents.iter().map(|z| z[d]).sum::<f64>() / n)
            .collect();
        let std: Vec<f64> = (0..latent_dims[2])
            .map(|d| {
                let m = mean[d];
                (top_latents.iter().map(|z| (z[d] - m).powi(2)).sum::<f64>() / n).sqrt()
            })
            .collect();

        println!(
            "{} regime (n={}): mean={:+.4?} std={:.4?}",
            regime.name(),
            regime_data.len(),
            mean,
            std
        );
    }

    // -----------------------------------------------------------------------
    // 6. Controlled generation with fixed top-level latents
    // -----------------------------------------------------------------------
    println!("\n=== Controlled Generation (Fixed Top Latent) ===\n");

    let z_top_positive = Array1::from(vec![1.5, 1.5]);
    let z_top_negative = Array1::from(vec![-1.5, -1.5]);

    let samples_pos = hvae.generate_with_fixed_top(&z_top_positive, 500, &mut rng);
    let samples_neg = hvae.generate_with_fixed_top(&z_top_negative, 500, &mut rng);

    let mean_pos = samples_pos.mean_axis(Axis(0)).unwrap();
    let mean_neg = samples_neg.mean_axis(Axis(0)).unwrap();
    let var_pos = samples_pos.var_axis(Axis(0), 0.0);
    let var_neg = samples_neg.var_axis(Axis(0), 0.0);

    println!("z_top = [+1.5, +1.5]:");
    println!("  Mean: {:+.4?}", mean_pos.to_vec());
    println!("  Var:  {:.4?}", var_pos.to_vec());

    println!("z_top = [-1.5, -1.5]:");
    println!("  Mean: {:+.4?}", mean_neg.to_vec());
    println!("  Var:  {:.4?}", var_neg.to_vec());

    // -----------------------------------------------------------------------
    // 7. Show example generated windows
    // -----------------------------------------------------------------------
    println!("\n=== Sample Generated Windows ===\n");

    let uncond_samples = hvae.generate(5, &mut rng);
    println!("Unconditional (5 sample windows):");
    for i in 0..uncond_samples.nrows() {
        let row: Vec<String> = uncond_samples
            .row(i)
            .iter()
            .map(|v| format!("{:+.4}", v))
            .collect();
        println!("  [{}]", row.join(", "));
    }

    println!("\nDone!");
    Ok(())
}

/// Generate synthetic price data for offline testing.
fn generate_synthetic_prices(n: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(123);
    let mut prices = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    for i in 0..n {
        let drift = if i < n / 3 {
            0.002 // bull
        } else if i < 2 * n / 3 {
            -0.003 // bear
        } else {
            0.0001 // sideways
        };

        let vol = if i < n / 3 {
            0.01
        } else if i < 2 * n / 3 {
            0.025
        } else {
            0.008
        };

        let r = drift + vol * rng.gen_range(-1.0..1.0);
        price *= 1.0 + r;
        prices.push(price);
    }

    prices
}
