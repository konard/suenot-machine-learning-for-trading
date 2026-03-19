//! Trading Example: Conditional VAE with Bybit BTCUSDT Data
//!
//! This example:
//! 1. Fetches BTCUSDT kline data from Bybit
//! 2. Detects market regimes (bull/bear/sideways)
//! 3. Trains a CVAE with regime conditioning
//! 4. Generates scenarios for each regime
//! 5. Compares generated vs real regime statistics

use anyhow::Result;
use conditional_vae_trading::*;
use ndarray::Axis;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> Result<()> {
    println!("=== Conditional VAE Trading Example ===\n");

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
            println!("Could not fetch from Bybit ({}), using synthetic data.\n", e);
            generate_synthetic_prices(500)
        }
    };

    // -----------------------------------------------------------------------
    // 2. Detect regimes
    // -----------------------------------------------------------------------
    let window = 10;
    let regimes = detect_regimes(&prices, window);

    let bull_count = regimes.iter().filter(|r| **r == MarketRegime::Bull).count();
    let bear_count = regimes.iter().filter(|r| **r == MarketRegime::Bear).count();
    let side_count = regimes.iter().filter(|r| **r == MarketRegime::Sideways).count();

    println!("Regime distribution:");
    println!("  Bull:     {} ({:.1}%)", bull_count, 100.0 * bull_count as f64 / regimes.len() as f64);
    println!("  Bear:     {} ({:.1}%)", bear_count, 100.0 * bear_count as f64 / regimes.len() as f64);
    println!("  Sideways: {} ({:.1}%)\n", side_count, 100.0 * side_count as f64 / regimes.len() as f64);

    // -----------------------------------------------------------------------
    // 3. Prepare training data
    // -----------------------------------------------------------------------
    let feature_window = 5;
    let data = prepare_training_data(&prices, feature_window);
    println!("Training samples: {}\n", data.len());

    if data.is_empty() {
        println!("Not enough data to train. Exiting.");
        return Ok(());
    }

    // -----------------------------------------------------------------------
    // 4. Create and train CVAE
    // -----------------------------------------------------------------------
    let mut rng = StdRng::seed_from_u64(42);

    let input_dim = feature_window;
    let cond_raw_dim = MarketRegime::COUNT;
    let cond_encoded_dim = 4;
    let hidden_dim = 16;
    let latent_dim = 3;

    let mut cvae = CVAE::new(
        input_dim,
        cond_raw_dim,
        cond_encoded_dim,
        hidden_dim,
        latent_dim,
        &mut rng,
    );

    println!("Training CVAE (input={}, latent={}, hidden={})...\n",
        input_dim, latent_dim, hidden_dim);

    let epochs = 30;
    let lr = 0.001;
    let loss_history = train_cvae(&mut cvae, &data, epochs, lr, &mut rng);

    println!("\nFinal loss: {:.6}\n", loss_history.last().unwrap_or(&0.0));

    // -----------------------------------------------------------------------
    // 5. Generate scenarios per regime and compare
    // -----------------------------------------------------------------------
    println!("=== Per-Regime Evaluation ===\n");

    let metrics = evaluate_per_regime(&cvae, &data, 200, &mut rng);

    for m in &metrics {
        println!("{}", m);
    }

    // -----------------------------------------------------------------------
    // 6. Show example generated samples per regime
    // -----------------------------------------------------------------------
    println!("\n=== Sample Generated Windows ===\n");

    for regime_idx in 0..MarketRegime::COUNT {
        let regime = MarketRegime::from_index(regime_idx);
        let samples = cvae.generate(regime, 5, &mut rng);

        println!("{:?} regime (5 sample windows):", regime);
        for i in 0..samples.nrows() {
            let row: Vec<String> = samples
                .row(i)
                .iter()
                .map(|v| format!("{:+.4}", v))
                .collect();
            println!("  [{}]", row.join(", "));
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // 7. Compare regime means
    // -----------------------------------------------------------------------
    println!("=== Regime Mean Comparison ===\n");

    for regime_idx in 0..MarketRegime::COUNT {
        let regime = MarketRegime::from_index(regime_idx);
        let samples = cvae.generate(regime, 1000, &mut rng);
        let mean = samples.mean_axis(Axis(0)).unwrap();
        let var = samples.var_axis(Axis(0), 0.0);

        println!("{:?}:", regime);
        println!("  Mean: {:+.6?}", mean.to_vec());
        println!("  Var:  {:+.6?}", var.to_vec());
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
        // Create regime-like behavior
        let drift = if i < n / 3 {
            0.002 // bull
        } else if i < 2 * n / 3 {
            -0.003 // bear (higher vol)
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
