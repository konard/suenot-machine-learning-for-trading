//! Volatility Surface Operator Example
//!
//! Demonstrates using operator learning for implied volatility surface dynamics.
//! The operator maps today's vol surface to a predicted future vol surface.
//!
//! Uses synthetic data based on SABR-like parametric vol surfaces with
//! random perturbations to simulate surface evolution.
//!
//! # Usage
//! ```bash
//! cargo run --example vol_surface
//! ```

use colored::Colorize;
use ndarray::{Array1, Array2};
use operator_learning_trading::operator::{DeepONet, DeepONetConfig, black_scholes_call};
use rand::Rng;

fn main() {
    println!("{}", "═══════════════════════════════════════════════════════".blue());
    println!("{}", "  Volatility Surface Operator Example".blue().bold());
    println!("{}", "═══════════════════════════════════════════════════════".blue());

    let n_strikes = 10;
    let n_maturities = 6;
    let n_samples = 500;

    let strikes: Vec<f64> = (0..n_strikes)
        .map(|i| 0.8 + 0.4 * i as f64 / (n_strikes - 1) as f64) // 0.8 to 1.2
        .collect();
    let maturities: Vec<f64> = vec![0.083, 0.25, 0.5, 1.0, 2.0, 5.0]; // 1M to 5Y

    // Generate synthetic vol surface data
    println!("\n{} synthetic vol surface data...", "Generating".yellow());

    let (surfaces_today, surfaces_future) =
        generate_vol_surface_data(n_samples, &strikes, &maturities);

    println!("  Surfaces today:  ({}, {})", surfaces_today.nrows(), surfaces_today.ncols());
    println!("  Surfaces future: ({}, {})", surfaces_future.nrows(), surfaces_future.ncols());

    // Show sample surface
    let surface_dim = n_strikes * n_maturities;
    println!("\n  Sample vol surface (today, in %):");
    print!("  {:>8}", "K\\T");
    for &t in &maturities {
        print!("  {:>6.2}Y", t);
    }
    println!();

    for i in 0..n_strikes {
        print!("  {:>8.2}", strikes[i]);
        for j in 0..n_maturities {
            let idx = i * n_maturities + j;
            print!("  {:>7.1}", surfaces_today[[0, idx]] * 100.0);
        }
        println!();
    }

    // Prepare DeepONet data
    // Branch: flattened today's vol surface
    // Trunk: (strike, maturity) pair
    // Target: future vol at that (strike, maturity)
    println!("\n{} DeepONet training data...", "Preparing".yellow());

    let total = n_samples * surface_dim;
    let mut branch_data = Array2::zeros((total, surface_dim));
    let mut trunk_data = Array2::zeros((total, 2));
    let mut targets = Array1::zeros(total);

    let mut idx = 0;
    for s in 0..n_samples {
        for i in 0..n_strikes {
            for j in 0..n_maturities {
                for k in 0..surface_dim {
                    branch_data[[idx, k]] = surfaces_today[[s, k]];
                }
                trunk_data[[idx, 0]] = (strikes[i] - 0.8) / 0.4; // normalized to [0, 1]
                trunk_data[[idx, 1]] = maturities[j] / 5.0;       // normalized to [0, 1]
                targets[idx] = surfaces_future[[s, i * n_maturities + j]];
                idx += 1;
            }
        }
    }

    let train_end = (total as f64 * 0.8) as usize;

    let branch_train = branch_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let trunk_train = trunk_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let targets_train = targets.slice(ndarray::s![..train_end]).to_owned();

    let branch_test = branch_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let trunk_test = trunk_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let targets_test = targets.slice(ndarray::s![train_end..]).to_owned();

    // Train
    let config = DeepONetConfig {
        branch_input_dim: surface_dim,
        trunk_input_dim: 2,
        latent_dim: 64,
        branch_hidden_dims: vec![128, 128],
        trunk_hidden_dims: vec![64, 64],
        learning_rate: 0.002,
        n_epochs: 60,
        batch_size: 128,
    };

    let mut model = DeepONet::new(config);
    println!("\n{} DeepONet for vol surface operator...", "Training".yellow());
    let losses = model.train(&branch_train, &trunk_train, &targets_train);

    let test_error = model.evaluate(&branch_test, &trunk_test, &targets_test);
    println!("\n  Test Relative L2 Error: {}", format!("{:.6}", test_error).cyan());

    // Show predicted vs actual future surface
    println!("\n{}", "Predicted Future Vol Surface:".yellow().bold());
    let test_sample = train_end;

    print!("  {:>8}", "K\\T");
    for &t in &maturities {
        print!("  {:>6.2}Y", t);
    }
    println!("   (Pred / Actual)");

    for i in 0..n_strikes {
        print!("  {:>8.2}", strikes[i]);
        for j in 0..n_maturities {
            let data_idx = test_sample + i * n_maturities + j;
            if data_idx >= total { continue; }

            let branch_in = branch_data.row(data_idx).to_owned();
            let trunk_in = trunk_data.row(data_idx).to_owned();
            let pred = model.predict(&branch_in, &trunk_in) * 100.0;
            print!("  {:>7.1}", pred);
        }
        println!();
    }

    // Option pricing using predicted vol surface
    println!("\n{}", "Option Pricing with Predicted Vol Surface:".yellow().bold());
    println!("  Using operator-predicted vol to price options via Black-Scholes:\n");
    println!("  {:>8}  {:>6}  {:>10}  {:>10}", "Strike", "T", "PredVol(%)", "CallPrice");

    let sample_strikes = vec![0.9, 0.95, 1.0, 1.05, 1.1];
    let sample_mats = vec![0.25, 1.0, 2.0];

    let surface_input = branch_data.row(test_sample).to_owned();

    for &k in &sample_strikes {
        for &t in &sample_mats {
            let trunk_in = Array1::from_vec(vec![
                (k - 0.8) / 0.4,
                t / 5.0,
            ]);
            let pred_vol = model.predict(&surface_input, &trunk_in);
            let call_price = black_scholes_call(k, t, pred_vol.abs().max(0.01), 0.03);

            println!(
                "  {:>8.2}  {:>6.2}  {:>10.2}  {:>10.4}",
                k, t, pred_vol * 100.0, call_price
            );
        }
    }

    // Zero-shot to novel strikes
    println!("\n{}", "Zero-Shot to Novel Strikes:".yellow().bold());
    println!("  Predicting vol at strikes NOT in training grid:\n");

    let novel_strikes = vec![0.75, 0.85, 1.15, 1.25, 1.30];
    let fixed_mat = 1.0;

    for &k in &novel_strikes {
        let trunk_in = Array1::from_vec(vec![
            (k - 0.8) / 0.4,
            fixed_mat / 5.0,
        ]);
        let pred_vol = model.predict(&surface_input, &trunk_in);
        println!(
            "  K={:.2}, T={:.1}Y: Predicted IV = {:.1}%",
            k, fixed_mat, pred_vol * 100.0
        );
    }

    println!("\n{}", "Key takeaway:".green().bold());
    println!("  The vol surface operator can predict implied volatility at any");
    println!("  (strike, maturity) pair, including combinations never seen in training.");
    println!("  This enables real-time vol surface interpolation and extrapolation.");

    println!("\n{}", "Done!".green().bold());
}

/// Generate synthetic vol surface evolution data.
///
/// Uses SABR-like parameterization:
///   sigma(K, T) = atm_vol + skew * (K - 1) + smile * (K - 1)^2 + term * sqrt(T)
fn generate_vol_surface_data(
    n_samples: usize,
    strikes: &[f64],
    maturities: &[f64],
) -> (Array2<f64>, Array2<f64>) {
    let mut rng = rand::thread_rng();
    let surface_dim = strikes.len() * maturities.len();

    let mut surfaces_today = Array2::zeros((n_samples, surface_dim));
    let mut surfaces_future = Array2::zeros((n_samples, surface_dim));

    for s in 0..n_samples {
        // Today's parameters
        let atm_vol = rng.gen_range(0.15..0.40);
        let skew = rng.gen_range(-0.15..0.0);
        let smile = rng.gen_range(0.05..0.20);
        let term_struct = rng.gen_range(-0.05..0.05);

        // Future's parameters (small perturbation)
        let d_atm = rng.gen_range(-0.02..0.02);
        let d_skew = rng.gen_range(-0.02..0.02);
        let d_smile = rng.gen_range(-0.01..0.01);
        let d_term = rng.gen_range(-0.01..0.01);

        let mut idx = 0;
        for (i, &k) in strikes.iter().enumerate() {
            for (j, &t) in maturities.iter().enumerate() {
                let moneyness = k - 1.0;

                let vol_today = atm_vol
                    + skew * moneyness
                    + smile * moneyness.powi(2)
                    + term_struct * t.sqrt();
                let vol_today = vol_today.clamp(0.05, 1.0);

                let vol_future = (atm_vol + d_atm)
                    + (skew + d_skew) * moneyness
                    + (smile + d_smile) * moneyness.powi(2)
                    + (term_struct + d_term) * t.sqrt()
                    + rng.gen_range(-0.005..0.005); // noise
                let vol_future = vol_future.clamp(0.05, 1.0);

                surfaces_today[[s, idx]] = vol_today;
                surfaces_future[[s, idx]] = vol_future;
                idx += 1;
            }
        }
    }

    (surfaces_today, surfaces_future)
}
